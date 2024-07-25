#include <opencv2/text/ocr.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <ros/ros.h>
#include <omp.h>
//#include <cpprest/json.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <tesseract/baseapi.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;

bool isNum(const string& str){
    return !str.empty() && all_of(str.begin(), str.end()-1, ::isdigit);
}

void decode(const Mat& scores, const Mat& geometry, float scoreThresh,
            std::vector<RotatedRect>& detections, std::vector<float>& confidences)
{
    detections.clear();
    // Check dimensions
    // CV_Assert(scores.dims == 4, geometry.dims == 4, scores.size[0] == 1,
    //           geometry.size[0] == 1, scores.size[1] == 1, geometry.size[1] == 5,
    //           scores.size[2] == geometry.size[2], scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                           offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
            Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
            // Draw a rotated rectangle
            // 0.5 * (p1 + p3) = center of rectangle
            // Size2f(w,h) = width and height of rectangle
            // -angle*180/pi = Rotation angle in clockwise directions (in degrees)
            RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}
int main(int argc, char** argv){
    ros::init(argc, argv, "test_text_detection");
    ros::NodeHandle nh;
    // Parse command line arguments.
    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
    api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    api->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_LINE);
    api->SetVariable("user_defined_dpi", "300");
    float confThreshold = 0.8;//parser.get<float>("thr");
    float nmsThreshold = 0.5;//parser.get<float>("nms");
    int inpWidth = 800;//parser.get<int>("width");
    int inpHeight = 800;//parser.get<int>("height");
    String model = "/home/nuninu98/Downloads/frozen_east_text_detection.pb";//parser.get<String>("model");

    CV_Assert(!model.empty());
    // Load network.
    Net net = readNetFromTensorflow(model);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    // get network output
    vector<Mat> outs;
    vector<String> outNames(2);
    outNames[0] = "feature_fusion/Conv_7/Sigmoid";
    outNames[1] = "feature_fusion/concat_3";

    static const std::string kWinName = "EAST: An Efficient and Accurate Scene Text Detector";
    namedWindow(kWinName, WINDOW_NORMAL);

    Mat frame, blob;
    string folder = "/home/nuninu98/catkin_ws/src/data_collector/images/d435_front/";
    string name = "1721617423_903701067";
    frame = imread(folder + name + ".jpg");

    //for(int cnt = 0; cnt < 100; cnt++){
        blobFromImage(frame, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
        net.setInput(blob);
        ros::Time tic = ros::Time::now();
        net.forward(outs, outNames);
        cout<<"T1: "<<(ros::Time::now() - tic).toSec()<<endl;
        Mat scores = outs[0];
        Mat geometry = outs[1];
        tic = ros::Time::now();
        vector<RotatedRect> boxes;
        vector<float> confidences;
        vector<vector<RotatedRect>> line_boxes;
        vector<vector<float>> line_confidences;
        decode(scores, geometry, confThreshold, boxes, confidences);
        
        // Apply non-maximum suppression procedure.
        std::vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        Mat frame_copy = frame.clone();
        // Render detections.
        Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            RotatedRect& box = boxes[indices[i]];

            Point2f vertices[4];
            box.points(vertices);
            for (int j = 0; j < 4; ++j)
            {
                vertices[j].x *= ratio.x;
                vertices[j].y *= ratio.y;
            }
            for (int j = 0; j < 4; ++j){
                line(frame_copy, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 3);
            }
                
            vector<Point2f> vert_vec(4);
            for (int j = 0; j < 4; ++j)
            {
                vert_vec[j] = vertices[j];
            }
            Rect rect3 = minAreaRect(vert_vec).boundingRect();
            rect3+= cv::Size(10, 10);
            Mat box_img = frame(rect3);
            api->SetImage(box_img.data, box_img.cols, box_img.rows, 1, box_img.step);
            string text_detected = string(api->GetUTF8Text());
            if(isNum(text_detected)){
                //int room_num = stoi(text_detected);
                cout<<"FOUND: "<<text_detected<<endl;
                rectangle(frame_copy, rect3, cv::Scalar(0, 0, 255), 5);
            }
            else{
                cout<<"DROPPED: "<<text_detected<<endl;
            }
        
        }
        // Put efficiency information.
        //putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        cout<<"T2: "<<(ros::Time::now() - tic).toSec()<<endl;
        
   // }
    imshow(kWinName, frame_copy);
    waitKey(0);
    delete api;
    return 0;

}
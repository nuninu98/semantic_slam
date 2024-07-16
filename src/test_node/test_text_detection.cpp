#include <opencv2/text/ocr.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <ros/ros.h>

//#include <cpprest/json.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <tesseract/baseapi.h>

// using namespace web;
// using namespace std;
// using namespace cv;
// using namespace cv::dnn;
// using namespace cv::text;

// void decode(const Mat &scores, const Mat &geometry, float scoreThresh,
//             vector<RotatedRect> &detections, vector<vector<RotatedRect>> &line_detections,
//             vector<float> &confidences, vector<vector<float>> &box_confidences)
// {
//     detections.clear();
//     confidences.clear();
//     CV_Assert(scores.dims == 4);
//     CV_Assert(geometry.dims == 4);
//     CV_Assert(scores.size[0] == 1);
//     CV_Assert(geometry.size[0] == 1);
//     CV_Assert(scores.size[1] == 1);
//     CV_Assert(geometry.size[1] == 5);
//     CV_Assert(scores.size[2] == geometry.size[2]);
//     CV_Assert(scores.size[3] == geometry.size[3]);

//     const int height = scores.size[2];
//     const int width = scores.size[3];
//     for (int y = 0; y < height; ++y)
//     {
//         vector<RotatedRect> boxes;
//         vector<float> conf;
//         const float *scoresData = scores.ptr<float>(0, 0, y);
//         const float *x0_data = geometry.ptr<float>(0, 0, y);
//         const float *x1_data = geometry.ptr<float>(0, 1, y);
//         const float *x2_data = geometry.ptr<float>(0, 2, y);
//         const float *x3_data = geometry.ptr<float>(0, 3, y);
//         const float *anglesData = geometry.ptr<float>(0, 4, y);
//         for (int x = 0; x < width; ++x)
//         {
//             float score = scoresData[x];
//             if (score < scoreThresh)
//                 continue;

//             // Decode a prediction.
//             // Multiple by 4 because feature maps are 4 time less than input image.
//             float offsetX = x * 4.0f, offsetY = y * 4.0f;
//             float angle = anglesData[x];
//             float cosA = std::cos(angle);
//             float sinA = std::sin(angle);
//             float h = x0_data[x] + x2_data[x];
//             float w = x1_data[x] + x3_data[x];

//             Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
//                            offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
//             Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
//             Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
//             RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);

//             boxes.push_back(r);
//             detections.push_back(r);
//             conf.push_back(score);
//             confidences.push_back(score);

//         }
//         line_detections.push_back(boxes);
//         box_confidences.push_back(conf);
//     }
// }

// struct Triplets
// {
//     long unsigned int idx;
//     vector<RotatedRect> box;
//     vector<int> ind;
// };

// int main(int argc, char** argv){
//     ros::init(argc, argv, "test_text_detection");

//     // Parse command line arguments.
//     Ptr<OCRTesseract> ocr = OCRTesseract::create(NULL, NULL, NULL, 3, 8);

//     float confThreshold = 0.6;//parser.get<float>("thr");
//     float nmsThreshold = 0.5;//parser.get<float>("nms");
//     int inpWidth = 800;//parser.get<int>("width");
//     int inpHeight = 800;//parser.get<int>("height");
//     String model = "/home/nuninu98/Downloads/frozen_east_text_detection.pb";//parser.get<String>("model");
//     String detect = "word";//parser.get<String>("detect");

//     CV_Assert(!model.empty());

//     // Load network.
//     Net net = readNetFromTensorflow(model);

//     // get network output
//     vector<Mat> outs;
//     vector<String> outNames(2);
//     outNames[0] = "feature_fusion/Conv_7/Sigmoid";
//     outNames[1] = "feature_fusion/concat_3";

//     static const std::string kWinName = "EAST: An Efficient and Accurate Scene Text Detector";
//     namedWindow(kWinName, WINDOW_NORMAL);

//     Mat frame, blob;
//     frame = imread("/home/nuninu98/Downloads/iloveimg-converted/IMG_5081.jpg");

//     blobFromImage(frame, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
//     net.setInput(blob);
//     net.forward(outs, outNames);
//     Mat scores = outs[0];
//     Mat geometry = outs[1];

//     vector<RotatedRect> boxes;
//     vector<float> confidences;
//     vector<vector<RotatedRect>> line_boxes;
//     vector<vector<float>> line_confidences;
//     decode(scores, geometry, confThreshold, boxes, line_boxes, confidences, line_confidences);


//     int prev_size = 0;
//     vector<int> indices;
//     vector<Triplets> lines;
//     //applying non maximum suppression
//     for (int k = 0; k < line_boxes.size(); ++k)
//     {
//         int curr_size = line_boxes[k].size();
//         if (curr_size == prev_size)
//             continue;
//         NMSBoxes(line_boxes[k], line_confidences[k], confThreshold, nmsThreshold, indices);
//         lines.push_back({line_boxes[k].size(), line_boxes[k], indices});
//         prev_size = curr_size;
//     }

//     //remove lines with zero bounding boxes
//     int curr_max = 0;
//     long unsigned int curr_idx;
//     vector<RotatedRect> curr_box;
//     vector<int> curr_ind;
//     vector<Triplets> box_indices;
//     for (int i = 0; i < lines.size(); i++)
//     {
//         if (lines[i].idx == 0)
//         {
//             box_indices.push_back({curr_idx, curr_box, curr_ind});
//             curr_max = 0;
//         }
//         else
//         {
//             if (lines[i].idx > curr_max)
//             {
//                 curr_idx = lines[i].idx;
//                 curr_box = lines[i].box;
//                 curr_ind = lines[i].ind;

//                 curr_max = lines[i].idx;
//             }
//         }
//     }

//     vector<json::value> imageText;
//     // Render detections and extract text.
//     for (int i = 0; i < box_indices.size(); i++)
//     {
//         vector<int> indices = box_indices[i].ind;
//         vector<RotatedRect> line_boxes = box_indices[i].box;
//         if (indices.size() != 0)
//         {
//             sort(indices.begin(), indices.end());

//             Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
//             Point2f vertices[4];
//             if (detect == "word")
//             {
//                 for (size_t n = 0; n < indices.size(); ++n)
//                 {
//                     RotatedRect &box = line_boxes[indices[n]];
//                     box.points(vertices);

//                     for (int j = 0; j < 4; ++j)
//                     {
//                         vertices[j].x *= ratio.x;
//                         vertices[j].y *= ratio.y;
//                     }

//                     for (int j = 0; j < 4; ++j)
//                         line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 3);
//                 }
//             }
//             else
//             {
//                 RotatedRect &start_box = line_boxes[indices[0]];
//                 Point2f start_vertices[4];
//                 start_box.points(start_vertices);
//                 vertices[0] = start_vertices[0];
//                 vertices[1] = start_vertices[1];

//                 int last = indices.size() - 1;
//                 RotatedRect &end_box = line_boxes[indices[last]];
//                 Point2f end_vertices[4];
//                 end_box.points(end_vertices);
//                 vertices[2] = end_vertices[2];
//                 vertices[3] = end_vertices[3];

//                 for (int j = 0; j < 4; ++j)
//                 {
//                     vertices[j].x *= ratio.x;
//                     vertices[j].y *= ratio.y;
//                 }
//                 for (int j = 0; j < 4; ++j)
//                     line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 3);
//             }

//             vector<Point2f> vert{{vertices[0].x, vertices[0].y},
//                                  {vertices[1].x, vertices[1].y},
//                                  {vertices[2].x, vertices[2].y},
//                                  {vertices[3].x, vertices[3].y}};

//             // Text recognition
//             RotatedRect new_box = minAreaRect(vert);
//             Rect rect3 = new_box.boundingRect();
//             Mat box_img = frame(rect3);
//             string output;
//             ocr->run(box_img, output);
//             output.erase(remove(output.begin(), output.end(), '\n'), output.end());

//             //create json data
//             json::value text_data;
//             vector<json::value> box_data;
//             for (int m = 0; m < 4; ++m)
//             {
//                 json::value box_coord;
//                 stringstream xstream;
//                 xstream << fixed << setprecision(2) << vertices[m].x;
//                 string x = xstream.str();
//                 box_coord["x"] = json::value::string(x);
//                 stringstream ystream;
//                 ystream << fixed << setprecision(2) << vertices[m].y;
//                 string y = ystream.str();
//                 box_coord["y"] = json::value::string(y);

//                 box_data.push_back(box_coord);
//             }

//             text_data["text"] = json::value::string(output);
//             text_data["boundingBox"] = json::value::array(box_data);
//             imageText.push_back(text_data);
//         }
//     }

//     for(int j = 0; j < imageText.size(); ++j){
//         cout<<imageText[j];
//         break;
//     }
//     // Put efficiency information.
//     vector<double> layersTimes;
//     double freq = getTickFrequency() / 1000;
//     double t = net.getPerfProfile(layersTimes) / freq;
//     string label = format("Inference time: %.2f ms", t);
//     putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
//     imshow(kWinName, frame);
//     waitKey(0);
//     return 0;

// }

//======================================================================
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
    api->SetVariable("user_defined_dpi", "70");
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
    frame = imread("/home/nuninu98/Downloads/iloveimg-converted/IMG_5082.jpg");
    blobFromImage(frame, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
    net.setInput(blob);
    ros::Time tic = ros::Time::now();
    net.forward(outs, outNames);
    cout<<"T1: "<<(ros::Time::now() - tic).toSec()<<endl;

    Mat scores = outs[0];
    Mat geometry = outs[1];

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
        Mat box_img = frame(rect3);
        string output;
        api->SetImage(box_img.data, box_img.cols, box_img.rows, 3, box_img.step);
        string text_detected = string(api->GetUTF8Text());
        if(isNum(text_detected)){
            int room_num = stoi(text_detected);
            cout<<"NUMBER: "<<room_num<<endl;
        }
      
        //ocr->run(box_img, output);
        output.erase(remove(output.begin(), output.end(), '\n'), output.end());
        rectangle(frame_copy, rect3, cv::Scalar(0, 0, 255), 3);

    }
    // Put efficiency information.
    //putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    imshow(kWinName, frame_copy);
    waitKey(0);
    return 0;

}
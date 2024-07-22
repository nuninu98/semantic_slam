#include <semantic_slam/algorithm/ocr.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

using namespace cv;
using namespace dnn;
using namespace std;
OCR* ocr = nullptr;

void imageCallback(const sensor_msgs::ImageConstPtr& image){
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvShare(image, "bgr8");
    Mat frame = cv_rgb_bridge->image;
    if(ocr != nullptr){
        ocr->detect_rec(frame);
    }
    string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    imshow(kWinName, frame);
    waitKey(1);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "test_crnn");
    ros::NodeHandle nh;
    // Parse command line arguments.

    // Load network.
    ocr = new OCR("/home/nuninu98/Downloads/crnn_cs.onnx", "0123456789abcdefghijklmnopqrstuvwxyz");
    ros::Subscriber sub_image = nh.subscribe("/d455/color/image_raw", 1, imageCallback);
    ros::spin();
    // string folder = "/home/nuninu98/catkin_ws/src/data_collector/images/d435_front/";
    // string name = "1721617563_795256138";
    // cv::Mat frame = imread(folder + name + ".jpg");
    // ocr->detect_rec(frame);
    
    // static const string kWinName = "Deep learning object detection in OpenCV";
    // namedWindow(kWinName, WINDOW_NORMAL);
    // imshow(kWinName, frame);
    // waitKey(0);
    // destroyAllWindows();
    return 0;

}
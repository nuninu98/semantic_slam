#include <semantic_slam/algorithm/ocr.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/subscriber.h>
#include <image_transport/image_transport.h>

void compImageCallback(const sensor_msgs::ImageConstPtr& image){
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvCopy(image);
    //cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvCopy(*image, "bgr8");
    cv::Mat img_mat = cv_rgb_bridge->image;
    cv::imshow("TEST COMP" ,img_mat);
    cv::waitKey(1);
    
}

void imageCallback(const sensor_msgs::ImageConstPtr& image){
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvCopy(*image);
    cv::Mat img_mat = cv_rgb_bridge->image;
    cv::imshow("TEST" ,img_mat);
    cv::waitKey(1);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "test_crnn");
    ros::NodeHandle nh;
    image_transport::Subscriber itSub;
    image_transport::ImageTransport it(nh);
    itSub = it.subscribe("/front/aligned_depth_to_color/image_raw/compressed", 1,compImageCallback, image_transport::TransportHints("compressed"));

    ros::Subscriber sub_image_comp = nh.subscribe("/front/aligned_depth_to_color/image_raw/compressed", 1, compImageCallback);
    ros::Subscriber sub_image = nh.subscribe("/front/aligned_depth_to_color/image_raw", 1, imageCallback);
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
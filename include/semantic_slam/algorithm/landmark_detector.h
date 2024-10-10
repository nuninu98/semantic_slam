#ifndef __SEMANTIC_LANDMARK_DETECTOR_HEADER__
#define __SEMANTIC_LANDMARK_DETECTOR_HEADER__

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/callback_queue.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
#include <opencv4/opencv2/core/version.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <fstream>
#include <semantic_slam/data_type/DataType.h>
#include <opencv2/cudaimgproc.hpp>
using namespace std;

const double CONFIDENCE_THRESHOLD = 0.6;
const double NMS_TRESHOLD = 0.4;
class LandmarkDetector{
    private:
        vector<string> class_names_;
        //ros::NodeHandle pnh_;
        mutex lock_camera_;
        cv::dnn::Net network_;
        //vector<string> last_layer_names_;
        
        cv::Mat formatToSquare(const cv::Mat& source);
    
    public:
        //LandmarkDetector();

        LandmarkDetector(const string& onnx, const vector<string>& class_names);

        ~LandmarkDetector();
        
        void detectObjectYOLO(const cv::Mat& rgb_image, const cv::Mat& depth_image, const Eigen::Matrix3f& K, vector<Detection*>& detection_output);


        vector<string> getClassNames() const;
};
#endif
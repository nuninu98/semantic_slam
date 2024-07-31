#ifndef __SEMANTIC_SLAM_HEADER_H__
#define __SEMANTIC_SLAM_HEADER_H__
#include <ros/ros.h>
//#include <include/System.h>
#include "System.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unordered_set>
#include <pcl/common/common.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <boost/functional/hash.hpp>
#include <boost/filesystem.hpp>
#include <omp.h>
#include <algorithm>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/gicp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <semantic_slam/algorithm/landmark_detector.h>
#include <semantic_slam/algorithm/ocr.h>
#include <sensor_msgs/Imu.h>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/Path.h>
using namespace std;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;

class SemanticSLAM{
    private:
        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;
        ORB_SLAM3::System* visual_odom_;
        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> rgb_subscriber_;
        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_subscriber_;
        shared_ptr< message_filters::Synchronizer<sync_pol>> sync_;
        vector<cv::Scalar> colors_;
        LandmarkDetector ld_;
        vector<string> class_names_;

        ros::Subscriber sub_detection_image_;

        ros::Subscriber sub_imu_;

        shared_ptr<OCR> ocr_;

        mutex imu_lock_;
        
        queue<sensor_msgs::Imu> imu_buf_;

        mutex object_lock_;
        queue<pair<ros::Time, vector<Detection>>> obj_detection_buf_;

        void trackingImageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image);
        
        void imuCallback(const sensor_msgs::ImuConstPtr& imu);

        void detectionImageCallback(const sensor_msgs::ImageConstPtr& color_image);
    
        Eigen::Matrix4f optic_in_base_;
        //============Visualization===========
        bool kill_flag_;
        bool thread_killed_;
        
        thread keyframe_thread_;
        mutex keyframe_lock_;
        void keyframeCallback();
        bool keyframe_updated_;
        condition_variable keyframe_cv_;

        tf2_ros::TransformBroadcaster broadcaster_;
        ros::Publisher pub_path_;
        //====================================
    public:
        SemanticSLAM();

        ~SemanticSLAM();
};
#endif
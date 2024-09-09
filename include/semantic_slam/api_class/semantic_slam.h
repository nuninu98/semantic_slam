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
#include <sensor_msgs/PointCloud2.h>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <semantic_slam/data_type/KeyFrame.h>
#include <semantic_slam/data_type/HGraph.h>
using namespace std;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;

class SemanticSLAM{
    private:
        Eigen::Matrix4f OPTIC_TF = (Eigen::Matrix4f()<< 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1).finished();

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;
        ORB_SLAM3::System* visual_odom_;
        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> tracking_color_;
        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> tracking_depth_;
        shared_ptr< message_filters::Synchronizer<sync_pol>> tracking_sync_;

        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> side_color_;
        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> side_depth_;
        shared_ptr< message_filters::Synchronizer<sync_pol>> side_sync_;

        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> front_color_;
        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> front_depth_;
        shared_ptr< message_filters::Synchronizer<sync_pol>> front_sync_;
        //============Test yolov8 seg=============
        ros::Subscriber sub_frontcam_;

        void frontCamCallback(const sensor_msgs::ImageConstPtr& color_img);
        //========================================
        vector<cv::Scalar> colors_;
        shared_ptr<LandmarkDetector> door_detector_;
        shared_ptr<LandmarkDetector> obj_detector_;
        vector<string> class_names_;


        ros::Subscriber sub_imu_;

        shared_ptr<OCR> ocr_;

        mutex imu_lock_;
        
        queue<sensor_msgs::Imu> imu_buf_;

        mutex object_lock_;
        Eigen::Matrix3f K_side_;
        Eigen::Matrix3f K_front_;
        //queue<pair<ros::Time, vector<Detection>>> obj_detection_buf_;
        queue<DetectionGroup> obj_detection_buf_;
        void trackingImageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image);
        
        void imuCallback(const sensor_msgs::ImuConstPtr& imu);

        //void detectionImageCallback(const sensor_msgs::ImageConstPtr& color_image, const shared_ptr<LandmarkDetector>& detector, const Eigen::Matrix4f& sensor_pose);
        void detectionImageCallback(const sensor_msgs::ImageConstPtr& color_image, const sensor_msgs::ImageConstPtr& depth_image, const Eigen::Matrix4f& sensor_pose, const Eigen::Matrix3f& K);
        Eigen::Matrix4f sidecam_in_frontcam_; // optic
        double depth_factor_;

        //============Visualization===========
        bool kill_flag_;
        bool thread_killed_;
        
       

        tf2_ros::TransformBroadcaster broadcaster_;
        //ros::Publisher pub_path_;
        vector<ros::Publisher> pub_floor_path_;
        ros::Publisher pub_object_cloud_;
        ros::Publisher pub_h_graph_;
        ros::Publisher pub_map_cloud_;
        
        void visualizeHGraph(const HGraph& h_graph, visualization_msgs::MarkerArray& output);
        
        //=========Test Floor Plane===========
        ros::Publisher pub_floor_;
        //====================================

        //=============Test new struct========
        condition_variable keyframe_cv_;
        bool kf_updated_;
        unordered_map<size_t, KeyFrame*> kfs_;
        Floor* floor_;
        thread keyframe_thread_;
        mutex keyframe_lock_;
        void keyframeCallback();
        HGraph h_graph_;

        
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        SemanticSLAM();

        ~SemanticSLAM();
};
#endif
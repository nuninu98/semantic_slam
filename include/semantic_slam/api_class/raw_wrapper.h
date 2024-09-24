#ifndef __ORB_SLAM_RAW_WRAPPER_H__
#define __ORB_SLAM_RAW_WRAPPER_H__

#include <ros/ros.h>
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

#include "System.h"
using namespace std;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;

class RawWrapper{
    public: 
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
        Eigen::Matrix4f OPTIC_TF = (Eigen::Matrix4f()<< 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1).finished();

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;
        ORB_SLAM3::System* visual_odom_;
        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> tracking_color_;
        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> tracking_depth_;
        shared_ptr< message_filters::Synchronizer<sync_pol>> tracking_sync_;

        Eigen::Matrix3f K_front_;
        bool kill_flag_;
        bool thread_killed_;
        thread keyframe_thread_;
        mutex keyframe_lock_;
        condition_variable keyframe_cv_;
        bool kf_updated_;
        void keyframeCallback();
        tf2_ros::TransformBroadcaster broadcaster_;

        void trackingImageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image);

        ros::Publisher pub_path_;
    public:
        RawWrapper();

        ~RawWrapper();
};
#endif
#ifndef __SEMANTIC_SLAM_HEADER_H__
#define __SEMANTIC_SLAM_HEADER_H__
#include <ros/ros.h>
#include <include/System.h>
#include <Eigen/Core>
#include <Eigen/Dense>
//#include <orb_semantic_slam/algorithm/gtsam_quadrics/geometry/ConstrainedDualQuadric.h>
// #include <orb_semantic_slam/algorithm/gtsam_quadrics/geometry/BoundingBoxFactor.h>
// #include <orb_semantic_slam/algorithm/gtsam_quadrics/geometry/QuadricCamera.h>
#include <unordered_set>
// #include <gtsam/geometry/Rot3.h>
// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/slam/PriorFactor.h>
// #include <gtsam/slam/BetweenFactor.h>
// #include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/nonlinear/ISAM2.h>
// #include <gtsam/nonlinear/Marginals.h>
// #include <gtsam/geometry/PinholeCamera.h>
// #include <gtsam/inference/Symbol.h>
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

        void imageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image);
        
        void imuCallback(const sensor_msgs::ImuConstPtr& imu);

        void detectionImageCallback(const sensor_msgs::ImageConstPtr& color_image);
    public:
        SemanticSLAM();

        ~SemanticSLAM();
};
#endif
#ifndef __ORB_SLAM_SEMANTIC_DATA_TYPES_HEADER__
#define __ORB_SLAM_SEMANTIC_DATA_TYPES_HEADER__
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <Eigen/StdVector>
#include <deque>
#include <unordered_set>
#include "KeyFrame.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <gtsam_quadrics/geometry/ConstrainedDualQuadric.h>
#include <gtsam_quadrics/geometry/DualConic.h>
#include <gtsam_quadrics/geometry/QuadricCamera.h>
#include <gtsam_quadrics/geometry/BoundingBoxFactor.h>

using namespace std;
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix4f)

    class DetectionGroup;
    class KeyFrame;
    class Object;
    class OCRDetection{
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            OCRDetection();

            OCRDetection(const OCRDetection& ocr);

            OCRDetection(const cv::Rect& roi, const string& content);

            string getContent() const;

            cv::Rect getRoI() const;

            OCRDetection& operator=(const OCRDetection& ocr);

        private:
            string content_;

            cv::Rect roi_;
    };

    class Detection{
        
        public: 
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            Detection();

            Detection(const cv::Rect& roi, const cv::Mat& mask, const string& name);

            ~Detection();   

            cv::Rect getROI_CV() const;
            gtsam_quadrics::AlignedBox2 getROI() const;

            cv::Mat getMask() const;

            string getClassName() const;

            //string getContent() const; // only for room number

            void copyContent(const OCRDetection& ocr_output);

            void calcCentroid(const cv::Mat& color_mat, const cv::Mat& depth_mat, const Eigen::Matrix3f& K);

            //void getCloud(pcl::PointCloud<pcl::PointXYZRGB>& output) const;

            void setDetectionGroup(DetectionGroup* dg);

            const DetectionGroup* getDetectionGroup() const;
        
            Eigen::Vector3f center3D() const;

            void setCorrespondence(Object* obj);

            Object* getCorrespondence() const;
        private:
            Object* matched_obj_ = nullptr;
            cv::Rect roi_;
            cv::Mat mask_;
            string name_;
            //string content_;
            DetectionGroup* dg_;
            Eigen::Vector3f centroid_;
            
    };

    class Object{
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        private:
            size_t id_;
            //Eigen::Vector3f centroid_;
            string name_;
            vector<const Detection*> seens_;
            gtsam_quadrics::ConstrainedDualQuadric Q_;
        public:
            Object();

            Object(const Object& obj);
            
            Object(const string& name, size_t id, const gtsam_quadrics::ConstrainedDualQuadric& Q);

            string getClassName() const;

            // bool getEstBbox(const Eigen::Matrix3f& K, const Eigen::Matrix4f& cam_in_map, cv::Rect& output) const;

            // void getCloud(pcl::PointCloud<pcl::PointXYZRGB>& output) const;
        
            void addDetection(const Detection* det);
        
            void getConnectedKeyFrames(vector<KeyFrame*>& output) const;
        
            // Eigen::Vector3f getCentroid() const;

            void merge(Object* obj);

            // void setCentroid(const Eigen::Vector3f& centroid);

            size_t id() const;

            //=============Dual Quad test======
            gtsam_quadrics::ConstrainedDualQuadric Q() const;

            void setQ(const gtsam_quadrics::ConstrainedDualQuadric& Q);
    };

    

    class DetectionGroup{
        
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW      
            cv::Mat gray_;
        private:
            double stamp_;
            // cv::Mat color_img;
            // cv::Mat depth_img;
            Eigen::Matrix4f sensor_pose_;
            Eigen::Matrix3f K_;
            vector<Detection*> detections_;
            KeyFrame* kf_;
            char sid_;
        public:
            DetectionGroup();

            DetectionGroup(const DetectionGroup& dg);

            DetectionGroup(const Eigen::Matrix4f& sensor_pose, const vector<Detection*>& detections, const Eigen::Matrix3f& K, double stamp, char sid);

            ~DetectionGroup();

            double stamp() const;

            void detections(vector<Detection*>& output) const;

            Eigen::Matrix4f getSensorPose() const;

            Eigen::Matrix3f getIntrinsic() const;

            void setKeyFrame(KeyFrame* kf);

            KeyFrame* getKeyFrame() const;

            char sID() const;

            // cv::Mat getColorImage() const;

            // cv::Mat getDepthImage() const;
    };

    class Floor{
        private:
            KeyFrame* plane_kf_;
            deque<KeyFrame*> kfs_;
            int label_;
            
        public:
            Floor(int label, KeyFrame* kf);
            ~Floor();

            void refine();
            
            void addKeyFrame(KeyFrame* kf);


            bool isInlier(KeyFrame* kf);

            

    };









#endif
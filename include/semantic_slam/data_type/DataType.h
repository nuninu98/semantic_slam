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

using namespace std;
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix4f)
    class DetectionGroup;
    class KeyFrame;
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

            cv::Rect getRoI() const;

            cv::Mat getMask() const;

            string getClassName() const;

            //string getContent() const; // only for room number

            void copyContent(const OCRDetection& ocr_output);

            void generateCloud(const cv::Mat& color_mat, const cv::Mat& depth_mat, const Eigen::Matrix3f& K);

            void getCloud(pcl::PointCloud<pcl::PointXYZRGB>& output) const;

            void setDetectionGroup(DetectionGroup* dg);

            const DetectionGroup* getDetectionGroup() const;
        private:
            cv::Rect roi_;
            cv::Mat mask_;
            string name_;
            //string content_;
            pcl::PointCloud<pcl::PointXYZRGB> cloud_;
            DetectionGroup* dg_;
            
    };

    class Object{
        
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            Object();

            Object(const Object& obj);
            
            Object(const string& name);

            string getClassName() const;

            bool getEstBbox(const Eigen::Matrix3f& K, const Eigen::Matrix4f& cam_in_map, cv::Rect& output) const;

            void getCloud(pcl::PointCloud<pcl::PointXYZRGB>& output) const;
        
            void addDetection(const Detection* det);
        
            void getConnectedKeyFrames(vector<KeyFrame*>& output) const;
        
            Eigen::Vector3f getCentroid() const;

            void merge(Object* obj);
        private:
            string name_;
            vector<const Detection*> seens_;
    };

    

    class DetectionGroup{
        
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW      
        private:
            double stamp_;
            cv::Mat color_img;
            cv::Mat depth_img;
            Eigen::Matrix4f sensor_pose_;
            Eigen::Matrix3f K_;
            vector<Detection> detections_;
            KeyFrame* kf_;
        public:
            DetectionGroup();

            DetectionGroup(const DetectionGroup& dg);

            DetectionGroup(const cv::Mat& color, const cv::Mat& depth, const Eigen::Matrix4f& sensor_pose,
            const vector<Detection>& detections, const Eigen::Matrix3f& K, double stamp);

            ~DetectionGroup();

            double stamp() const;

            void detections(vector<const Detection*>& output) const;

            Eigen::Matrix4f getSensorPose() const;

            Eigen::Matrix3f getIntrinsic() const;

            void setKeyFrame(KeyFrame* kf);

            KeyFrame* getKeyFrame() const;
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
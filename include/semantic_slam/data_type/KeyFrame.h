#ifndef __SLAM_KEYFRAME_H__
#define __SLAM_KEYFRAME_H__

#include <semantic_slam/data_type/DataType.h>
#include <vector>
#include "DBoW2/BowVector.h"
class Floor;
class DetectionGroup;
class Detection;
using namespace std;
class KeyFrame{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        DBoW2::BowVector bow_vec;
        //===Visualization===
        // cv::Mat color_;
        // cv::Mat depth_;
        //===================
    private:
        Floor* floor_;
        size_t id_;
        Eigen::Matrix4f pose_;
        Eigen::Matrix4f odom_pose_;
        vector<DetectionGroup> detections_;
        
    public:
        Eigen::Matrix4f getPose() const;

        Eigen::Matrix4f getOdomPose() const;
        KeyFrame(size_t id, const Eigen::Matrix4f& pose);

        KeyFrame(const KeyFrame& k);

        Floor* getFloor();

        void setFloor(Floor* floor);

        void setDetection(const vector<DetectionGroup>& dets);
            

        void getDetectionGroup(vector<const DetectionGroup*>& output) const;

        void getDetections(vector<Detection*>& output) const;

        void setPose(const Eigen::Matrix4f& pose);
        
        size_t id() const;

        void printDets() const;

        int numDetectionsWithName(const string& name);
};

#endif
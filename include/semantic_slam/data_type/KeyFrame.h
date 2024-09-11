#ifndef __SLAM_KEYFRAME_H__
#define __SLAM_KEYFRAME_H__

#include <semantic_slam/data_type/DataType.h>
#include <vector>
class Floor;
class DetectionGroup;
using namespace std;
class KeyFrame{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
            

        void getDetection(vector<const DetectionGroup*>& output) const;

        void setPose(const Eigen::Matrix4f& pose);
        
        size_t id() const;
};

#endif
#include <semantic_slam/data_type/KeyFrame.h>

KeyFrame::KeyFrame(size_t id, const Eigen::Matrix4f& odom_pose) : id_(id), odom_pose_(odom_pose), pose_(odom_pose), floor_(nullptr){

}

KeyFrame::KeyFrame(const KeyFrame& k): pose_(k.pose_), id_(k.id_), floor_(k.floor_), odom_pose_(k.odom_pose_){

}

Eigen::Matrix4f KeyFrame::getPose() const{
    return pose_;
}

Eigen::Matrix4f KeyFrame::getOdomPose() const{
    return odom_pose_;
}

Floor* KeyFrame::getFloor(){
    return floor_;
}

void KeyFrame::setFloor(Floor* floor){
    floor_ = floor;
}

void KeyFrame::setDetection(const vector<DetectionGroup>& dets){
    detections_ = dets;
    for(auto& elem : detections_){
        if(elem.sID() == 'F'){
            // depth_ = elem.getDepthImage().clone();
            // color_ = elem.getColorImage().clone();
        }
        elem.setKeyFrame(this);
    }
  
}

void KeyFrame::getDetectionGroup(vector<const DetectionGroup*>& output) const{
    output.clear();
    for(int i = 0; i < detections_.size(); ++i){
        output.push_back(&detections_[i]);
    }
}

void KeyFrame::getDetections(vector<Detection*>& output) const{
    for(auto& elem : detections_){
        vector<Detection*> dets;
        elem.detections(dets);
        for(auto& det : dets){
            output.push_back(det);
        }
    }
}

void KeyFrame::setPose(const Eigen::Matrix4f& pose){
    pose_ = pose;
}

size_t KeyFrame::id() const{
    return id_;
}

void KeyFrame::printDets() const{
    cout<<"KF "<<id()<<": ";
    for(auto& dg : detections_){
        vector<Detection*> dets;
        dg.detections(dets);
        for(auto& det : dets){
            cout<<(det->getClassName())<<" ";
        }
    }
    cout<<endl;
}

int KeyFrame::numDetectionsWithName(const string& name){
    int answer = 0;
    for(auto& dg : detections_){
        vector<Detection*> dets;
        dg.detections(dets);
        for(auto& det : dets){
            if(name == det->getClassName()){
                answer++;
            }
        }
    }
    return answer;
}
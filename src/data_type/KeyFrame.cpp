#include <semantic_slam/data_type/KeyFrame.h>

KeyFrame::KeyFrame(size_t id, const Eigen::Matrix4f& pose) : id_(id), pose_(pose), floor_(nullptr){

}

KeyFrame::KeyFrame(const KeyFrame& k): pose_(k.pose_), id_(k.id_), floor_(k.floor_){

}

Eigen::Matrix4f KeyFrame::getPose() const{
    return pose_;
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
        elem.setKeyFrame(this);
    }
  
}

void KeyFrame::getDetection(vector<const DetectionGroup*>& output) const{
    output.clear();
    for(int i = 0; i < detections_.size(); ++i){
        output.push_back(&detections_[i]);
    }
}

size_t KeyFrame::id() const{
    return id_;
}
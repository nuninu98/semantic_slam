#include <semantic_slam/data_types/data_types.h>

Detection::Detection(){

}

Detection::Detection(const cv::Rect& roi, const cv::Mat& mask, const size_t& id): roi_(roi), mask_(mask), id_(id){

}

Detection::~Detection(){

}

cv::Rect Detection::getRoI() const{
    return roi_;
}

cv::Mat Detection::getMask() const{
    return mask_;
}

size_t Detection::getClassID() const{
    return id_;
}

OCRDetection::OCRDetection(){
    
}

OCRDetection::OCRDetection(const cv::Rect& roi, const size_t& id){
    roi_ = roi;
    id_ = id;
}
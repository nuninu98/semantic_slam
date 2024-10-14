#include <semantic_slam/algorithm/loop_matcher.h>

LoopMatcher::LoopMatcher(){

}

LoopMatcher::~LoopMatcher(){

}

bool LoopMatcher::match(KeyFrame* kf, const unordered_map<Object*,float>& object_uscores){
    vector<Detection*> dets;
    kf->getDetections(dets);
    
}
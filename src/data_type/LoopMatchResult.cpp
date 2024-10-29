#include <semantic_slam/data_type/LoopMatchResult.h>

LoopMatchResult::LoopMatchResult(): query(0), target(0), drift(Eigen::Matrix4f::Identity()), score(0.0){

}

LoopMatchResult::~LoopMatchResult(){
    
}

LoopMatchResult::LoopMatchResult(size_t query, size_t target, const Eigen::Matrix4f& drift, const vector<pair<Detection*, Object*>>& obj_matches, float score)
: query(query), target(target), drift(drift), score(score){

}

LoopMatchResult::LoopMatchResult(const LoopMatchResult& lr): query(lr.query), target(lr.target), drift(lr.drift), score(lr.score),
object_matches(lr.object_matches){

}
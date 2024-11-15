#ifndef __SEMANTIC_SLAM_LOOP_MATCH_RESULT_H__
#define __SEMANTIC_SLAM_LOOP_MATCH_RESULT_H__
#include <semantic_slam/data_type/DataType.h>
class Detection;
class KeyFrame;

struct LoopMatchResult{
    size_t query;
    size_t target;
    Eigen::Matrix4f drift;
    vector<pair<Detection*, Object*>> object_matches;
    Object* unique_obj;
    float score;

    LoopMatchResult();

    ~LoopMatchResult();

    LoopMatchResult(size_t query, size_t target, const Eigen::Matrix4f& drift, const vector<pair<Detection*, Object*>>& obj_matches, float score, Object* u_obj);

    LoopMatchResult(const LoopMatchResult& lr);
};
#endif 
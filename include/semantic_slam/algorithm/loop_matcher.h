#ifndef __LOOP_MATCHER_H__
#define __LOOP_MATCHER_H__
#include <semantic_slam/data_type/DataType.h>
#include <ortools/base/logging.h>
#include <ortools/linear_solver/linear_solver.h>
#include <ortools/sat/cp_model.h>
#include <ortools/base/version.h>
#include <ortools/sat/cp_model.pb.h>
#include <ortools/sat/cp_model_solver.h>
#include <ros/ros.h>
#include <boost/filesystem.hpp>
using namespace std;
using namespace gtsam::symbol_shorthand;

class LoopMatcher{
    public:
        LoopMatcher();

        ~LoopMatcher();

        bool match(KeyFrame* qkf, KeyFrame* tkf, const vector<pair<Object*,float>>& object_uscores, Eigen::Matrix4f& Ttq_output, vector<pair<Detection*, Object*>>& corr_output);
};

#endif
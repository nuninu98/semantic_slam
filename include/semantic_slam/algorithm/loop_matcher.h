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
#include <semantic_slam/data_type/LoopMatchResult.h>
#include <semantic_slam/data_type/HGraph.h>
using namespace std;
using namespace gtsam::symbol_shorthand;

class LoopMatcher{
    private:
        mutex lock_;

        void extractiVisibles(const vector<Object*> objects, const Eigen::Matrix3f& K, const Eigen::Matrix4d& pose, unordered_map<Object*, gtsam_quadrics::AlignedBox2>& output);

        bool matchStep1(KeyFrame* qkf, KeyFrame* tkf, HGraph& h_graph, Eigen::Matrix4d& opt_pose, vector<pair<Detection*, Object*>>& unique_matches);

        bool matchStep2(KeyFrame* qkf, KeyFrame* tkf, HGraph& h_graph, const vector<pair<Detection*, Object*>>& unique_matches, Eigen::Matrix4d& opt_pose, double& score);
    
        bool patternMatched( unordered_map<Object*, gtsam_quadrics::AlignedBox2>& visible1, unordered_map<Object*, gtsam_quadrics::AlignedBox2>& visible2);
    
    public:
        LoopMatcher();

        ~LoopMatcher();

        // bool match(KeyFrame* qkf, KeyFrame* tkf, const vector<pair<Object*,float>>& object_uscores, Eigen::Matrix4f& Ttq_output, vector<pair<Detection*, Object*>>& corr_output);

        bool match2(KeyFrame* qkf, KeyFrame* tkf, const vector<pair<Object*,float>>& object_uscores, LoopMatchResult& output);

        bool match3(KeyFrame* qkf, KeyFrame* tkf, HGraph& h_graph, LoopMatchResult& output);
};

#endif
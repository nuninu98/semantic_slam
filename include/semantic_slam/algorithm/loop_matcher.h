#ifndef __LOOP_MATCHER_H__
#define __LOOP_MATCHER_H__
#include <semantic_slam/data_type/DataType.h>
#include <ortools/base/logging.h>
#include <ortools/linear_solver/linear_solver.h>
using namespace std;
using namespace gtsam::symbol_shorthand;

class LoopMatcher{
    public:
        LoopMatcher();

        ~LoopMatcher();

        bool match(KeyFrame* kf, const vector<pair<Object*,float>>& object_uscores);
};

#endif
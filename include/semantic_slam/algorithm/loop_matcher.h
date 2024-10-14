#ifndef __LOOP_MATCHER_H__
#define __LOOP_MATCHER_H__
#include <semantic_slam/data_type/DataType.h>
using namespace std;

class LoopMatcher{
    public:
        LoopMatcher();

        ~LoopMatcher();

        bool match(KeyFrame* kf, const unordered_map<Object*,float>& object_uscores);
};

#endif
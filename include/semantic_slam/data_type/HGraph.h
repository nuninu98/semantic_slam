#ifndef __HGRAPH_H__
#define __HGRAPH_H__

#include <semantic_slam/data_type/DataType.h>
#include "LoopQuery.h"

using namespace std;
using namespace gtsam::symbol_shorthand;


class Floor;
class Object;

class HGraph{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
        mutex lock_;
        unordered_map<Floor*, unordered_map<string, vector<Object*>>> fl_name_objs_;
        unordered_map<size_t, KeyFrame*> kfs_;
    public:
        HGraph();

        HGraph(const HGraph& g);

        ~HGraph();

        void insert(Floor* floor, Object* obj = nullptr);

        void refineObject();

        vector<Object*> getObjects(Floor* floor, string obj_name="");

        vector<Object*> getEveryObjects();

        void getMatchedKFs(KeyFrame* kf, unordered_map<KeyFrame*, float>& kf_scores);
    
        vector<Floor*> floors();

        void updatePoses(const gtsam::Values& opt_stats);
    
        float getUScore(Floor* floor, string class_name);
    
        KeyFrame* getKeyFrame(size_t id);

        void insert(KeyFrame* kf);
    };



#endif
#ifndef __HGRAPH_H__
#define __HGRAPH_H__

#include "DataType.h"

using namespace std;

class Floor;
class Object;

class HGraph{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
        unordered_map<Floor*, unordered_map<string, vector<Object*>>> fl_name_objs_;
    public:
        HGraph();

        HGraph(const HGraph& g);

        ~HGraph();

        void insert(Floor* floor, Object* obj = nullptr);

        void refineObject();

        vector<Object*> getObjects(Floor* floor, string obj_name);

        vector<Object*> getEveryObjects() const;

        void getMatchedKFs(KeyFrame* kf, unordered_map<KeyFrame*, float>& kf_scores);
    
        vector<Floor*> floors() const;
    };



#endif
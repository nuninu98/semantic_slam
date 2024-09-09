#include <semantic_slam/data_type/HGraph.h>

using namespace std;

HGraph::HGraph(){

}

HGraph::HGraph(const HGraph& g): fl_name_objs_(g.fl_name_objs_){
    
}


HGraph::~HGraph(){
    for(const auto& fl_name : fl_name_objs_){
        delete fl_name.first;
        for(const auto& name_objs : fl_name.second){
            for(const auto& obj : name_objs.second){
                delete obj;
            }
        }
    }
}

void HGraph::insert(Floor* floor, Object* obj){
    if(fl_name_objs_.find(floor) == fl_name_objs_.end()){
        fl_name_objs_.insert(make_pair(floor, unordered_map<string, vector<Object*>>()));
    }
    if(obj != nullptr){
        if(fl_name_objs_[floor].find(obj->getClassName()) == fl_name_objs_[floor].end()){
            fl_name_objs_[floor].insert(make_pair(obj->getClassName(), vector<Object*>()));
        }
        fl_name_objs_[floor][obj->getClassName()].push_back(obj);
    }
    
}

void HGraph::refineObject(){
    for(auto& fl_map : fl_name_objs_){
        for(auto& name_objs: fl_map.second){
            vector<Object*> refined;
            for(int i = 0; i < name_objs.second.size(); ++i){
                Object* obj1 = name_objs.second[i];
                Eigen::Vector3f p1 = obj1->getCentroid();
                Object* to_merge = nullptr;
                float dist_min = 1.0;
                for(const auto& obj2 : refined){
                    Eigen::Vector3f p2 = obj2->getCentroid();
                    if((p1 - p2).norm() < dist_min){
                        to_merge = obj2;
                        dist_min = (p1 - p2).norm();
                    }
                    
                }
                if(to_merge == nullptr){
                    refined.push_back(obj1);
                }
                else{ //merge
                    to_merge->merge(obj1);
                    delete obj1;
                }
            }
            name_objs.second = refined;
        }
    }

    // for(auto& fo_pair: h_graph_){
    //     vector<Object*> refined;
    //     for(int i = 0; i < fo_pair.second.size(); ++i){
    //         Object* obj1 = fo_pair.second[i];
    //         Eigen::Vector3f p1 = obj1->getCentroid();
    //         Object* to_merge = nullptr;
    //         float dist_min = 1.0;
    //         for(const auto& obj2 : refined){
    //             if(obj1->getClassName() == obj2->getClassName()){
    //                 Eigen::Vector3f p2 = obj2->getCentroid();
    //                 if((p1 - p2).norm() < dist_min){
    //                     to_merge = obj2;
    //                     dist_min = (p1 - p2).norm();
    //                 }
    //             }
    //         }
    //         if(to_merge == nullptr){
    //             refined.push_back(obj1);
    //         }
    //         else{ //merge
    //             to_merge->merge(obj1);
    //             delete obj1;
    //         }
    //     }
    //     fo_pair.second = refined;
    // }
}

vector<Object*> HGraph::getObjects(Floor* floor, string obj_name){
    vector<Object*> emp;
    if(fl_name_objs_.find(floor) == fl_name_objs_.end()){
        return emp;
    }
    if(fl_name_objs_[floor].find(obj_name) == fl_name_objs_[floor].end()){
        return emp;
    }
    return fl_name_objs_[floor][obj_name];
}

vector<Object*> HGraph::getEveryObjects() const{
    vector<Object*> tmp;
    for(const auto& fl_name : fl_name_objs_){
        for(const auto& name_objs : fl_name.second){
            for(const auto& obj : name_objs.second){
                tmp.push_back(obj);
            }
        }
    }
    return tmp;
}

void HGraph::getMatchedKFs(KeyFrame* kf, unordered_map<KeyFrame*, float>& kf_scores){
    Floor* kf_floor = kf->getFloor();
    if(fl_name_objs_.find(kf_floor) == fl_name_objs_.end()){
        return;
    }
    auto name_objs = fl_name_objs_[kf_floor];
    vector<const DetectionGroup*> kf_dgs;
    kf->getDetection(kf_dgs);
    for(const auto& dg : kf_dgs){
        vector<const Detection*> dets;
        dg->detections(dets);
        for(const auto& det : dets){
            string name = det->getClassName();
            if(name_objs.find(name) == name_objs.end()){
                continue;
            }
            for(const auto& obj : name_objs[name]){
                vector<KeyFrame*> conns;
                obj->getConnectedKeyFrames(conns);
                for(const auto& pkf : conns){
                    unsigned long id1 = max(pkf->id(), kf->id());
                    unsigned long id2 = min(pkf->id(), kf->id());
                    if(id1 - id2 < 500){
                        continue;
                    }
                    if(kf_scores.find(pkf) == kf_scores.end()){
                        kf_scores.insert(make_pair(pkf, 0.0));
                    }
                    kf_scores[pkf] += 1.0 / name_objs[name].size();
                }
            }
        }
    }

}

vector<Floor*> HGraph::floors() const{
    vector<Floor*> output;
    for(const auto& elem : fl_name_objs_){
        output.push_back(elem.first);
    }
    return output;
}

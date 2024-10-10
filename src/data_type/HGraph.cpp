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
                Eigen::Vector3d p1 = obj1->Q().centroid();
                Object* to_merge = nullptr;
                double dist_min = 0.5;
                for(const auto& obj2 : refined){
                    Eigen::Vector3d p2 = obj2->Q().centroid();
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
}

vector<Object*> HGraph::getObjects(Floor* floor, string obj_name){
    vector<Object*> emp;
    if(fl_name_objs_.find(floor) == fl_name_objs_.end()){
        return emp;
    }
    
    if(obj_name.empty()){
        auto objs = fl_name_objs_[floor];
        for(const auto& name_objs : objs){
            for(const auto obj : name_objs.second){
                emp.push_back(obj);
            }
        }
        return emp;
    }
    else if(fl_name_objs_[floor].find(obj_name) == fl_name_objs_[floor].end()){
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

void HGraph::getMatchedKFs(KeyFrame* kf, unordered_map<KeyFrame*, float>& kf_scores, unordered_map<string, float>& obj_score){
    Floor* kf_floor = kf->getFloor();
    if(fl_name_objs_.find(kf_floor) == fl_name_objs_.end()){
        return;
    }
    auto name_objs = fl_name_objs_[kf_floor];
    vector<Detection*> kf_dets;
    kf->getDetections(kf_dets);
    for(const auto& det : kf_dets){
        string name = det->getClassName();
        if(name_objs.find(name) == name_objs.end()){
            continue;
        }
        
        if(obj_score.find(name) == obj_score.end()){
            obj_score.insert(make_pair(name, 1.0 / name_objs[name].size()));
        }
        for(const auto& obj : name_objs[name]){
            vector<KeyFrame*> conns;
            obj->getConnectedKeyFrames(conns);
            for(auto& ckf : conns){
                if(kf_scores.find(ckf) == kf_scores.end()){
                    kf_scores.insert(make_pair(ckf, 0.0));
                }
            }
        }
    }
    unordered_set<string> used_classes;
    for(const auto& det : kf_dets){
        if(used_classes.find(det->getClassName()) == used_classes.end()){
            int N_qkf = kf->numDetectionsWithName(det->getClassName());
            for(auto& ks_pair :kf_scores){
                int N_tkf = ks_pair.first->numDetectionsWithName(det->getClassName());
                int cnt = min(N_tkf, N_qkf);
                ks_pair.second += cnt * (1.0 / name_objs[det->getClassName()].size());
            }
            used_classes.insert(det->getClassName());
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

void HGraph::updateObjectPoses(const gtsam::Values& opt_stats){
    for(const auto& fl_name : fl_name_objs_){
        for(const auto& name_objs : fl_name.second){
            for(auto& obj : name_objs.second){
                if(opt_stats.exists(O(obj->id()))){
                    obj->setQ(opt_stats.at<gtsam_quadrics::ConstrainedDualQuadric>(O(obj->id())));
                    //obj->setCentroid(opt_stats.at<gtsam::Point3>(O(obj->id())).cast<float>());
                }
            }
        }
    }
}

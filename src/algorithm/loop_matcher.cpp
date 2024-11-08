#include <semantic_slam/algorithm/loop_matcher.h>

LoopMatcher::LoopMatcher(){

}

LoopMatcher::~LoopMatcher(){

}



bool LoopMatcher::match2(KeyFrame* qkf, KeyFrame* tkf, const vector<pair<Object*,float>>& object_uscores, LoopMatchResult& output){
    unique_lock<mutex> lock(lock_);
    vector<Detection*> dets;
    qkf->getDetections(dets);
    gtsam::NonlinearFactorGraph base_graph;
    gtsam::Values base_init;
    for(const auto& elem : object_uscores){
        base_init.insert(O(elem.first->id()), elem.first->Q());
        Eigen::VectorXd fix_noise = Eigen::VectorXd::Ones(9) * 1.0e-7; // to fix the object
        auto gtsam_noise = gtsam::noiseModel::Diagonal::Sigmas(fix_noise);
        gtsam::PriorFactor<gtsam_quadrics::ConstrainedDualQuadric> fix_factor(O(elem.first->id()), elem.first->Q(), gtsam_noise);
        base_graph.add(fix_factor);
    }
    int N = 10;
    double last_cost = 1.0e9;
    vector<pair<cv::Mat, float>> test_imgs;
    ros::Time begin = ros::Time::now();
    Eigen::Matrix4d opt_pose = qkf->getPose().cast<double>();
    vector<pair<int, int>> corrs;
    bool reliable = false;

    operations_research::sat::CpModelBuilder cp_model;
    vector<vector<operations_research::sat::BoolVar>> x(dets.size(), vector<operations_research::sat::BoolVar>(object_uscores.size()));
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            x[i][j] = cp_model.NewBoolVar();
        }
    }
    
    for(int j = 0; j < x[0].size(); ++j){
        vector<operations_research::sat::BoolVar> obj_const;
        for(int i = 0; i < x.size(); ++i){
            obj_const.push_back(x[i][j]);
        }
        cp_model.AddAtMostOne(obj_const);
    }
    for(int i = 0; i < x.size(); ++i){
        cp_model.AddExactlyOne(x[i]);
    }

    vector<vector<float>> costs(dets.size(), vector<float>(object_uscores.size()));
    for(int r = 0; r < costs.size(); ++r){
        for(int c = 0; c < costs[0].size(); ++ c){
            if(dets[r]->getClassName() != object_uscores[c].first->getClassName()){
                costs[r][c] = 1.0e9;
            }
            else{
                const DetectionGroup* dg = dets[r]->getDetectionGroup();
                Eigen::Matrix3f K = dg->getIntrinsic();
                gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
                gtsam_quadrics::QuadricCamera qcam;
                Eigen::Matrix4d Twc = opt_pose * dg->getSensorPose().cast<double>();
                if(object_uscores[c].first->Q().isBehind(gtsam::Pose3(Twc)) || object_uscores[c].first->Q().contains(gtsam::Pose3(Twc))){
                    costs[r][c] = 1.0e9;
                    continue;
                }
                gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(object_uscores[c].first->Q(), gtsam::Pose3(Twc), K_gtsam).bounds();
                gtsam_quadrics::AlignedBox2 bbox_act = dets[r]->getROI();
                float dist = (bbox_est.center() - bbox_act.center()).norm();
                double A1 = bbox_est.width() * bbox_est.height();
                double A2 = bbox_act.width() * bbox_act.height();
                costs[r][c] = dist * (abs(bbox_est.width() - bbox_act.width()) + abs(bbox_est.height() - bbox_act.height())); // object_uscores[c].second;
                // if(isnan(costs[r][c])){
                //     cout<<"NAN!"<<endl;
                //     cout<<"DST: "<<dist<<endl;
                //     cout<<"SCORE: "<<object_uscores[c].second<<endl;
                // }
            }
        }
    }
    // cout<<"COSTMAT: "<<endl;
    // for(int r = 0; r < costs.size(); ++r){
    //     for(int c = 0; c < costs[0].size(); ++ c){
    //         cout<<costs[r][c]<<" ";
    //     }
    //     cout<<endl;
    // }
    //==========SAT SOLVER==========
    // operations_research::sat::CpModelBuilder cp_model;
    cp_model.ClearObjective();
    //cp_model.ClearAssumptions();
    operations_research::sat::DoubleLinearExpr total_cost;
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            //total_cost += (x[i][j] * costs[i][j]);
            total_cost += costs[i][j] * operations_research::sat::DoubleLinearExpr(x[i][j]);
        }
    }
    cp_model.Minimize(total_cost);
    operations_research::sat::CpSolverResponse result = operations_research::sat::Solve(cp_model.Build());
    if(result.status() == operations_research::sat::CpSolverStatus::INFEASIBLE || result.status() != operations_research::sat::CpSolverStatus::OPTIMAL){
        return false;
    }
        
    double result_cost = result.objective_value();
    if(result_cost < 0.1){ // temporarily block error. 
        return false;
    }
    if((result_cost > 1000.0 && abs(last_cost - result_cost) < 100.0) || result_cost > 1.0e8){ // early drop
        return false;
    }  
    // cout<<"ITER "<<iter<<" COST: "<<result_cost<<endl;

    last_cost = result_cost;
    Eigen::VectorXd last_svals;
    bool is_svd_checked = false;
    for(int iter = 0; iter < N; ++iter){
        corrs.clear();
        cv::Mat dg_gray = dets[0]->getDetectionGroup()->gray_.clone();
        cv::Mat gray_color;
        cv::cvtColor(dg_gray, gray_color, cv::COLOR_GRAY2BGR);
        gtsam::NonlinearFactorGraph graph(base_graph);
        gtsam::Values init(base_init);
        init.insert(X(qkf->id()), gtsam::Pose3(opt_pose));
        //N(Objects) >= N(Dets)

        vector<gtsam::Point2> visible_centers;        
        for(int j = 0; j < x[0].size(); ++j){
            bool is_coupled = false;
            for(int i = 0; i < x.size(); ++i){
                const DetectionGroup* dg = dets[i]->getDetectionGroup();
                Eigen::Matrix3f K = dg->getIntrinsic();
                gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
                gtsam_quadrics::QuadricCamera qcam;
                Eigen::Matrix4d Twc = opt_pose * dg->getSensorPose().cast<double>();
                if(object_uscores[j].first->Q().isBehind(gtsam::Pose3(Twc)) || object_uscores[j].first->Q().contains(gtsam::Pose3(Twc))){
                    continue;
                }
                gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(object_uscores[j].first->Q(), gtsam::Pose3(Twc), K_gtsam).bounds();
                cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
                if(operations_research::sat::SolutionBooleanValue(result, x[i][j])){   
                    is_coupled = true;            
                    double A1 = bbox_est.width() * bbox_est.height();
                    double A2 = dets[i]->getROI().width() * dets[i]->getROI().height();
                    double cost = (bbox_est.center() - dets[i]->getROI().center()).norm() * abs(A1- A2)/A2;
                    corrs.push_back(make_pair(i, j));
                    cv::rectangle(gray_color, est_cv, cv::Scalar(0, 255, 0)); //green. est detection
                    cv::putText(gray_color, to_string(corrs.size()), est_cv.tl(), 1, 2, cv::Scalar(0, 255, 0));

                    cv::rectangle(gray_color, dets[i]->getROI_CV(), cv::Scalar(0, 0, 255)); //red. actual detection
                    cv::putText(gray_color, to_string(corrs.size()), dets[i]->getROI_CV().tl(), 1, 2, cv::Scalar(0, 0, 255));
                    break;
                }
                visible_centers.push_back(bbox_est.center());
            }
            if(!is_coupled){
                const DetectionGroup* dg = dets[0]->getDetectionGroup();
                Eigen::Matrix3f K = dg->getIntrinsic();
                gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
                gtsam_quadrics::QuadricCamera qcam;
                Eigen::Matrix4d Twc = opt_pose * dg->getSensorPose().cast<double>();
                if(object_uscores[j].first->Q().isBehind(gtsam::Pose3(Twc)) || object_uscores[j].first->Q().contains(gtsam::Pose3(Twc))){
                    continue;
                }
                gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(object_uscores[j].first->Q(), gtsam::Pose3(Twc), K_gtsam).bounds();
                cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
                cv::rectangle(gray_color, est_cv, cv::Scalar(255, 0, 0)); //blue. est detection
            }
        }
        for(const auto& cor : corrs){
            const DetectionGroup* dg = dets[cor.first]->getDetectionGroup();
            Eigen::Matrix3f K = dg->getIntrinsic();
            gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
            gtsam::Vector4 bbox_noise_vec = gtsam::Vector4(10.0, 10.0, 10.0, 10.0);
            auto bbox_noise = gtsam::noiseModel::Diagonal::Sigmas(bbox_noise_vec);
            gtsam_quadrics::BoundingBoxFactor bbf(dets[cor.first]->getROI(), K_gtsam, X(qkf->id()), O(object_uscores[cor.second].first->id()), bbox_noise);
            graph.add(bbf);
        }
        gtsam::LevenbergMarquardtOptimizer optim(graph, init);
        cout<<"ITER "<<iter<<" GTSAMERR: "<<optim.error()<<endl;
        gtsam::Values opt = optim.optimize();
        Eigen::Matrix4d calc_pose = opt.at<gtsam::Pose3>(X(qkf->id())).matrix();
        opt_pose = calc_pose;

        if(!is_svd_checked){
            Eigen::MatrixXd dist_mat = Eigen::MatrixXd::Zero(visible_centers.size(), visible_centers.size());
            Eigen::MatrixXd dist_mat_cpy = dist_mat;
            for(size_t r = 0; r < visible_centers.size(); ++r){
                for(size_t c = r; c < visible_centers.size(); ++c){
                    dist_mat(r, c) = (visible_centers[r] - visible_centers[c]).norm();
                    dist_mat(c, r) = (visible_centers[r] - visible_centers[c]).norm();
                }
            }
            dist_mat.rowwise().normalize();
            if(dist_mat.hasNaN()){
                return false;
            }
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(dist_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::VectorXd singular_vals = svd.singularValues();
            if(last_svals.size() != 0){
                size_t min_size = min(last_svals.size(), singular_vals.size());
                Eigen::VectorXd crop_last = last_svals.block(0, 0, min_size, 1);
                Eigen::VectorXd crop_sval = singular_vals.block(0, 0, min_size, 1);
                double err = (crop_last - crop_sval).norm();
                cout<<"Singval ERR: "<<err<<endl;
                if(err > 0.6){
                    return false;
                }
                is_svd_checked = true;
                test_imgs.push_back({gray_color, err});
            }
            last_svals = singular_vals; 
        }
               
        // test_imgs.push_back({gray_color, result_cost});
        //==========TODO==============
        //Loop Query Modify (query, vector of targets)
        //============================
        
        if(abs(optim.error() - last_cost) < 1.0e-4 && iter > 3){ //&& last_cost < 20.0 && iter > 5
            reliable = last_cost < 20.0;
            break;
        }
        last_cost = optim.error();
        
    }
    if(!reliable){
        return false;
    }
    for(auto& corr : corrs){
        cout<<"det prev: "<<(dets[corr.first]->getCorrespondence() == nullptr ? "NULL" : to_string(dets[corr.first]->getCorrespondence()->id()))<<endl;
        cout<<"match corr: "<<(object_uscores[corr.second].first->id())<<endl;
        //corr_output.push_back({dets[corr.first], object_uscores[corr.second].first});
        output.object_matches.push_back({dets[corr.first], object_uscores[corr.second].first});
    }

    string folder = "/home/nuninu98/match_test/"+to_string(qkf->id())+"/";

    if(!boost::filesystem::exists(folder)){
        boost::filesystem::create_directories(folder);
    }
    for(int i = 0; i < test_imgs.size(); ++i){
        cv::imwrite(folder + to_string(tkf->id())+"_"+to_string(i)+"_"+to_string(test_imgs[i].second)+".png", test_imgs[i].first);
    }
    output.drift = tkf->getPose().inverse() * opt_pose.cast<float>();
    output.score = last_cost;
    output.query = qkf->id();
    output.target = tkf->id();
    // cout<<"LOOP "<<qkf->id()<<" <-> "<<tkf->id()<<endl;
    // cout<<"BEF OPT: \n"<<qkf->getPose()<<endl;
    // cout<<"AFT OPT: \n"<<opt_pose<<endl;
    cout<<"-----"<<endl;
    return true;
}

bool LoopMatcher::matchStep1(KeyFrame* qkf, KeyFrame* tkf, HGraph& h_graph, Eigen::Matrix4d& opt_pose, vector<pair<Detection*, Object*>>& unique_matches){
    opt_pose = qkf->getPose().cast<double>();
    vector<Detection*> qkf_dets;
    qkf->getDetections(qkf_dets);

    vector<pair<Detection*, float>> dets_sort, dets_unique;
    for(int i = 0; i < qkf_dets.size(); ++i){
        // if(h_graph.getUScore(qkf->getFloor(), qkf_dets[i]->getClassName()) < 0.2){
        //     continue;
        // }
        dets_sort.push_back({qkf_dets[i], h_graph.getUScore(qkf->getFloor(), qkf_dets[i]->getClassName())});
    }
    if(dets_sort.empty()){
        cout<<"AA"<<endl;
        return false;
    }
    sort(dets_sort.begin(), dets_sort.end(), [](const pair<Detection*, float>& p1, const pair<Detection*, float>& p2){
        return p1.second > p2.second;
    });
    dets_unique = {dets_sort[0]};
    for(int i = 1; i < dets_sort.size(); ++i){
        if(dets_sort[i].second > 0.1){
            dets_unique.push_back(dets_sort[i]);
        }
    }

    gtsam::NonlinearFactorGraph base_graph;
    gtsam::Values base_init;
    for(const auto& elem : h_graph.getObjects(qkf->getFloor())){
        base_init.insert(O(elem->id()), elem->Q());
        Eigen::VectorXd fix_noise = Eigen::VectorXd::Ones(9) * 1.0e-7; // to fix the object
        auto gtsam_noise = gtsam::noiseModel::Diagonal::Sigmas(fix_noise);
        gtsam::PriorFactor<gtsam_quadrics::ConstrainedDualQuadric> fix_factor(O(elem->id()), elem->Q(), gtsam_noise);
        base_graph.add(fix_factor);
    }
    //=============1st Matching : Unique Object fitting================
    vector<Detection*> tkf_dets;
    tkf->getDetections(tkf_dets);
    vector<Object*> tkf_objects;
    for(int i = 0; i < tkf_dets.size(); ++i){
        if(tkf_dets[i]->getCorrespondence() != nullptr){
            tkf_objects.push_back(tkf_dets[i]->getCorrespondence());
        }
    }
    cv::Mat qkf_gray;
    cv::Mat tkf_gray;
    double err = 0.0;
    for(int iter = 0; iter < 1; iter++){
        operations_research::sat::CpModelBuilder cp_model;
        vector<vector<operations_research::sat::BoolVar>> x(dets_unique.size(), vector<operations_research::sat::BoolVar>(tkf_objects.size()));
        for(int i = 0; i < x.size(); ++i){
            for(int j = 0; j < x[0].size(); ++j){
                x[i][j] = cp_model.NewBoolVar();
            }
        }
        
        for(int j = 0; j < x[0].size(); ++j){
            vector<operations_research::sat::BoolVar> obj_const;
            for(int i = 0; i < x.size(); ++i){
                obj_const.push_back(x[i][j]);
            }
            cp_model.AddAtMostOne(obj_const);
        }
        for(int i = 0; i < x.size(); ++i){
            cp_model.AddExactlyOne(x[i]);
        }

        vector<vector<float>> costs(dets_unique.size(), vector<float>(tkf_objects.size()));
        for(int r = 0; r < costs.size(); ++r){
            for(int c = 0; c < costs[0].size(); ++ c){
                if(dets_unique[r].first->getClassName() != tkf_objects[c]->getClassName()){
                    costs[r][c] = 1.0e9;
                }
                else{
                    const DetectionGroup* dg = dets_unique[r].first->getDetectionGroup();
                    Eigen::Matrix3f K = dg->getIntrinsic();
                    gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
                    gtsam_quadrics::QuadricCamera qcam;
                    Eigen::Matrix4d Twc = opt_pose * dg->getSensorPose().cast<double>();
                    if(tkf_objects[c]->Q().isBehind(gtsam::Pose3(Twc)) || tkf_objects[c]->Q().contains(gtsam::Pose3(Twc))){
                        costs[r][c] = 1.0e9;
                        continue;
                    }
                    gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(tkf_objects[c]->Q(), gtsam::Pose3(Twc), K_gtsam).bounds();
                    gtsam_quadrics::AlignedBox2 bbox_act = dets_unique[r].first->getROI();
                    float dist = (bbox_est.center() - bbox_act.center()).norm();
                    costs[r][c] = dist + (abs(bbox_est.width() - bbox_act.width()) + abs(bbox_est.height() - bbox_act.height())); // object_uscores[c].second;
                }
            }
        }
        operations_research::sat::DoubleLinearExpr total_cost;
        for(int i = 0; i < x.size(); ++i){
            for(int j = 0; j < x[0].size(); ++j){
                total_cost += costs[i][j] * operations_research::sat::DoubleLinearExpr(x[i][j]);
            }
        }

        cp_model.Minimize(total_cost);
        operations_research::sat::CpSolverResponse result = operations_research::sat::Solve(cp_model.Build());
        if(result.status() == operations_research::sat::CpSolverStatus::INFEASIBLE){
            cout<<"BB"<<endl;
            return false;
        }
        double result_cost = result.objective_value();

        if(result_cost < 1.0e-8 || result_cost > 1.0e8){ // temporarily block error. 
            cout<<"CC. COST: "<<result_cost<<endl;
            return false;
        }

        //For Debug
        cv::Mat dg_gray = qkf_dets[0]->getDetectionGroup()->gray_.clone();
        
        cv::cvtColor(dg_gray, qkf_gray, cv::COLOR_GRAY2BGR);

        
        dg_gray = tkf_dets[0]->getDetectionGroup()->gray_.clone();
        cv::cvtColor(dg_gray, tkf_gray, cv::COLOR_GRAY2BGR);
        vector<pair<int, int>> corrs;
        for(int j = 0; j < x[0].size(); ++j){
            for(int i = 0; i < x.size(); ++i){
                if(operations_research::sat::SolutionBooleanValue(result, x[i][j])){   
                    corrs.push_back(make_pair(i, j));
                    unique_matches.push_back({dets_unique[i].first, tkf_objects[j]});
                    cv::rectangle(qkf_gray, dets_unique[i].first->getROI_CV(), cv::Scalar(0, 0, 255));
                    cv::putText(qkf_gray, to_string(corrs.size()), dets_unique[i].first->getROI_CV().tl(), 1, 2, cv::Scalar(0, 0, 255));

                    for(const auto& elem : tkf_dets){
                        if(elem->getCorrespondence() == tkf_objects[j]){
                            cv::rectangle(tkf_gray, elem->getROI_CV(), cv::Scalar(0, 0, 255));
                            cv::putText(tkf_gray, to_string(corrs.size()), elem->getROI_CV().tl(), 1, 2, cv::Scalar(0, 0, 255));
                            break;
                        }
                    }
                    break;
                }
            }
            
        }
    
        gtsam::NonlinearFactorGraph graph(base_graph);
        gtsam::Values init(base_init);
        init.insert(X(qkf->id()), gtsam::Pose3(opt_pose));
        auto init_pose_noise =  gtsam::noiseModel::Diagonal::Sigmas(0.1 *gtsam::Vector::Ones(6));
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(qkf->id()), gtsam::Pose3(opt_pose), init_pose_noise));
        for(const auto& cor : corrs){
            const DetectionGroup* dg = dets_unique[cor.first].first->getDetectionGroup();
            Eigen::Matrix3f K = dg->getIntrinsic();
            gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
            gtsam::Vector4 bbox_noise_vec = gtsam::Vector4(50.0, 50.0, 50.0, 50.0);
            auto bbox_noise = gtsam::noiseModel::Diagonal::Sigmas(bbox_noise_vec);
            gtsam_quadrics::BoundingBoxFactor bbf(dets_unique[cor.first].first->getROI(), K_gtsam, X(qkf->id()), O(tkf_objects[cor.second]->id()), bbox_noise);
            graph.add(bbf);
        }
        
        gtsam::LevenbergMarquardtOptimizer optim(graph, init);
        gtsam::Values opt = optim.optimize();
        opt_pose = opt.at<gtsam::Pose3>(X(qkf->id())).matrix();
        err = optim.error();

        //-------Debug---------
        for(const auto& cor : corrs){
            const DetectionGroup* dg = dets_unique[cor.first].first->getDetectionGroup();
            Eigen::Matrix3f K = dg->getIntrinsic();
            gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
            gtsam_quadrics::QuadricCamera qcam;
            gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(tkf_objects[cor.second]->Q(), gtsam::Pose3(opt_pose), K_gtsam).bounds();
            cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
            cv::rectangle(qkf_gray, est_cv, cv::Scalar(0, 255, 0));
        }
        for(const auto& obj : h_graph.getObjects(qkf->getFloor())){
            const DetectionGroup* dg = qkf_dets[0]->getDetectionGroup();
            Eigen::Matrix3f K = dg->getIntrinsic();
            gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
            if(obj->Q().isBehind(gtsam::Pose3(opt_pose)) || obj->Q().contains(gtsam::Pose3(opt_pose))){
                continue;
            }
            gtsam_quadrics::QuadricCamera qcam;
            gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(obj->Q(), gtsam::Pose3(opt_pose), K_gtsam).bounds();
            cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
            cv::rectangle(qkf_gray, est_cv, cv::Scalar(255, 0, 0));
           // cv::putText(qkf_gray,obj->getClassName()+to_string(obj->id()), est_cv.tl(),1, 1,cv::Scalar(255, 0, 0));
        }
    }
    

    
    //---------------------

    cv::Mat match_image;
    cv::drawMatches(qkf_gray, vector<cv::KeyPoint>(), tkf_gray, vector<cv::KeyPoint>(), vector<cv::DMatch>(), match_image);
    string folder = "/home/nuninu98/match_test/"+to_string(qkf->id())+"/";
    if(!boost::filesystem::exists(folder)){
        boost::filesystem::create_directories(folder);
    }
    string filename = folder + to_string(qkf->id())+"_"+to_string(tkf->id())+"_"+"1stmatch";
    cv::imwrite(filename+"_"+to_string(err)+"_.png", match_image);
    return true;
    //=================================================================
}


bool LoopMatcher::matchStep2(KeyFrame* qkf, KeyFrame* tkf, HGraph& h_graph, const vector<pair<Detection*, Object*>>& unique_matches, Eigen::Matrix4d& opt_pose, double& score){
    vector<Detection*> qkf_dets;
    qkf->getDetections(qkf_dets);

    gtsam::NonlinearFactorGraph base_graph;
    gtsam::Values base_init;
    for(const auto& elem : h_graph.getObjects(qkf->getFloor())){
        base_init.insert(O(elem->id()), elem->Q());
        Eigen::VectorXd fix_noise = Eigen::VectorXd::Ones(9) * 1.0e-7; // to fix the object
        auto gtsam_noise = gtsam::noiseModel::Diagonal::Sigmas(fix_noise);
        gtsam::PriorFactor<gtsam_quadrics::ConstrainedDualQuadric> fix_factor(O(elem->id()), elem->Q(), gtsam_noise);
        base_graph.add(fix_factor);
    }
    //=============2nd Matching : Overall Object fitting================
    vector<Object*> floor_objects = h_graph.getObjects(qkf->getFloor());
    vector<Object*> objects;
    for(auto& elem : floor_objects){
        vector<KeyFrame*> conn_kfs;
        elem->getConnectedKeyFrames(conn_kfs);
        if(find(conn_kfs.begin(), conn_kfs.end(), qkf) == conn_kfs.end()){
            objects.push_back(elem);
        }
        
    }
    operations_research::sat::CpModelBuilder cp_model;
    vector<vector<operations_research::sat::BoolVar>> x(qkf_dets.size(), vector<operations_research::sat::BoolVar>(objects.size()));
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            x[i][j] = cp_model.NewBoolVar();
        }
    }
    for(int j = 0; j < x[0].size(); ++j){
        vector<operations_research::sat::BoolVar> obj_const;
        for(int i = 0; i < x.size(); ++i){
            obj_const.push_back(x[i][j]);
        }
        cp_model.AddAtMostOne(obj_const);
    }
    for(int i = 0; i < x.size(); ++i){
        cp_model.AddExactlyOne(x[i]);
    }
    vector<vector<float>> costs(qkf_dets.size(), vector<float>(objects.size()));
    for(int r = 0; r < costs.size(); ++r){
        for(int c = 0; c < costs[0].size(); ++ c){
            if(qkf_dets[r]->getClassName() != objects[c]->getClassName()){
                costs[r][c] = 1.0e9;
            }
            else{
                const DetectionGroup* dg = qkf_dets[r]->getDetectionGroup();
                Eigen::Matrix3f K = dg->getIntrinsic();
                gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
                gtsam_quadrics::QuadricCamera qcam;
                Eigen::Matrix4d Twc = opt_pose * dg->getSensorPose().cast<double>();
                if(objects[c]->Q().isBehind(gtsam::Pose3(Twc)) || objects[c]->Q().contains(gtsam::Pose3(Twc))){
                    costs[r][c] = 1.0e9;
                    continue;
                }
                gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(objects[c]->Q(), gtsam::Pose3(Twc), K_gtsam).bounds();
                gtsam_quadrics::AlignedBox2 bbox_act = qkf_dets[r]->getROI();
                float dist = (bbox_est.center() - bbox_act.center()).norm();
                costs[r][c] = dist + ((abs(bbox_est.width() - bbox_act.width()) + abs(bbox_est.height() - bbox_act.height()))); // object_uscores[c].second;
            }
        }
    }
    for(const auto& um : unique_matches){
        for(int i = 0; i < x.size(); ++i){
            if(um.first == qkf_dets[i]){
                for(int j = 0; j < x[0].size(); ++j){
                    if(um.second == objects[j]){
                        costs[i][j] = 0.0;
                    }
                    else{
                        costs[i][j] = 1.0e9;
                    }
                }
                break;
            }
        }
    }
    operations_research::sat::DoubleLinearExpr total_cost;
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            total_cost += costs[i][j] * operations_research::sat::DoubleLinearExpr(x[i][j]);
        }
    }

    cp_model.Minimize(total_cost);
    operations_research::sat::CpSolverResponse result = operations_research::sat::Solve(cp_model.Build());
    if(result.status() == operations_research::sat::CpSolverStatus::INFEASIBLE){
        cout<<"2nd failed. INFI"<<endl;
        return false;
    }
    double result_cost = result.objective_value();
    cout<<"RESULT? "<<result_cost<<endl;
    if(result_cost < 1.0e-3 ){ // temporarily block error. 
        cout<<"2nd failed. COST: "<<result_cost<<endl;
        return false;
    }



    vector<pair<int, int>> corrs;
    for(int j = 0; j < x[0].size(); ++j){
        for(int i = 0; i < x.size(); ++i){
            if(operations_research::sat::SolutionBooleanValue(result, x[i][j])){   
                corrs.push_back(make_pair(i, j));
                break; 
            }
        }
        
    }
    //For Debug
    cv::Mat dg_gray = qkf_dets[0]->getDetectionGroup()->gray_.clone();
    cv::Mat qkf_gray_prev, qkf_gray_aft;
    cv::cvtColor(dg_gray, qkf_gray_prev, cv::COLOR_GRAY2BGR);
    cv::cvtColor(dg_gray, qkf_gray_aft, cv::COLOR_GRAY2BGR);
    //-------Debug---------
    unordered_set<Object*> matched_prev;
    for(int i = 0; i < corrs.size(); ++i){
        auto cor = corrs[i];        
        const DetectionGroup* dg = qkf_dets[cor.first]->getDetectionGroup();
        Eigen::Matrix3f K = dg->getIntrinsic();
        gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
        gtsam_quadrics::QuadricCamera qcam;
        gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(objects[cor.second]->Q(), gtsam::Pose3(opt_pose), K_gtsam).bounds();
        cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
        cv::rectangle(qkf_gray_prev, est_cv, cv::Scalar(0, 255, 0));
        cv::putText(qkf_gray_prev, to_string(i), est_cv.tl(), 1, 2, cv::Scalar(0, 255, 0));

        cv::rectangle(qkf_gray_prev, qkf_dets[cor.first]->getROI_CV(), cv::Scalar(0, 0, 255));
        cv::putText(qkf_gray_prev, to_string(i), qkf_dets[cor.first]->getROI_CV().tl(), 1, 2, cv::Scalar(0, 0, 255));
        matched_prev.insert(objects[cor.second]);
    }

    const DetectionGroup* dg = qkf_dets[0]->getDetectionGroup();
    Eigen::Matrix3f K = dg->getIntrinsic();
    unordered_map<Object*, gtsam_quadrics::AlignedBox2> visibles_prev;
    extractiVisibles(objects, K, opt_pose, visibles_prev);

    for(const auto& om_pair : visibles_prev){
        auto bbox_est = om_pair.second;
        cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
        cv::rectangle(qkf_gray_prev, est_cv, cv::Scalar(255, 0, 0));
        //cv::putText(qkf_gray_prev, om_pair.first->getClassName(), est_cv.tl(),1, 1,cv::Scalar(255, 0, 0));
    }

    gtsam::NonlinearFactorGraph graph(base_graph);
    gtsam::Values init(base_init);
    init.insert(X(qkf->id()), gtsam::Pose3(opt_pose));
    for(const auto& cor : corrs){
        const DetectionGroup* dg = qkf_dets[cor.first]->getDetectionGroup();
        Eigen::Matrix3f K = dg->getIntrinsic();
        gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
        gtsam::Vector4 bbox_noise_vec = gtsam::Vector4(10.0, 10.0, 10.0, 10.0);
        auto bbox_noise = gtsam::noiseModel::Diagonal::Sigmas(bbox_noise_vec);
        gtsam_quadrics::BoundingBoxFactor bbf(qkf_dets[cor.first]->getROI(), K_gtsam, X(qkf->id()), O(objects[cor.second]->id()), bbox_noise);
        graph.add(bbf);
    }
    
    gtsam::LevenbergMarquardtOptimizer optim(graph, init);
    gtsam::Values opt = optim.optimize();
    opt_pose = opt.at<gtsam::Pose3>(X(qkf->id())).matrix();
    //-------Debug---------
    unordered_set<Object*> matched_aft;
    for(int i = 0; i < corrs.size(); ++i){
        auto cor = corrs[i];        
        const DetectionGroup* dg = qkf_dets[cor.first]->getDetectionGroup();
        Eigen::Matrix3f K = dg->getIntrinsic();
        gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
        gtsam_quadrics::QuadricCamera qcam;
        gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(objects[cor.second]->Q(), gtsam::Pose3(opt_pose), K_gtsam).bounds();
        cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
        cv::rectangle(qkf_gray_aft, est_cv, cv::Scalar(0, 255, 0));
        cv::putText(qkf_gray_aft, to_string(i), est_cv.tl(), 1, 2, cv::Scalar(0, 255, 0));

        cv::rectangle(qkf_gray_aft, qkf_dets[cor.first]->getROI_CV(), cv::Scalar(0, 0, 255));
        cv::putText(qkf_gray_aft, to_string(i), qkf_dets[cor.first]->getROI_CV().tl(), 1, 2, cv::Scalar(0, 0, 255));
        matched_aft.insert(objects[cor.second]);
    }

    unordered_map<Object*, gtsam_quadrics::AlignedBox2> visibles_aft;
    extractiVisibles(objects, K, opt_pose, visibles_aft);
    for(const auto& om_pair : visibles_aft){
        gtsam_quadrics::AlignedBox2 bbox_est = om_pair.second;
        cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
        cv::rectangle(qkf_gray_aft, est_cv, cv::Scalar(255, 0, 0));
        //cv::putText(qkf_gray_aft, om_pair.first->getClassName(), est_cv.tl(),1, 1,cv::Scalar(255, 0, 0));
    }
    double err = optim.error();
    cout<<"ERR: "<<err<<endl;
   
    
    score = err;
    bool same_pattern = patternMatched(visibles_prev, visibles_aft);
    // if(err > 20.0){
    //     return false;
    // }
    //---------------------
    if(err > 30.0 || !same_pattern){
        return false;
    }
    
    string folder = "/home/nuninu98/match_test/"+to_string(qkf->id())+"/";
    if(!boost::filesystem::exists(folder)){
        boost::filesystem::create_directories(folder);
    }
    cv::Mat match_image;
    cv::drawMatches(qkf_gray_prev, vector<cv::KeyPoint>(), qkf_gray_aft, vector<cv::KeyPoint>(), vector<cv::DMatch>(), match_image);
    string filename = folder + to_string(qkf->id())+"_"+to_string(tkf->id())+"_"+"2ndmatch";
    cv::imwrite(filename+to_string(err)+"_.png", match_image);
    
    //=================================================================
    return true;
}

void LoopMatcher::extractiVisibles(const vector<Object*> objects, const Eigen::Matrix3f& K, const Eigen::Matrix4d& pose, unordered_map<Object*, gtsam_quadrics::AlignedBox2>& output){
    gtsam_quadrics::AlignedBox2 screen(0.0, 0.0, 1280.0, 720.0);
    for(const auto& obj : objects){
        gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
        if(obj->Q().isBehind(gtsam::Pose3(pose)) || obj->Q().contains(gtsam::Pose3(pose)) ){
            continue;
        }
        gtsam_quadrics::QuadricCamera qcam;
        gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(obj->Q(), gtsam::Pose3(pose), K_gtsam).bounds();
        if(screen.iou(bbox_est) > 1.0e-6){
            output.insert({obj, bbox_est});
        }
        
    }
}

bool LoopMatcher::patternMatched(unordered_map<Object*, gtsam_quadrics::AlignedBox2>& visible1,  unordered_map<Object*, gtsam_quadrics::AlignedBox2>& visible2){
    vector<Object*> commons;
    for(const auto& elem : visible1){
        if(visible2.find(elem.first) != visible2.end()){
            commons.push_back(elem.first);
        }
    }

    if(commons.size() < 3){
        return false;
    }
    gtsam::Point2 mu1(0.0, 0.0);
    gtsam::Point2 mu2(0.0, 0.0);
    for(const auto& common_obj: commons){
        mu1 += visible1[common_obj].center();
        mu2 += visible1[common_obj].center();
    }
    mu1 /= commons.size();
    mu2 /= commons.size();
    
    Eigen::Matrix2d cov1 = Eigen::Matrix2d::Zero();
    Eigen::Matrix2d cov2 = Eigen::Matrix2d::Zero();

    for(const auto& common_obj: commons){
        Eigen::Vector2d p1 = visible1[common_obj].center() - mu1;
        cov1 += p1 * p1.transpose();

        Eigen::Vector2d p2 = visible2[common_obj].center() - mu2;
        cov2 += p2 * p2.transpose();
    }
    cov1 /= commons.size();
    cov2 /= commons.size();
    // cout<<"COV1: \n"<<cov1<<endl;
    // cout<<"COV2: \n"<<cov2<<endl;

    Eigen::JacobiSVD<Eigen::Matrix2d> svd1(cov1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::Matrix2d> svd2(cov2, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector2d axis1 = svd1.matrixU().block<2, 1>(0, 0);
    Eigen::Vector2d axis2 = svd2.matrixU().block<2, 1>(0, 0);
    double cos_sim = axis1.dot(axis2);
    //cout<<"COS SIM: "<<cos_sim<<endl;
    
    Eigen::Vector2d S1 = svd1.singularValues().normalized();
    Eigen::Vector2d S2 = svd2.singularValues().normalized();
    double shape_sim = S1.dot(S2);//(S1 - S2).norm();
    // cout<<"S1 "<<S1.transpose()<<endl;
    // cout<<"S2 "<<S2.transpose()<<endl;
    // cout<<"SHAPE SIM: "<<shape_sim<<endl;

    Eigen::MatrixXd adjacent1 = Eigen::MatrixXd::Zero(commons.size(), commons.size());
    Eigen::MatrixXd adjacent2 = Eigen::MatrixXd::Zero(commons.size(), commons.size());
    for(int i = 0; i < commons.size(); ++i){
        for(int j = 0; j < commons.size(); ++j){
            double dist1 = (visible1[commons[i]].center() - visible1[commons[j]].center()).norm();
            double dist2 = (visible2[commons[i]].center() - visible2[commons[j]].center()).norm();
        
            adjacent1(i, j) = dist1;
            adjacent1(j, i) = dist1;

            adjacent2(i, j) = dist2;
            adjacent2(i, j) = dist2;
        }
    }
    adjacent1.rowwise().normalize();
    adjacent2.rowwise().normalize();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_adj1(adjacent1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_adj2(adjacent2, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto sigs1 = svd_adj1.singularValues();
    auto sigs2 = svd_adj2.singularValues();

    Eigen::VectorXd v1 = svd_adj1.matrixV().col(0);
    Eigen::VectorXd v2 = svd_adj2.matrixV().col(0);
    cout<<"SIG ERR: "<<(sigs1 - sigs2).norm()<<endl;
    cout<<"---"<<endl;
    if((sigs1 - sigs2).norm() > 0.5){
        return false;
    }
    //cout<<"ADJ cosSim: "<<v1.dot(v2)<<endl;
    
    
    return true;
}

bool LoopMatcher::match3(KeyFrame* qkf, KeyFrame* tkf, HGraph& h_graph, LoopMatchResult& output){
    unique_lock<mutex> lock(lock_);
    Eigen::Matrix4d opt_pose = qkf->getPose().cast<double>();
    vector<pair<Detection*, Object*>> unique_matches;
    cout<<"MATCH "<<qkf->id()<<" "<<tkf->id()<<endl;
    if(!matchStep1(qkf, tkf, h_graph, opt_pose, unique_matches)){
        return false;
    }
    
    double score = 0.0;
    bool result = matchStep2(qkf, tkf, h_graph, unique_matches, opt_pose, score);
    output.drift = tkf->getPose().inverse() * opt_pose.cast<float>();
    
    output.score = score;
    output.query = qkf->id();
    output.target = tkf->id();
    return result;
}
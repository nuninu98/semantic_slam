#include <semantic_slam/algorithm/loop_matcher.h>

LoopMatcher::LoopMatcher(){

}

LoopMatcher::~LoopMatcher(){

}

// bool LoopMatcher::match(KeyFrame* qkf, KeyFrame* tkf, const vector<pair<Object*,float>>& object_uscores, 
//                         Eigen::Matrix4f& Ttq_output, vector<pair<Detection*, Object*>>& corr_output){
//     vector<Detection*> dets;
//     qkf->getDetections(dets);
//     gtsam::NonlinearFactorGraph base_graph;
//     gtsam::Values base_init;
//     for(const auto& elem : object_uscores){
//         base_init.insert(O(elem.first->id()), elem.first->Q());
//         Eigen::VectorXd fix_noise = Eigen::VectorXd::Ones(9) * 1.0e-5; // to fix the object
//         auto gtsam_noise = gtsam::noiseModel::Diagonal::Sigmas(fix_noise);
//         gtsam::PriorFactor<gtsam_quadrics::ConstrainedDualQuadric> fix_factor(O(elem.first->id()), elem.first->Q(), gtsam_noise);
//         base_graph.add(fix_factor);
//     }
//     int N = 10;
//     double last_cost = 1.0e9;
//     vector<pair<cv::Mat, float>> test_imgs;
//     ros::Time begin = ros::Time::now();
//     Eigen::Matrix4d opt_pose = qkf->getPose().cast<double>();
//     vector<pair<int, int>> corrs;
//     bool reliable = false;

//     operations_research::sat::CpModelBuilder cp_model;
//     for(int iter = 0; iter < N; ++iter){
//         corrs.clear();
//         cv::Mat dg_gray = dets[0]->getDetectionGroup()->gray_.clone();
//         cv::Mat gray_color;
//         cv::cvtColor(dg_gray, gray_color, cv::COLOR_GRAY2BGR);
//         gtsam::NonlinearFactorGraph graph(base_graph);
//         gtsam::Values init(base_init);
//         init.insert(X(qkf->id()), gtsam::Pose3(opt_pose));
//         //N(Objects) >= N(Dets)
//         vector<vector<float>> costs(dets.size(), vector<float>(object_uscores.size()));
//         for(int r = 0; r < costs.size(); ++r){
//             for(int c = 0; c < costs[0].size(); ++ c){
//                 if(dets[r]->getClassName() != object_uscores[c].first->getClassName()){
//                     costs[r][c] = 1.0e9;
//                 }
//                 else{
//                     const DetectionGroup* dg = dets[r]->getDetectionGroup();
//                     Eigen::Matrix3f K = dg->getIntrinsic();
//                     gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
//                     gtsam_quadrics::QuadricCamera qcam;
//                     Eigen::Matrix4d Twc = opt_pose * dg->getSensorPose().cast<double>();
//                     if(object_uscores[c].first->Q().isBehind(gtsam::Pose3(Twc)) || object_uscores[c].first->Q().contains(gtsam::Pose3(Twc))){
//                         costs[r][c] = 1.0e9;
//                         continue;
//                     }
//                     gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(object_uscores[c].first->Q(), gtsam::Pose3(Twc), K_gtsam).bounds();
//                     gtsam_quadrics::AlignedBox2 bbox_act = dets[r]->getROI();
//                     float dist = (bbox_est.center() - bbox_act.center()).norm();
//                     double A1 = bbox_est.width() * bbox_est.height();
//                     double A2 = bbox_act.width() * bbox_act.height();
//                     costs[r][c] = dist * (abs(bbox_est.width() - bbox_act.width()) + abs(bbox_est.height() - bbox_act.height())); // object_uscores[c].second;
//                     cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
//                     cv::rectangle(gray_color, est_cv, cv::Scalar(0, 255, 0)); //green. est detection
//                     // if(isnan(costs[r][c])){
//                     //     cout<<"NAN!"<<endl;
//                     //     cout<<"DST: "<<dist<<endl;
//                     //     cout<<"SCORE: "<<object_uscores[c].second<<endl;
//                     // }
//                 }
//             }
//         }
//         // cout<<"COSTMAT: "<<endl;
//         // for(int r = 0; r < costs.size(); ++r){
//         //     for(int c = 0; c < costs[0].size(); ++ c){
//         //         cout<<costs[r][c]<<" ";
//         //     }
//         //     cout<<endl;
//         // }
//         //==========SAT SOLVER==========
//         // operations_research::sat::CpModelBuilder cp_model;
//         cp_model.ClearObjective();
//         cp_model.ClearAssumptions();
//         vector<vector<operations_research::sat::BoolVar>> x(costs.size(), vector<operations_research::sat::BoolVar>(costs[0].size()));
//         for(int i = 0; i < x.size(); ++i){
//             for(int j = 0; j < x[0].size(); ++j){
//                 x[i][j] = cp_model.NewBoolVar();
//             }
//         }
        
//         for(int j = 0; j < x[0].size(); ++j){
//             vector<operations_research::sat::BoolVar> obj_const;
//             for(int i = 0; i < x.size(); ++i){
//                 obj_const.push_back(x[i][j]);
//             }
//             cp_model.AddAtMostOne(obj_const);
//         }
//         for(int i = 0; i < x.size(); ++i){
//             cp_model.AddExactlyOne(x[i]);
//         }
//         //operations_research::sat::LinearExpr total_cost;
//         operations_research::sat::DoubleLinearExpr total_cost;
//         for(int i = 0; i < x.size(); ++i){
//             for(int j = 0; j < x[0].size(); ++j){
//                 //total_cost += (x[i][j] * costs[i][j]);
//                 total_cost += costs[i][j] * operations_research::sat::DoubleLinearExpr(x[i][j]);
//             }
//         }
//         cp_model.Minimize(total_cost);
//         operations_research::sat::CpSolverResponse result = operations_research::sat::Solve(cp_model.Build());
//         if(result.status() == operations_research::sat::CpSolverStatus::INFEASIBLE || result.status() != operations_research::sat::CpSolverStatus::OPTIMAL){
//             return false;
//         }
//         if(result.objective_value() > 1.0e8){
//             return false;
//         }   
//         double result_cost = result.objective_value();
        
        
//         cout<<"ITER "<<iter<<" COST: "<<result_cost<<endl;
//         if(result_cost < 0.1){ // temporarily block error. 
//             return false;
//         }
//         last_cost = result_cost;        
//         for(int i = 0; i < x.size(); ++i){
//             for(int j = 0; j < x[0].size(); ++j){
//                 if(operations_research::sat::SolutionBooleanValue(result, x[i][j])){   
//                     const DetectionGroup* dg = dets[i]->getDetectionGroup();
//                     Eigen::Matrix3f K = dg->getIntrinsic();
//                     gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
//                     gtsam_quadrics::QuadricCamera qcam;
//                     Eigen::Matrix4d Twc = opt_pose * dg->getSensorPose().cast<double>();
                    
//                     gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(object_uscores[j].first->Q(), gtsam::Pose3(Twc), K_gtsam).bounds();
//                     cv::Rect est_cv = cv::Rect(bbox_est.xmin(), bbox_est.ymin(), bbox_est.width(), bbox_est.height()) & cv::Rect(0, 0, 1280, 720);
//                     double A1 = bbox_est.width() * bbox_est.height();
//                     double A2 = dets[i]->getROI().width() * dets[i]->getROI().height();
//                     double cost = (bbox_est.center() - dets[i]->getROI().center()).norm() * abs(A1- A2)/A2;
//                     corrs.push_back(make_pair(i, j));
//                     cv::rectangle(gray_color, est_cv, cv::Scalar(0, 255, 0)); //green. est detection
//                     cv::putText(gray_color, to_string(corrs.size()), est_cv.tl(), 1, 2, cv::Scalar(0, 255, 0));

//                     cv::rectangle(gray_color, dets[i]->getROI_CV(), cv::Scalar(0, 0, 255)); //red. actual detection
//                     cv::putText(gray_color, to_string(corrs.size()), dets[i]->getROI_CV().tl(), 1, 2, cv::Scalar(0, 0, 255));
                    
//                 }
//             }
//         }
//         test_imgs.push_back({gray_color, result_cost});
//         //==========TODO==============
//         //Loop Query Modify (query, vector of targets)
//         //============================
//         for(const auto& cor : corrs){
//             // const DetectionGroup* dg = cor.first->getDetectionGroup();
//             const DetectionGroup* dg = dets[cor.first]->getDetectionGroup();
//             Eigen::Matrix3f K = dg->getIntrinsic();
//             gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
//             gtsam::Vector4 bbox_noise_vec = gtsam::Vector4(10.0, 10.0, 10.0, 10.0);
//             auto bbox_noise = gtsam::noiseModel::Diagonal::Sigmas(bbox_noise_vec);
//             gtsam_quadrics::BoundingBoxFactor bbf(dets[cor.first]->getROI(), K_gtsam, X(qkf->id()), O(object_uscores[cor.second].first->id()), bbox_noise);
//             //gtsam_quadrics::BoundingBoxFactor bbf(cor.first->getROI(), K_gtsam, X(qkf->id()), O(cor.second->id()), bbox_noise);
//             graph.add(bbf);
//         }
//         gtsam::LevenbergMarquardtOptimizer optim(graph, init);
//         gtsam::Values opt = optim.optimize();
//         opt_pose = opt.at<gtsam::Pose3>(X(qkf->id())).matrix();
//         if(abs(result_cost - last_cost) < 1.0e-4 && last_cost < 1000.0 && iter > 5){
//             reliable = true;
//             break;
//         }
        
//     }
//     if(!reliable){
//         return false;
//     }
//     for(auto& corr : corrs){
//         cout<<"det prev: "<<(dets[corr.first]->getCorrespondence() == nullptr ? "NULL" : to_string(dets[corr.first]->getCorrespondence()->id()))<<endl;
//         cout<<"match corr: "<<(object_uscores[corr.second].first->id())<<endl;
//         corr_output.push_back({dets[corr.first], object_uscores[corr.second].first});
//     }

//     string folder = "/home/nuninu98/match_test/"+to_string(qkf->id())+"/";

//     if(!boost::filesystem::exists(folder)){
//         boost::filesystem::create_directories(folder);
//     }
//     for(int i = 0; i < test_imgs.size(); ++i){
//         cv::imwrite(folder + to_string(tkf->id())+"_"+to_string(i)+"_"+to_string(test_imgs[i].second)+".png", test_imgs[i].first);
//     }
//     Ttq_output = tkf->getPose().inverse() * opt_pose.cast<float>();
//     // cout<<"LOOP "<<qkf->id()<<" <-> "<<tkf->id()<<endl;
//     // cout<<"BEF OPT: \n"<<qkf->getPose()<<endl;
//     // cout<<"AFT OPT: \n"<<opt_pose<<endl;
//     cout<<"-----"<<endl;
//     return true;
// }

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
#include <semantic_slam/algorithm/loop_matcher.h>

LoopMatcher::LoopMatcher(){

}

LoopMatcher::~LoopMatcher(){

}

bool LoopMatcher::match(KeyFrame* kf, const vector<pair<Object*,float>>& object_uscores){
    vector<Detection*> dets;
    kf->getDetections(dets);
    gtsam::NonlinearFactorGraph base_graph;
    gtsam::Values base_init;
    for(const auto& elem : object_uscores){
        base_init.insert(O(elem.first->id()), elem.first->Q());
        Eigen::VectorXd fix_noise = Eigen::VectorXd::Ones(9) * 1.0e-5; // to fix the object
        auto gtsam_noise = gtsam::noiseModel::Diagonal::Sigmas(fix_noise);
        gtsam::PriorFactor<gtsam_quadrics::ConstrainedDualQuadric> fix_factor(O(elem.first->id()), elem.first->Q(), gtsam_noise);
        base_graph.add(fix_factor);
    }
    
    int N = 100;
    Eigen::Matrix4d opt_pose = kf->getPose().cast<double>();
    for(int iter = 0; iter < N; ++iter){
        gtsam::NonlinearFactorGraph graph(base_graph);
        gtsam::Values init(base_init);
        init.insert(X(kf->id()), gtsam::Pose3(opt_pose));

        //N(Objects) >= N(Dets)
        vector<vector<float>> costs(dets.size(), vector<float>(object_uscores.size()));
        for(int r = 0; r < costs.size(); ++r){
            gtsam_quadrics::QuadricCamera qcam;
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
                    gtsam_quadrics::AlignedBox2 bbox_est = qcam.project(object_uscores[c].first->Q(), gtsam::Pose3(Twc), K_gtsam).bounds();
                    gtsam_quadrics::AlignedBox2 bbox_act = dets[r]->getROI();
                    float dist = (bbox_est.center() - bbox_act.center()).norm();
                    costs[r][c] = dist * object_uscores[c].second;
                }
            }
        }
        unique_ptr<operations_research::MPSolver> solver(operations_research::MPSolver::CreateSolver("SCIP"));
        vector<vector<operations_research::MPVariable*>> x(costs.size(), vector<operations_research::MPVariable*>(costs[0].size()));
        for(int i = 0; i < x.size(); ++i){
            for(int j = 0; j < x[0].size(); ++j){
                x[i][j] = solver->MakeIntVar(0, 1, "");
            }
        }

        for(int i = 0; i < x.size(); ++i){
            operations_research::LinearExpr det_sum;
            for(int j = 0; j < x[0].size(); ++j){
                det_sum += x[i][j];
            }
            solver->MakeRowConstraint(det_sum == 1.0);
        }

        for(int j = 0; j < x[0].size(); ++j){
            operations_research::LinearExpr obj_sum;
            for(int i = 0; i < x.size(); ++i){
                obj_sum += x[i][j];
            }
            solver->MakeRowConstraint(obj_sum <= 1.0);
        }

        operations_research::MPObjective* const objective = solver->MutableObjective();
        for(int i = 0; i < x.size(); ++i){
            for(int j = 0; j < x[0].size(); ++j){
                objective->SetCoefficient(x[i][j], costs[i][j]);
            }
        }
        objective->SetMinimization();
        const operations_research::MPSolver::ResultStatus result = solver->Solve();
        //====TODO====
        //Hungarian algorithm
        //BBF based on the Hungarian output
        //============
        
    }
    return true;
}
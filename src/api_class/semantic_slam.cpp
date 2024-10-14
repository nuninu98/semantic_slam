#include <semantic_slam/api_class/semantic_slam.h>
SemanticSLAM::SemanticSLAM(): pnh_("~"), kill_flag_(false), thread_killed_(false), depth_factor_(1000.0), floor_(nullptr), kf_updated_(false), last_key_(nullptr), last_oid_(-1)
{

    string voc_file;
    pnh_.param<string>("vocabulary_file", voc_file, "/home/nuninu98/catkin_ws/src/orb_semantic_slam/model/ORBvoc.txt");

    string setting_file;
    pnh_.param<string>("setting_file", setting_file, "/home/nuninu98/catkin_ws/src/orb_semantic_slam/setting/tum_rgbd.yaml");

    string crnn_file;
    pnh_.param<string>("crnn_file", crnn_file, "");
    
    string text_list;
    pnh_.param<string>("text_list", text_list, "");

    string door_detection_onnx;
    pnh_.param<string>("door_detection_onnx", door_detection_onnx, "");
    vector<string> door_detection_classes = {"floor_sign", "room_number"};
    door_detector_.reset(new LandmarkDetector(door_detection_onnx, door_detection_classes));

    string obj_detection_onnx;
    pnh_.param<string>("obj_detection_onnx", obj_detection_onnx, "");
    string obj_detection_classes_file;
    pnh_.param<string>("obj_detection_classes", obj_detection_classes_file, "");
    vector<string> obj_detection_classes;
    ifstream file(obj_detection_classes_file);
    string line;
    while(getline(file, line)){
        obj_detection_classes.push_back(line);
    }
    obj_detector_.reset(new LandmarkDetector(obj_detection_onnx, obj_detection_classes));
    

    ocr_.reset(new OCR(crnn_file, text_list));

    sidecam_in_frontcam_ = Eigen::Matrix4f::Identity();
    sidecam_in_frontcam_(0, 0) = 0.0;
    sidecam_in_frontcam_(0, 2) = 1.0;
    sidecam_in_frontcam_(2, 0) = -1.0;
    sidecam_in_frontcam_(2, 2) = 0.0;
    sidecam_in_frontcam_(0, 3) = 0.065;
    sidecam_in_frontcam_(1, 3) = 0.0;
    sidecam_in_frontcam_(2, 3) = -0.065;

    K_side_ = Eigen::Matrix3f::Identity();
    K_side_(0, 0) = 645.3115844726562; //d455
    K_side_(0, 2) = 644.2869873046875;
    K_side_(1, 1) = 644.4506225585938;
    K_side_(1, 2) = 361.4469299316406;
    // K_side_(0, 0) = 927.8262329101562;
    // K_side_(0, 2) = 658.4143676757812;
    // K_side_(1, 1) = 928.4344482421875;
    // K_side_(1, 2) = 359.071044921875;

    //pub_path_ = nh_.advertise<nav_msgs::Path>("slam_path", 1);
    for(int i = 0; i < 10; ++i){
        pub_floor_path_.push_back(nh_.advertise<nav_msgs::Path>("slam_path_"+to_string(i), 1));
    }
    cv::FileStorage fsSettings(setting_file.c_str(), cv::FileStorage::READ);
    K_front_ = Eigen::Matrix3f::Identity();
    K_front_(0, 0) = static_cast<float>(fsSettings["Camera1.fx"]);
    K_front_(0, 2) = static_cast<float>(fsSettings["Camera1.cx"]);
    K_front_(1, 1) = static_cast<float>(fsSettings["Camera1.fy"]);
    K_front_(1, 2) = static_cast<float>(fsSettings["Camera1.cy"]);

    

    visual_odom_ = new ORB_SLAM3::System(voc_file, setting_file ,ORB_SLAM3::System::RGBD, false);
    visual_odom_->registerKeyframeCall(&kf_updated_, &keyframe_cv_);
    visual_odom_->registerLoopCall(&lc_buf_, &loop_cv_, &loop_lock_);
    keyframe_thread_ = thread(&SemanticSLAM::keyframeCallback, this);
    keyframe_thread_.detach();

    loop_thread_ = thread(&SemanticSLAM::loopQueryCallback, this);
    loop_thread_.detach();
    
   
    string rgb_topic, depth_topic, imu_topic;
    pnh_.param<string>("rgb_topic", rgb_topic, "/camera/color/image_raw");
    pnh_.param<string>("depth_topic", depth_topic, "/camera/aligned_depth_to_color/image_raw");
    pnh_.param<string>("imu_topic", imu_topic, "/imu/data");

    sub_imu_ = nh_.subscribe(imu_topic, 1, &SemanticSLAM::imuCallback, this);
    pub_object_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("object_cloud", 1);
    pub_h_graph_ = nh_.advertise<visualization_msgs::MarkerArray>("hierarchy_graph", 1);
    pub_map_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("map_cloud", 1);
    pub_floor_ = nh_.advertise<visualization_msgs::Marker>("floor", 1);

    tracking_color_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, rgb_topic, 1));
    tracking_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, depth_topic, 1));
    tracking_sync_.reset(new message_filters::Synchronizer<track_sync_pol> (track_sync_pol(10), *tracking_color_, *tracking_depth_));
    tracking_sync_->registerCallback(boost::bind(&SemanticSLAM::trackingImageCallback, this, _1, _2));

    side_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, "/side/aligned_depth_to_color/image_raw", 1));
    side_yolo_.reset(new message_filters::Subscriber<yolo_protocol::YoloResult> (nh_, "/side/color/image_raw/yolo", 1));
    side_sync_.reset(new message_filters::Synchronizer<yolo_sync_pol> (yolo_sync_pol(10), *side_depth_, *side_yolo_));
    side_sync_->registerCallback(boost::bind(&SemanticSLAM::detectionImageCallback, this, _1, _2, sidecam_in_frontcam_, K_side_, 'S'));

    front_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, depth_topic, 1));
    front_yolo_.reset(new message_filters::Subscriber<yolo_protocol::YoloResult> (nh_, rgb_topic+"/yolo", 1));
    front_sync_.reset(new message_filters::Synchronizer<yolo_sync_pol> (yolo_sync_pol(10), *front_depth_, *front_yolo_));
    front_sync_->registerCallback(boost::bind(&SemanticSLAM::detectionImageCallback, this, _1, _2, Eigen::Matrix4f::Identity(), K_front_, 'F'));
}

void SemanticSLAM::trackingImageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image){
    //ros::Time tic = ros::Time::now();
    ros::Time stamp = rgb_image->header.stamp;
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvShare(rgb_image, "bgr8");
    cv_bridge::CvImageConstPtr cv_depth_bridge = cv_bridge::toCvShare(depth_image, depth_image->encoding);
    vector<ORB_SLAM3::IMU::Point> imu_points;
    imu_lock_.lock();
    while(!imu_buf_.empty()){
        sensor_msgs::Imu data = imu_buf_.front();
        double imu_stamp = data.header.stamp.toSec();
        if(imu_stamp > rgb_image->header.stamp.toSec()){
            break;
        }
        cv::Point3f accel(data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z);
        cv::Point3f gyro(data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z);
        ORB_SLAM3::IMU::Point p(accel, gyro, imu_stamp);
        imu_points.push_back(p);
        imu_buf_.pop();
    }
    imu_lock_.unlock();

    keyframe_lock_.lock();
    Eigen::Matrix4f vo_pose = visual_odom_->TrackRGBD(cv_rgb_bridge->image, cv_depth_bridge->image, stamp.toSec(), imu_points).matrix().inverse();
    Eigen::Matrix4f act_pose = Eigen::Matrix4f::Identity();
    if(last_key_ == nullptr){
        act_pose = vo_pose;
    }
    else{
        act_pose = last_key_->getPose() * (last_key_->getOdomPose().inverse() * vo_pose);
    }
    keyframe_lock_.unlock();
    Eigen::Matrix4f optic_in_map = act_pose;
    Eigen::Matrix4f base_in_map = optic_in_map * OPTIC_TF.inverse();

    vector<geometry_msgs::TransformStamped> tfs;
    geometry_msgs::TransformStamped tf;
    tf.header.frame_id = "map_optic";
    tf.header.stamp = rgb_image->header.stamp;
    tf.child_frame_id = "base_link";
    tf.transform.translation.x = base_in_map(0, 3);
    tf.transform.translation.y = base_in_map(1, 3);
    tf.transform.translation.z = base_in_map(2, 3);
    Eigen::Quaternionf q_tf(base_in_map.block<3, 3>(0, 0));
    tf.transform.rotation.w = q_tf.w();
    tf.transform.rotation.x = q_tf.x();
    tf.transform.rotation.y = q_tf.y();
    tf.transform.rotation.z = q_tf.z();
    tfs.push_back(tf);

    geometry_msgs::TransformStamped tf_offset;
    tf_offset.header.frame_id = "map";
    tf_offset.header.stamp = rgb_image->header.stamp;
    tf_offset.child_frame_id = "map_optic";
    tf_offset.transform.translation.x = 0.0;
    tf_offset.transform.translation.y = 0.0;
    tf_offset.transform.translation.z = 0.0;
    Eigen::Quaternionf q_tf_basecam(OPTIC_TF.block<3, 3>(0, 0));
    tf_offset.transform.rotation.w = q_tf_basecam.w();
    tf_offset.transform.rotation.x = q_tf_basecam.x();
    tf_offset.transform.rotation.y = q_tf_basecam.y();
    tf_offset.transform.rotation.z = q_tf_basecam.z();
    tfs.push_back(tf_offset);

    broadcaster_.sendTransform(tfs);

}
bool init = false;
void SemanticSLAM::detectionImageCallback(const sensor_msgs::ImageConstPtr& depth_image, const yolo_protocol::YoloResultConstPtr& yolo_result, const Eigen::Matrix4f& sensor_pose, const Eigen::Matrix3f& K, char sID){
    sensor_msgs::ImageConstPtr color_img = boost::make_shared<sensor_msgs::Image const>(yolo_result->original);
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvShare(color_img, "bgr8");
    cv_bridge::CvImageConstPtr cv_depth_bridge = cv_bridge::toCvShare(depth_image, depth_image->encoding);
    cv::Mat image = cv_rgb_bridge->image.clone();
    cv::Mat depth_mat = cv_depth_bridge->image.clone();
    cv::Mat depth_scaled;
    if((fabs(depth_factor_-1.0)>1e-5) || depth_mat.type()!=CV_32F){
        depth_mat.convertTo(depth_scaled,CV_32F, 1.0/depth_factor_);
    }
    vector<Detection*> detections;
    if(!init){
        door_detector_->detectObjectYOLO(image, depth_scaled, K, detections);
        init = true;
    }
    
    detections.clear();
    for(int i = 0; i < yolo_result->detections.detections.size(); ++i){
        auto detect = yolo_result->detections.detections[i];
        
        cv::Mat mask;
        if(!yolo_result->masks[i].data.empty()){
            sensor_msgs::ImageConstPtr mask_msg = boost::make_shared<sensor_msgs::Image const>(yolo_result->masks[i]);
            cv_bridge::CvImageConstPtr mask_cv = cv_bridge::toCvShare(mask_msg, "mono8");
            mask = mask_cv->image.clone();
        }
        cv::Rect roi(cv::Point(detect.bbox.center.x- detect.bbox.size_x/2, detect.bbox.center.y - detect.bbox.size_y/2), cv::Size(detect.bbox.size_x, detect.bbox.size_y));
        
        cv::Mat img_gray;
        cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
        if(detect.header.frame_id == "desk"){ //temporarily disabled
            continue;
        }
        Detection* det_p = new Detection(roi, cv::Mat(), detect.header.frame_id);
        det_p->calcInitQuadric(depth_scaled, mask, K);
        detections.push_back(det_p);
    }
    
    for(auto& m : detections){
        //==========Testing Room Number=========
        if(m->getClassName() == "room_number"){
            cv::Rect roi = m->getROI_CV();
            OCRDetection text_out;
            bool found_txt = ocr_->textRecognition(image, roi, text_out);
            if(found_txt){
                m->copyContent(text_out);
                cv::rectangle(image, roi, cv::Scalar(0, 0, 255), 2);
                cv::putText(image, m->getClassName(), roi.tl(), 1, 2, cv::Scalar(0, 0, 255));
            }
        }
        else {
            cv::rectangle(image, m->getROI_CV(), cv::Scalar(0, 0, 255), 2);
            cv::putText(image, m->getClassName(), m->getROI_CV().tl(), 1, 2, cv::Scalar(0, 0, 255));
            cv::rectangle(image, m->getROI_CV(), cv::Scalar(0, 0, 255), 2);
            cv::putText(image, m->getClassName(), m->getROI_CV().tl(), 1, 2, cv::Scalar(0, 0, 255));
        }
    }
   
    cv::Mat Gray;
    cv::cvtColor(image, Gray, cv::COLOR_BGR2GRAY);
    if(!detections.empty()){
        DetectionGroup dg(sensor_pose, detections, K, yolo_result->header.stamp.toSec(), sID);
        dg.gray_ = Gray.clone();
        object_lock_.lock();
        obj_detection_buf_.push(dg);
        object_lock_.unlock();
    }
    cv::imshow(yolo_result->header.frame_id, image);
    cv::waitKey(1);
    //============TODO===============
    /*
    1. Door + Floor sign dataset (clear)
    2. Yolo training (clear)
    3. Door -> detect room number (clear)
    4. Door -> Wall plane projection (dropped)
    5. Floor, Room info to ORB SLAM (clear)
    6. Semantic Object as child node of graph node (Jinwhan's feedback. Generalizing for rooms with no number)
    7. Comparison 
    */
    //===============================
}


void SemanticSLAM::imuCallback(const sensor_msgs::ImuConstPtr& imu){
    imu_lock_.lock();
    imu_buf_.push(*imu);
    imu_lock_.unlock();
}

SemanticSLAM::~SemanticSLAM(){
    kill_flag_ = true;
    keyframe_cv_.notify_all();
    visual_odom_->Shutdown();
}

void SemanticSLAM::addKeyFrame(KeyFrame* kf, const vector<DetectionGroup>& dgs){
    kf->setDetection(dgs);
    // vector<const DetectionGroup*> kf_dets;
    // kf->getDetection(kf_dets);
    vector<Detection*> kf_detections;
    kf->getDetections(kf_detections);

    if(floor_ == nullptr){
        floor_ = new Floor(0, kf);
        cout<<"Generate Floor"<<endl;
        kf->setFloor(floor_);
        h_graph_.insert(floor_);
    }
    else{
        floor_->refine();
        if(floor_->isInlier(kf)){
            kf->setFloor(floor_);
            floor_->addKeyFrame(kf);
        }
        else{
            floor_ = nullptr;
            for(const auto& f : h_graph_.floors()){
                if(f == nullptr){
                    cout<<"NULL FLOOR"<<endl;
                }
                if(f->isInlier(kf)){
                    floor_ = f;
                    break;
                }
            }
        }
    }

    if(last_key_ == nullptr){    
        new_values_.insert(X(kf->id()), gtsam::Pose3(kf->getPose().cast<double>()));
        gtsam_values_.insert(X(kf->id()), gtsam::Pose3(kf->getPose().cast<double>()));
        auto init_pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3).finished());
        gtsam::PriorFactor<gtsam::Pose3> init(X(kf->id()), gtsam::Pose3(kf->getPose().cast<double>()), init_pose_noise);
        gtsam_factors_.add(init);
        new_factors_.add(init);
        cout<<"INIT KF"<<endl;
        //new_conns.add(init);
    }
    else{
        auto pose_diff_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1).finished());
        Eigen::Matrix4f odom_meas = last_key_->getOdomPose().inverse() * kf->getOdomPose();
        gtsam::BetweenFactor<gtsam::Pose3> bf(X(last_key_->id()), X(kf->id()),  gtsam::Pose3(odom_meas.cast<double>()), pose_diff_noise);
        kf->setPose(last_key_->getPose() * odom_meas);
        new_values_.insert(X(kf->id()), gtsam::Pose3(kf->getPose().cast<double>()));
        gtsam_values_.insert(X(kf->id()), gtsam::Pose3(kf->getPose().cast<double>()));
        gtsam_factors_.add(bf);
        new_factors_.add(bf);
        //new_conns.add(gtsam::BetweenFactor<gtsam::Pose3>(X(last_key_->id()), X(kf->id()),  p1.inverse()*p2, pose_diff_noise));
    }
    vector<Object*> tgt_objs = h_graph_.getObjects(kf->getFloor());
    for(auto& det : kf_detections){
        Eigen::Matrix3f K = det->getDetectionGroup()->getIntrinsic();
        gtsam::Key sensor_id = gtsam::Symbol(det->getDetectionGroup()->sID(), kf->id());
        //dg->sID() == 'X' ? X(kf->id()) : S(kf->id());
        gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
        Eigen::Matrix4f cam_in_map = kf->getPose()* det->getDetectionGroup()->getSensorPose();
        gtsam::Pose3 Twc = gtsam::Pose3(cam_in_map.cast<double>());
        gtsam_quadrics::QuadricCamera quadric_cam;
        
        if(!isam_.valueExists(sensor_id) && !new_values_.exists(sensor_id)){
            new_values_.insert(sensor_id, gtsam::Pose3(cam_in_map.cast<double>()));
            gtsam_values_.insert(sensor_id, gtsam::Pose3(cam_in_map.cast<double>()));
            auto mount_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5).finished());
            gtsam::BetweenFactor<gtsam::Pose3> mount(X(kf->id()), sensor_id, gtsam::Pose3(det->getDetectionGroup()->getSensorPose().cast<double>()), mount_noise);
            gtsam_factors_.add(mount);
            new_factors_.add(mount);
        }
        int boundary_thresh = 20;
        int img_width = 1280;
        int img_height = 720;
        
        //===========IOU METHOD============
        double max_iou = 0.0;
        Object* matched_obj = nullptr;
        gtsam_quadrics::AlignedBox2 meas = det->getROI();
        
        for(int i = 0; i < tgt_objs.size(); ++i){
            gtsam_quadrics::ConstrainedDualQuadric Q_obj = tgt_objs[i]->Q();
            if(Q_obj.contains(Twc) || Q_obj.isBehind(Twc)){
                continue;
            }
            gtsam_quadrics::AlignedBox2 est = quadric_cam.project(Q_obj, gtsam::Pose3(cam_in_map.cast<double>()), K_gtsam).bounds();
            double iou = meas.iou(est);
            if(iou > max_iou){
                matched_obj = tgt_objs[i];
                max_iou = iou;
            }
        }

        bool matched = false;
        if(matched_obj == nullptr){
            matched = false;
        }
        else{
            matched = max_iou > 0.2;
            if(!matched){
                gtsam_quadrics::AlignedBox2 est = quadric_cam.project(matched_obj->Q(), gtsam::Pose3(cam_in_map.cast<double>()), K_gtsam).bounds();
                gtsam::Point2 est_center = est.center();
                gtsam::Point2 meas_center = meas.center();
                double dist = (est_center - meas_center).norm();
                if(dist < max(est.width(), est.height())){
                    matched = true;
                }
            }
        }
        gtsam::Vector4 bbox_noise_vec(50.0, 50.0, 50.0, 50.0);
        if(max_iou > 1.0e-5){
            bbox_noise_vec = bbox_noise_vec * (1.0 / max_iou);
        }
        auto bbox_noise = gtsam::noiseModel::Diagonal::Sigmas(bbox_noise_vec);
        if(matched){
            if(meas.xmin() < boundary_thresh || meas.xmax() > img_width - boundary_thresh || meas.ymin() < boundary_thresh || meas.ymax() > img_height - boundary_thresh){
                continue;
            }
            gtsam_quadrics::BoundingBoxFactor bbf(meas, K_gtsam, sensor_id, O(matched_obj->id()), bbox_noise, gtsam_quadrics::BoundingBoxFactor::TRUNCATED);
            new_factors_.add(bbf);
            det->setCorrespondence(matched_obj);
            matched_obj->addDetection(det);
        }
        else{
            
            gtsam_quadrics::ConstrainedDualQuadric dQc = det->Q_;
            gtsam::Pose3 dQc_pose = dQc.pose();

            if(dQc_pose.z() > 5.0 || dQc_pose.z() < 0){ // too far
                continue;
            }
            //gtsam_quadrics::ConstrainedDualQuadric Q(quadric_pose, radii);
            
            if(det->Q_.radii().norm() < 1.0e-3){
                continue;
            }
            
            gtsam::Pose3 dQw_pose = Twc.transformPoseFrom(dQc_pose);
            gtsam_quadrics::ConstrainedDualQuadric Q(dQw_pose, dQc.radii());
            //========IOU TEST=========
            gtsam_quadrics::AlignedBox2 est_box = quadric_cam.project(Q, gtsam::Pose3(cam_in_map.cast<double>()), K_gtsam).bounds();
            if(est_box.iou(meas) < 0.15){
                continue;
            }
            //=========================

            Object* new_obj = new Object(det->getClassName(), last_oid_ == -1 ? 0 : last_oid_ + 1, Q);
            // pcl::io::savePCDFile("/home/nuninu98/cloud_test/" + new_obj->getClassName()+to_string(new_obj->id())+".pcd", det->depth_cloud_);
            new_obj->addDetection(det);
            det->setCorrespondence(new_obj);
            if(kf->getFloor() == nullptr){
                cout<<"NULL FLOOR INSERT"<<endl;
            }
            h_graph_.insert(kf->getFloor(), new_obj);
            Eigen::VectorXd opf_noise_vec = Eigen::VectorXd::Ones(9);
            auto init_obj_noise = gtsam::noiseModel::Diagonal::Sigmas(opf_noise_vec);
            //auto init_obj_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(9) << 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0).finished());
            gtsam::PriorFactor<gtsam_quadrics::ConstrainedDualQuadric> opf(O(new_obj->id()), Q, init_obj_noise);
            new_values_.insert(O(new_obj->id()), Q);
            
            new_factors_.add(opf);

            gtsam_quadrics::BoundingBoxFactor bbf(meas, K_gtsam, sensor_id, O(new_obj->id()), bbox_noise, gtsam_quadrics::BoundingBoxFactor::TRUNCATED);
            new_factors_.add(bbf);
            last_oid_ = new_obj->id();

            
        }
        
    }
    
    //gtsam::NonlinearFactorGraph new_conns;
    isam_.update(new_factors_, new_values_);
    kfs_.insert(make_pair(kf->id(), kf));
    new_factors_.resize(0);
    new_values_.clear();

    gtsam::Values opt = isam_.calculateEstimate();
    
    for(auto& elem : kfs_){
        gtsam::Pose3 opt_pose = opt.at<gtsam::Pose3>(X(elem.first));
        elem.second->setPose(opt_pose.matrix().cast<float>());
    }
    h_graph_.updateObjectPoses(opt);
    
    last_key_ = kf;
}

void SemanticSLAM::publishPath(){
    if(last_key_ == nullptr){
        return;
    }
    vector<nav_msgs::Path> paths_floor;
    for(int i = 0; i < pub_floor_path_.size(); ++i){
        nav_msgs::Path path;
        path.header.frame_id = "map_optic";
        path.header.stamp = ros::Time::now();
        paths_floor.push_back(path);
    }

    for(size_t i = 1; i < last_key_->id(); ++i){
        if(kfs_.find(i) == kfs_.end()){
            continue;
        }
        auto k = kfs_[i];
        Eigen::Matrix4f pose = k->getPose() * OPTIC_TF.inverse();
        geometry_msgs::PoseStamped p;
        p.pose.position.x = pose(0, 3);
        p.pose.position.y = pose(1, 3);
        p.pose.position.z = pose(2, 3);
        vector<Floor*> floors = h_graph_.floors();
        for(int fl = 0; fl < floors.size(); ++fl){
            if(floors[fl] == k->getFloor()){
                paths_floor[fl].poses.push_back(p);
                break;
            }
        }
    }

    for(int i = 0; i < pub_floor_path_.size(); ++i){
        pub_floor_path_[i].publish(paths_floor[i]);
    }
}

void SemanticSLAM::keyframeCallback(){
    while(true){
        unique_lock<mutex> key_lock(keyframe_lock_);
        keyframe_cv_.wait(key_lock, [this]{return this->kf_updated_ || this->kill_flag_;});
        if(kill_flag_){
            thread_killed_ = true;
            break;
        }
        auto orb_kf = visual_odom_->getLastKF();

        double stamp = orb_kf->mTimeStamp;
        vector<DetectionGroup> detection_groups;
        unordered_map<char, DetectionGroup> id_dg;

        object_lock_.lock();
        while(!obj_detection_buf_.empty()){
            double obj_stamp = obj_detection_buf_.front().stamp();
            if(obj_stamp > stamp){
                break;
            }
            if(stamp - obj_stamp < 0.1){
                if(id_dg.find(obj_detection_buf_.front().sID()) == id_dg.end()){
                    id_dg.insert(make_pair(obj_detection_buf_.front().sID(), obj_detection_buf_.front()));
                }
                else if(id_dg[obj_detection_buf_.front().sID()].stamp() < obj_stamp){
                    id_dg[obj_detection_buf_.front().sID()] = obj_detection_buf_.front();
                }
                //detection_groups.push_back(obj_detection_buf_.front());
            }
            obj_detection_buf_.pop();
        }
        object_lock_.unlock();
        for(const auto& elem : id_dg){
            detection_groups.push_back(elem.second);
        }
        KeyFrame* new_kf = new KeyFrame(orb_kf->mnId, orb_kf->GetPoseInverse().matrix());
        new_kf->bow_vec = orb_kf->mBowVec;
        // new_kf->color_ = orb_kf->color_;
        // new_kf->depth_ = orb_kf->depth_;
        addKeyFrame(new_kf, detection_groups);
        kf_updated_ = false;    
        key_lock.unlock();
        publishPath();
        // if(kfs_.size() % 10 == 0){
        //     pcl::PointCloud<pcl::PointXYZRGB> map_cloud;
        //     getMapCloud(map_cloud);
        //     sensor_msgs::PointCloud2 map_cloud_ros;
        //     pcl::toROSMsg(map_cloud, map_cloud_ros);
        //     map_cloud_ros.header.stamp = ros::Time::now();
        //     map_cloud_ros.header.frame_id = "map_optic";
        //     pub_map_cloud_.publish(map_cloud_ros);
        // }

        // sensor_msgs::PointCloud2 map_cloud_ros;
        // pcl::toROSMsg(map_cloud, map_cloud_ros);
        // map_cloud_ros.header.stamp = ros::Time::now();
        // map_cloud_ros.header.frame_id = "map_optic";
        // pub_map_cloud_.publish(map_cloud_ros);

        visualization_msgs::MarkerArray h_graph_vis;
        visualizeHGraph(h_graph_vis);
        pub_h_graph_.publish(h_graph_vis);
        vector<pair<KeyFrame*, float>> loop_candidates;
        findSemanticLoopCandidates(new_kf, 3, loop_candidates);
        for(int i = 0; i < loop_candidates.size(); ++i){
            if(loop_candidates[i].second > 0.8){
                ORB_SLAM3::LoopQuery lq(ORB_SLAM3::LOOP_TYPE::SEMANTIC ,new_kf->id(), loop_candidates[i].first->id(), Eigen::Matrix4f::Zero());
                loop_lock_.lock();
                lc_buf_.push(lq);
                loop_lock_.unlock();
                loop_cv_.notify_all();
            }
            
        }
        // if(loop_candidates[0].second < 0.8){
        //     continue;
        // }
        // ORB_SLAM3::LoopQuery lq(ORB_SLAM3::LOOP_TYPE::SEMANTIC ,new_kf->id(), loop_candidates[0].first->id(), Eigen::Matrix4f::Zero());
        //===========Test===============
        // string fileFolder = "/home/nuninu98/loopscore/";
        // ofstream semantic_score(fileFolder + "sem"+to_string(new_kf->id())+".txt", ios::app);
        // ofstream bow_score(fileFolder + "bow"+to_string(new_kf->id())+".txt", ios::app);
        // for(int i = 0; i < loop_candidates.size(); ++i){
        //     semantic_score << loop_candidates[i].first->id()<<" "<<loop_candidates[i].second<<endl;
        // }
        // for(int i = 0; i < loop_candidates.size(); ++i){
        //     //bow_score << lq.candidates[i].second<<" "<<lq.candidates[i].first<<endl;
        //     bow_score << loop_candidates[i].first->id()<<" "<<L1Score(kfs_[new_kf->id()]->bow_vec, loop_candidates[i].first->bow_vec) <<endl;
        // }
        // string queryFolder = "/home/nuninu98/loopscore/" + to_string(new_kf->id())+"/";
        // if(!boost::filesystem::exists(queryFolder)){
        //     boost::filesystem::create_directory(queryFolder);
        // }
        // vector<const DetectionGroup*> query_dgs, best_sem_dgs;
        // new_kf->getDetection(query_dgs);
        // loop_candidates[0].first->getDetection(best_sem_dgs);
        // for(auto& elem : query_dgs){
        //     cv::imwrite(queryFolder + "query_img"+elem->sID() + ".png", elem->gray_); 
        // }

        // for(auto& elem : best_sem_dgs){
        //     cv::imwrite(queryFolder + to_string(loop_candidates[0].first->id())+elem->sID() + ".png", elem->gray_); 
        // }
        //==============================
    }
}

void SemanticSLAM::loopQueryCallback(){
    while(true){
        unique_lock<mutex> loop_lock(loop_lock_);
        loop_cv_.wait(loop_lock, [this]{return !this->lc_buf_.empty() || this->kill_flag_;});
        if(kill_flag_){
            thread_killed_ = true;
            break;
        }
        
        while(!lc_buf_.empty()){
            ORB_SLAM3::LoopQuery lq = lc_buf_.front();
            if(lq.type == ORB_SLAM3::LOOP_TYPE::BAG_OF_WORDS){
                if(isam_.valueExists(X(lq.id_query)) && isam_.valueExists(X(lq.id_target)) &&(kfs_[lq.id_query]->getFloor() == kfs_[lq.id_target]->getFloor())){
                    auto loop_noise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2).finished());
                    gtsam::BetweenFactor<gtsam::Pose3> lc((X(lq.id_target)), X(lq.id_query), gtsam::Pose3(lq.drift.cast<double>()), loop_noise_);
                    gtsam_factors_.add(lc);
                    new_factors_.add(lc);
                }
            }
            else if(lq.type == ORB_SLAM3::LOOP_TYPE::SEMANTIC){
                vector<Detection*> qry_dets;
                kfs_[lq.id_query]->getDetections(qry_dets);
                if(isam_.valueExists(X(lq.id_query)) && isam_.valueExists(X(lq.id_target))){
                    unordered_map<Object*, float> obj_uscores;
                    vector<Detection*> tgt_dets;

                    size_t id_begin = lq.id_target > 10 ? lq.id_target - 10 : 0;
                    size_t id_end = lq.id_target + 10;
                    for(size_t i = id_begin; i < id_end; ++i){
                        if(kfs_.find(i) != kfs_.end()){
                            kfs_[i]->getDetections(tgt_dets);
                            for(auto& elem : tgt_dets){
                                Object* corr = elem->getCorrespondence();
                                if(corr != nullptr){
                                    if(obj_uscores.find(corr) == obj_uscores.end()){
                                        obj_uscores.insert(make_pair(corr, h_graph_.getUScore(floor_, elem->getClassName())));
                                    }
                                }
                            }
                        }
                    }
                    
                    //cout<<"LQ! sem: "<<loop_candidates[i].second<<" bow: "<<L1Score(new_kf->bow_vec, loop_candidates[i].first->bow_vec) <<endl;
                    // kfs_[lq.id_query]->printDets();
                    // kfs_[lq.id_target]->printDets();
                    // cout<<"OBJS: "<<tgt_objects.size()<<endl;
                    // cout<<"----"<<endl;
                    
                    if(obj_uscores.size() >= 3 && qry_dets.size() > 3){
                        cout<<"ENOUGH OBJS"<<endl;
                    }

                }
            }
            
            lc_buf_.pop();
            
        }

        

        
        isam_.update(new_factors_);
        new_factors_.resize(0);
        //gtsam_factors_.resize(0);
        gtsam::Values opt = isam_.calculateEstimate();
        for(auto& elem : kfs_){
            gtsam::Pose3 opt_pose = opt.at<gtsam::Pose3>(X(elem.first));
            elem.second->setPose(opt_pose.matrix().cast<float>());
        }
        h_graph_.updateObjectPoses(opt);
        
        h_graph_.refineObject();
    }
}

void SemanticSLAM::getMapCloud(pcl::PointCloud<pcl::PointXYZRGB>& output){
    // pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
    // pcl::VoxelGrid<pcl::PCLPointCloud2> voxel;
    // voxel.setLeafSize(0.5, 0.5, 0.5);
    // for(size_t i = 0; i < kfs_.size(); i += 10){
    //     if(kfs_.find(i) == kfs_.end()){
    //         continue;
    //     }
    //     KeyFrame* kf = kfs_[i];
    //     //=====Generate Cloud====
    //     pcl::PointCloud<pcl::PointXYZRGB> raw_cloud, cloud_tf;
    //     pcl::PCLPointCloud2 tf_pcl2;
    //     for(int r = 0; r < kf->color_.rows; r += 3){
    //         for(int c = 0; c < kf->color_.cols; c += 3){
    //             float depth = kf->depth_.at<float>(r, c);
    //             if(isnanf(depth) || depth < 1.0e-4 || depth > 5.0){
    //                 continue;
    //             }
    //             Eigen::Vector3f pix(c, r, 1.0);
    //             float x = (c - K_front_(0, 2)) * depth / K_front_(0, 0);
    //             float y = (r - K_front_(1, 2)) * depth / K_front_(1, 1);
    //             pcl::PointXYZRGB pt;
    //             pt.x = x;
    //             pt.y = y;
    //             pt.z = depth;
    //             pt.r = kf->color_.at<cv::Vec3b>(r, c)[2];
    //             pt.g = kf->color_.at<cv::Vec3b>(r, c)[1];
    //             pt.b = kf->color_.at<cv::Vec3b>(r, c)[0];
    //             raw_cloud.push_back(pt);
    //         }
    //     }
    //     pcl::transformPointCloud(raw_cloud, cloud_tf, kf->getPose());
    //     pcl::toPCLPointCloud2(cloud_tf, tf_pcl2);
    //     *cloud += tf_pcl2;
    //     voxel.setInputCloud(cloud);
    //     voxel.filter(*cloud);
    //     //=======================
    // }
    // pcl::fromPCLPointCloud2(*cloud, output);
    // output = *cloud;
}

void SemanticSLAM::visualizeHGraph(visualization_msgs::MarkerArray& output){
    output.markers.clear();
    size_t id = 0;
    vector<Object*> objs = h_graph_.getEveryObjects();
    vector<geometry_msgs::TransformStamped> tfs;
    for(const auto& obj : objs){
        visualization_msgs::Marker obj_marker;
        obj_marker.type = visualization_msgs::Marker::SPHERE;
        obj_marker.id = id;
        id++;
        obj_marker.header.stamp = ros::Time::now();
        obj_marker.header.frame_id ="map_optic";
        if(obj->getClassName() == "room_sign"){
            obj_marker.color.a = 0.6;
            obj_marker.color.r = 0.0;
            obj_marker.color.g = 0.0;
            obj_marker.color.b = 255.0;
        }
        else if(obj->getClassName()== "extinguisher"){
            obj_marker.color.a = 0.6;
            obj_marker.color.r = 255.0;
            obj_marker.color.g = 0.0;
            obj_marker.color.b = 0.0;
        }
        else{
            obj_marker.color.a = 0.6;
            obj_marker.color.r = 0.0;
            obj_marker.color.g = 255.0;
            obj_marker.color.b = 0.0;
        }
        
        obj_marker.pose.position.x = obj->Q().centroid().x();
        obj_marker.pose.position.y = obj->Q().centroid().y();
        obj_marker.pose.position.z = obj->Q().centroid().z();
        obj_marker.pose.orientation.w = obj->Q().pose().rotation().toQuaternion().w();
        obj_marker.pose.orientation.x = obj->Q().pose().rotation().toQuaternion().x();
        obj_marker.pose.orientation.y = obj->Q().pose().rotation().toQuaternion().y();
        obj_marker.pose.orientation.z = obj->Q().pose().rotation().toQuaternion().z();
        obj_marker.scale.x = obj->Q().radii()(0);
        obj_marker.scale.y = obj->Q().radii()(1);
        obj_marker.scale.z = obj->Q().radii()(2);
        output.markers.push_back(obj_marker);

        geometry_msgs::TransformStamped quad_axis;
        quad_axis.header.frame_id = "map_optic";
        quad_axis.header.stamp = ros::Time::now();
        quad_axis.transform.rotation = obj_marker.pose.orientation;
        quad_axis.transform.translation.x = obj_marker.pose.position.x;
        quad_axis.transform.translation.y = obj_marker.pose.position.y;
        quad_axis.transform.translation.z = obj_marker.pose.position.z;
        quad_axis.child_frame_id= obj->getClassName()+to_string(obj->id());
        tfs.push_back(quad_axis);
    }
    broadcaster_.sendTransform(tfs);
    // unordered_set<const ORB_SLAM3::KeyFrame*> keys;
    // size_t id = 0;
    // vector<std_msgs::ColorRGBA> colors(4);
    // colors[0].a = 1.0;
    // colors[0].r = 255.0;
    // colors[0].g = 255.0;
    // colors[0].b = 255.0;

    // colors[1].a = 1.0;
    // colors[1].r = 255.0;
    // colors[1].g = 0.0;
    // colors[1].b = 0.0;

    // colors[2].a = 1.0;
    // colors[2].r = 0.0;
    // colors[2].g = 255.0;
    // colors[2].b = 0.0;

    // colors[3].a = 1.0;
    // colors[3].r = 0.0;
    // colors[3].g = 0.0;
    // colors[3].b = 255.0;
    // int floor = 0;
    // for(const auto& fo_pair : h_graph){
        
    //     for(const auto& obj : fo_pair.second){
    //         pcl::PointCloud<pcl::PointXYZRGB> ocl;
    //         Eigen::Vector3f centroid = obj->getCentroid();
    //         visualization_msgs::Marker obj_marker;
    //         obj_marker.type = visualization_msgs::Marker::SPHERE;
    //         obj_marker.id = id;
    //         id++;
    //         obj_marker.header.stamp = ros::Time::now();
    //         obj_marker.header.frame_id ="map_optic";
    //         obj_marker.color = colors[(floor + 1) % 3];
    //         obj_marker.pose.position.x = centroid(0);
    //         obj_marker.pose.position.y = centroid(1);
    //         obj_marker.pose.position.z = centroid(2);
    //         obj_marker.pose.orientation.w = 1.0;
    //         obj_marker.pose.orientation.x = 0.0;
    //         obj_marker.pose.orientation.y = 0.0;
    //         obj_marker.pose.orientation.z = 0.0;
    //         obj_marker.scale.x = 0.2;
    //         obj_marker.scale.y = 0.2;
    //         obj_marker.scale.z = 0.2;
    //         output.markers.push_back(obj_marker);
            
    //         vector< ORB_SLAM3::KeyFrame*> conn_keys;
    //         obj->getConnectedKeyFrames(conn_keys);
    //         for(int i = 0; i < conn_keys.size(); i += 3){
    //             auto k = conn_keys[i];
    //             if(keys.find(k) != keys.end()){
    //                 continue;
    //             }
    //             visualization_msgs::Marker kf_marker;
    //             kf_marker.header.stamp = ros::Time::now();
    //             kf_marker.header.frame_id ="map_optic";
    //             kf_marker.color = colors[(floor + 1) % 3];
    //             kf_marker.id = id;
    //             id++;
    //             Eigen::Matrix4f pose = k->GetPoseInverse().matrix();
    //             Eigen::Quaternionf q(pose.block<3, 3>(0, 0));
    //             kf_marker.type = visualization_msgs::Marker::SPHERE;
    //             kf_marker.scale.x = 0.1;
    //             kf_marker.scale.y = 0.1;
    //             kf_marker.scale.z = 0.1;
    //             kf_marker.pose.position.x = pose(0, 3);
    //             kf_marker.pose.position.y = pose(1, 3);
    //             kf_marker.pose.position.z = pose(2, 3);
    //             kf_marker.pose.orientation.w = q.w(); 
    //             kf_marker.pose.orientation.x = q.x();
    //             kf_marker.pose.orientation.y = q.y();
    //             kf_marker.pose.orientation.z = q.z();
    //             output.markers.push_back(kf_marker);
    //             keys.insert(k);

    //             visualization_msgs::Marker edge_marker;
    //             edge_marker.header = kf_marker.header;
    //             edge_marker.id = id;
    //             id++;
    //             edge_marker.color.a = 1.0;
    //             edge_marker.color.r = 255.0;
    //             edge_marker.color.g = 255.0;
    //             edge_marker.color.b = 0.0;
    //             edge_marker.points.push_back(obj_marker.pose.position);
    //             edge_marker.points.push_back(kf_marker.pose.position);
    //             edge_marker.scale.x = 0.1;
    //             edge_marker.scale.y = 0.1;
    //             output.markers.push_back(edge_marker);
    //         }
    //     }
    //     floor++;
    // }
}

void SemanticSLAM::findSemanticLoopCandidates(KeyFrame* kf, int N, vector<pair<KeyFrame*, float>>& output){
    unordered_map<KeyFrame*, float> kf_scores;
    h_graph_.getMatchedKFs(kf, kf_scores);
    vector<pair<KeyFrame*, float>> score_sorted;
    for(const auto& elem : kf_scores){
        if(kf->id() - elem.first->id() > 500){
            score_sorted.push_back(elem);
        }
    }
    sort(score_sorted.begin(), score_sorted.end(), [](const pair<KeyFrame*, float>& p1, const pair<KeyFrame*, float>& p2){
        //return p1.first->id() < p2.first->id();
        return p1.second > p2.second;
    });
    for(size_t i = 0; i < min((size_t)N, score_sorted.size()); ++i){
        output.push_back(score_sorted[i]);
    }
}

double SemanticSLAM::L1Score(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2) const{
    DBoW2::BowVector::const_iterator v1_it, v2_it;
    const DBoW2::BowVector::const_iterator v1_end = v1.end();
    const DBoW2::BowVector::const_iterator v2_end = v2.end();
    
    v1_it = v1.begin();
    v2_it = v2.begin();
    
    double score = 0;
    
    while(v1_it != v1_end && v2_it != v2_end)
    {
        const DBoW2::WordValue& vi = v1_it->second;
        const DBoW2::WordValue& wi = v2_it->second;
        
        if(v1_it->first == v2_it->first)
        {
        score += fabs(vi - wi) - fabs(vi) - fabs(wi);
        
        // move v1 and v2 forward
        ++v1_it;
        ++v2_it;
        }
        else if(v1_it->first < v2_it->first)
        {
        // move v1 forward
        v1_it = v1.lower_bound(v2_it->first);
        // v1_it = (first element >= v2_it.id)
        }
        else
        {
        // move v2 forward
        v2_it = v2.lower_bound(v1_it->first);
        // v2_it = (first element >= v1_it.id)
        }
    }
    
    // ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|) 
    //		for all i | v_i != 0 and w_i != 0 
    // (Nister, 2006)
    // scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
    score = -score/2.0;

    return score; // [0..1]
}

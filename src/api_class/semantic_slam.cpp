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
    // K_side_(0, 0) = 645.3115844726562; //d455
    // K_side_(0, 2) = 644.2869873046875;
    // K_side_(1, 1) = 644.4506225585938;
    // K_side_(1, 2) = 361.4469299316406;
    K_side_(0, 0) = 927.8262329101562;
    K_side_(0, 2) = 658.4143676757812;
    K_side_(1, 1) = 928.4344482421875;
    K_side_(1, 2) = 359.071044921875;

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
    visual_odom_->registerKeyframeCall(&kf_updated_, &keyframe_cv_, &lc_buf_);
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
    //sub_sidecam_detection_ = nh_.subscribe<sensor_msgs::Image>("/d435/color/image_raw", 1, boost::bind(&SemanticSLAM::detectionImageCallback, this, _1, door_detector_, sidecam_in_frontcam_));
    //sub_frontcam_detection_ = nh_.subscribe<sensor_msgs::Image>(rgb_topic, 1, boost::bind(&SemanticSLAM::detectionImageCallback, this, _1, obj_detector_));

    tracking_color_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, rgb_topic, 1));
    tracking_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, depth_topic, 1));
    tracking_sync_.reset(new message_filters::Synchronizer<sync_pol> (sync_pol(1000), *tracking_color_, *tracking_depth_));
    tracking_sync_->registerCallback(boost::bind(&SemanticSLAM::trackingImageCallback, this, _1, _2));

    side_color_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, "/side/color/image_raw", 1));
    side_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, "/side/aligned_depth_to_color/image_raw", 1));
    side_sync_.reset(new message_filters::Synchronizer<sync_pol> (sync_pol(1000), *side_color_, *side_depth_));
    side_sync_->registerCallback(boost::bind(&SemanticSLAM::detectionImageCallback, this, _1, _2, sidecam_in_frontcam_, K_side_));

    front_color_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, rgb_topic, 1));
    front_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, depth_topic, 1));
    front_sync_.reset(new message_filters::Synchronizer<sync_pol> (sync_pol(1000), *front_color_, *front_depth_));
    front_sync_->registerCallback(boost::bind(&SemanticSLAM::detectionImageCallback, this, _1, _2, Eigen::Matrix4f::Identity(), K_front_));
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

void SemanticSLAM::detectionImageCallback(const sensor_msgs::ImageConstPtr& color_image, const sensor_msgs::ImageConstPtr& depth_image, const Eigen::Matrix4f& sensor_pose, const Eigen::Matrix3f& K){
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvShare(color_image, "bgr8");
    cv_bridge::CvImageConstPtr cv_depth_bridge = cv_bridge::toCvShare(depth_image, depth_image->encoding);
    cv::Mat image = cv_rgb_bridge->image.clone();
    vector<Detection> detections;
    if(sensor_pose == Eigen::Matrix4f::Identity()){
        obj_detector_->detectObjectYOLO(image, detections);
    }
    else{
        door_detector_->detectObjectYOLO(image, detections);
    }
    
    
    //vector<OCRDetection> text_detections = ocr_->detect_rec(image);
    vector<Detection> doors;
    
    for(auto& m : detections){
        //m.sensor_pose_ = sensor_pose;
        //==========Testing Room Number=========
        if(m.getClassName() == "room_number"){
            cv::Rect roi = m.getRoI();
            OCRDetection text_out;
            bool found_txt = ocr_->textRecognition(image, roi, text_out);
            if(found_txt){
                m.copyContent(text_out);
                doors.push_back(m);
                cv::rectangle(image, m.getRoI(), cv::Scalar(0, 0, 255), 2);
                cv::putText(image, m.getClassName(), m.getRoI().tl(), 1, 2, cv::Scalar(0, 0, 255));
            }
        }
        // else {
        //     cv::Rect roi = m.getRoI();
        //     cv::rectangle(image, m.getRoI(), cv::Scalar(0, 0, 255), 2);
        //     cv::putText(image, m.getClassName(), m.getRoI().tl(), 1, 2, cv::Scalar(0, 0, 255));
        //     doors.push_back(m);
        //     cv::rectangle(image, m.getRoI(), cv::Scalar(0, 0, 255), 2);
        //     cv::putText(image, m.getClassName(), m.getRoI().tl(), 1, 2, cv::Scalar(0, 0, 255));
        //     //doors.push_back(m);
        // }
    }
   

    if(!doors.empty()){
        cv::Mat color_mat = cv_rgb_bridge->image.clone();
        cv::Mat depth_mat = cv_depth_bridge->image.clone();
        cv::Mat depth_scaled;
        if((fabs(depth_factor_-1.0)>1e-5) || depth_mat.type()!=CV_32F){
            depth_mat.convertTo(depth_scaled,CV_32F, 1.0/depth_factor_);
        }

        DetectionGroup dg(color_mat, depth_scaled, sensor_pose, doors, K, color_image->header.stamp.toSec());
        object_lock_.lock();
        obj_detection_buf_.push(dg);
        object_lock_.unlock();
    }
    cv::imshow(color_image->header.frame_id, image);
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

void SemanticSLAM::frontCamCallback(const sensor_msgs::ImageConstPtr& color_img){
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvShare(color_img, "bgr8");
    cv::Mat color_mat = cv_rgb_bridge->image.clone();
    cv::Mat image = cv_rgb_bridge->image.clone();
    vector<Detection> detections;
    obj_detector_->detectObjectYOLO(color_mat, detections);
    for(auto& m : detections){
        cv::rectangle(image, m.getRoI(), cv::Scalar(0, 0, 255), 2);
        cv::putText(image, m.getClassName(), m.getRoI().tl(), 1, 2, cv::Scalar(0, 0, 255));    
    }
    cv::imshow("SEGM", image);
    cv::waitKey(1);
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
    vector<const DetectionGroup*> kf_dets;
    kf->getDetection(kf_dets);
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
    for(const auto& dg : kf_dets){
        Eigen::Matrix3f K = dg->getIntrinsic();
        gtsam::Key sensor_id = dg->getSensorPose() == Eigen::Matrix4f::Identity() ? X(kf->id()) : S(kf->id());
        gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(K(0, 0), K(1, 1), 0.0, K(0, 2), K(1, 2)));
        vector<const Detection*> detections;
        dg->detections(detections);
        Eigen::Matrix4f cam_in_map = kf->getPose()* dg->getSensorPose();
        if(!isam_.valueExists(sensor_id) && !new_values_.exists(sensor_id)){
            new_values_.insert(sensor_id, gtsam::Pose3(cam_in_map.cast<double>()));
            gtsam_values_.insert(sensor_id, gtsam::Pose3(cam_in_map.cast<double>()));
            auto mount_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished());
            gtsam::BetweenFactor<gtsam::Pose3> mount(X(kf->id()), sensor_id, gtsam::Pose3(dg->getSensorPose().cast<double>()), mount_noise);
            gtsam_factors_.add(mount);
            new_factors_.add(mount);
        }
        for(auto& det : detections){
            double min_err = 10000.0;
            int idx = -1;
            cv::Rect roi = det->getRoI();
            cv::Point roi_center = roi.tl() + cv::Point(roi.width/ 2, roi.height/ 2);
            for(int i = 0; i < tgt_objs.size(); ++i){   
                Eigen::Vector3f obj_centroid = tgt_objs[i]->getCentroid();
                Eigen::Vector4f obj_homo(obj_centroid(0), obj_centroid(1), obj_centroid(2), 1.0);
                Eigen::Vector4f Tco = cam_in_map.inverse() * obj_homo;
                if(Tco(2) < 0.0){
                    continue;
                }
                Eigen::Vector3f pix_homo = K *Eigen::MatrixXf::Identity(3, 4) * Tco;
                pix_homo = pix_homo / pix_homo(2);
                cv::Point pix = cv::Point(pix_homo(0), pix_homo(1));
                cv::Point2d diff = (roi_center - pix);
                double err = sqrt(diff.x*diff.x + diff.y*diff.y);
                if(err < min_err){
                    min_err = err;
                    idx = i;
                }
                
            }

            bool matched = true;
            if(idx == -1){
                matched = false;
            }
            else{
                matched = (tgt_objs[idx]->getClassName() == det->getClassName() ? min_err < 200.0 : min_err < 150.0);
            }


            if(matched){ // matched
                pcl::PointCloud<pcl::PointXYZRGB> cloud;
                det->getCloud(cloud);
                Object* best_obj = tgt_objs[idx];
                best_obj->addDetection(det);
                gtsam::noiseModel::Isotropic::shared_ptr pix_noise = gtsam::noiseModel::Isotropic::Sigma(2, 300.0);    
                gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2> ibf(gtsam::Point2(roi_center.x, roi_center.y), pix_noise, sensor_id, O(best_obj->id()), K_gtsam);
                gtsam_factors_.add(ibf);
                new_factors_.add(ibf);
            }
            else{ //initialize
                pcl::PointCloud<pcl::PointXYZRGB> cloud;
                det->getCloud(cloud);
                if(!cloud.empty()){
                    // pcl::PointCloud<pcl::PointXYZRGB> cloud_tf;
                    // pcl::transformPointCloud(cloud, cloud_tf, cam_in_map);
                    // pcl::PointXYZRGB pt;
                    // pcl::computeCentroid(cloud_tf, pt);
                    Object* new_obj = new Object(det->getClassName(), last_oid_ == -1 ? 0 : last_oid_ + 1);
                    Eigen::Vector4f Pco = Eigen::Vector4f::Ones();
                    Pco.block<3, 1>(0,0) = det->center3D();
                    Eigen::Vector4f Pwo = cam_in_map * Pco;
                    new_obj->addDetection(det);
                    new_obj->setCentroid(Eigen::Vector3f(Pwo(0), Pwo(1), Pwo(2)));
                    h_graph_.insert(kf->getFloor(), new_obj);
                    auto init_obj_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 1.0, 1.0, 1.0).finished());
                    gtsam::PriorFactor<gtsam::Point3> opf(O(new_obj->id()), gtsam::Point3(new_obj->getCentroid().cast<double>()), init_obj_noise);
                    new_values_.insert(O(new_obj->id()), gtsam::Point3(new_obj->getCentroid().cast<double>()));
                    gtsam_values_.insert(O(new_obj->id()), gtsam::Point3(new_obj->getCentroid().cast<double>()));
                    gtsam_factors_.add(opf);
                    new_factors_.add(opf);
                    gtsam::noiseModel::Isotropic::shared_ptr pix_noise = gtsam::noiseModel::Isotropic::Sigma(2, 100.0);
                    gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2> ibf(gtsam::Point2(roi_center.x, roi_center.y), pix_noise, sensor_id, O(new_obj->id()), K_gtsam);
                    gtsam_factors_.add(ibf);
                    new_factors_.add(ibf);
                    last_oid_ = new_obj->id();
                }
            }
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
                if(f->isInlier(kf)){
                    floor_ = f;
                    break;
                }
            }
        }
    }
    
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
        object_lock_.lock();
        while(!obj_detection_buf_.empty()){
            double obj_stamp = obj_detection_buf_.front().stamp();
            if(obj_stamp > stamp){
                break;
            }
            if(stamp - obj_stamp < 0.1){
                detection_groups.push_back(obj_detection_buf_.front());
            }
            obj_detection_buf_.pop();
        }
        object_lock_.unlock();
        KeyFrame* new_kf = new KeyFrame(orb_kf->mnId, orb_kf->GetPoseInverse().matrix());
        new_kf->color_ = orb_kf->color_;
        new_kf->depth_ = orb_kf->depth_;
        addKeyFrame(new_kf, detection_groups);
        kf_updated_ = false;    
        key_lock.unlock();
        publishPath();
        
        pcl::PointCloud<pcl::PointXYZRGB> obj_cloud;
        for(const auto& obj : h_graph_.getEveryObjects()){
            // pcl::PointCloud<pcl::PointXYZRGB> ocl;
            // obj->getCloud(ocl);
            // obj_cloud += ocl;
            Eigen::Vector3f centroid = obj->getCentroid();
            pcl::PointXYZRGB pt;
            pt.x = centroid(0);
            pt.y = centroid(1);
            pt.z = centroid(2);
            pt.r = 255.0;
            pt.g = 255.0;
            pt.b = 255.0;
            obj_cloud.push_back(pt);
        }
        
        sensor_msgs::PointCloud2 obj_cloud_ros;
        pcl::toROSMsg(obj_cloud, obj_cloud_ros);
        obj_cloud_ros.header.frame_id = "map_optic";
        obj_cloud_ros.header.stamp = ros::Time::now();
        pub_object_cloud_.publish(obj_cloud_ros);

        if(kfs_.size() % 10 == 0){
            pcl::PointCloud<pcl::PointXYZRGB> map_cloud;
            getMapCloud(map_cloud);
            sensor_msgs::PointCloud2 map_cloud_ros;
            pcl::toROSMsg(map_cloud, map_cloud_ros);
            map_cloud_ros.header.stamp = ros::Time::now();
            map_cloud_ros.header.frame_id = "map_optic";
            pub_map_cloud_.publish(map_cloud_ros);
        }

        // sensor_msgs::PointCloud2 map_cloud_ros;
        // pcl::toROSMsg(map_cloud, map_cloud_ros);
        // map_cloud_ros.header.stamp = ros::Time::now();
        // map_cloud_ros.header.frame_id = "map_optic";
        // pub_map_cloud_.publish(map_cloud_ros);

        // // visualization_msgs::MarkerArray h_graph_vis;
        // // visualizeHGraph(h_graph, h_graph_vis);
        // // pub_h_graph_.publish(h_graph_vis);     
    }
}

void SemanticSLAM::loopQueryCallback(){
    while(true){
        unique_lock<mutex> key_lock(keyframe_lock_);
        keyframe_cv_.wait(key_lock, [this]{return !this->lc_buf_.empty() || this->kill_flag_;});
        if(kill_flag_){
            thread_killed_ = true;
            break;
        }
        while(!lc_buf_.empty()){
            ORB_SLAM3::LoopQuery lq = lc_buf_.front();
            if(isam_.valueExists(X(lq.id_query)) && isam_.valueExists(X(lq.id_target)) &&(kfs_[lq.id_query]->getFloor() == kfs_[lq.id_target]->getFloor())){
                auto loop_noise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2).finished());
                gtsam::BetweenFactor<gtsam::Pose3> lc((X(lq.id_target)), X(lq.id_query), gtsam::Pose3(lq.drift.cast<double>()), loop_noise_);
                gtsam_factors_.add(lc);
                new_factors_.add(lc);
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
        
        //h_graph_.refineObject();
    }
}

void SemanticSLAM::getMapCloud(pcl::PointCloud<pcl::PointXYZRGB>& output){
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
    pcl::VoxelGrid<pcl::PCLPointCloud2> voxel;
    voxel.setLeafSize(0.1, 0.1, 0.1);
    for(size_t i = 0; i < kfs_.size(); i += 10){
        if(kfs_.find(i) == kfs_.end()){
            continue;
        }
        KeyFrame* kf = kfs_[i];
        //=====Generate Cloud====
        pcl::PointCloud<pcl::PointXYZRGB> raw_cloud, cloud_tf;
        pcl::PCLPointCloud2 tf_pcl2;
        for(int r = 0; r < kf->color_.rows; r += 3){
            for(int c = 0; c < kf->color_.cols; c += 3){
                float depth = kf->depth_.at<float>(r, c);
                if(isnanf(depth) || depth < 1.0e-4 || depth > 5.0){
                    continue;
                }
                Eigen::Vector3f pix(c, r, 1.0);
                float x = (c - K_front_(0, 2)) * depth / K_front_(0, 0);
                float y = (r - K_front_(1, 2)) * depth / K_front_(1, 1);
                pcl::PointXYZRGB pt;
                pt.x = x;
                pt.y = y;
                pt.z = depth;
                pt.r = kf->color_.at<cv::Vec3b>(r, c)[2];
                pt.g = kf->color_.at<cv::Vec3b>(r, c)[1];
                pt.b = kf->color_.at<cv::Vec3b>(r, c)[0];
                raw_cloud.push_back(pt);
            }
        }
        pcl::transformPointCloud(raw_cloud, cloud_tf, kf->getPose());
        pcl::toPCLPointCloud2(cloud_tf, tf_pcl2);
        *cloud += tf_pcl2;
        voxel.setInputCloud(cloud);
        voxel.filter(*cloud);
        //=======================
    }
    pcl::fromPCLPointCloud2(*cloud, output);
    //output = *cloud;
}

void SemanticSLAM::visualizeHGraph(const HGraph& h_graph, visualization_msgs::MarkerArray& output){
    // output.markers.clear();
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


#include <semantic_slam/api_class/raw_wrapper.h>

RawWrapper::RawWrapper(): pnh_("~"), kf_updated_(false), kill_flag_(false), thread_killed_(true){
    string voc_file;
    pnh_.param<string>("vocabulary_file", voc_file, "/home/nuninu98/catkin_ws/src/orb_semantic_slam/model/ORBvoc.txt");

    string setting_file;
    pnh_.param<string>("setting_file", setting_file, "/home/nuninu98/catkin_ws/src/orb_semantic_slam/setting/tum_rgbd.yaml");

    cv::FileStorage fsSettings(setting_file.c_str(), cv::FileStorage::READ);
    K_front_ = Eigen::Matrix3f::Identity();
    K_front_(0, 0) = static_cast<float>(fsSettings["Camera1.fx"]);
    K_front_(0, 2) = static_cast<float>(fsSettings["Camera1.cx"]);
    K_front_(1, 1) = static_cast<float>(fsSettings["Camera1.fy"]);
    K_front_(1, 2) = static_cast<float>(fsSettings["Camera1.cy"]);

    visual_odom_ = new ORB_SLAM3::System(voc_file, setting_file ,ORB_SLAM3::System::RGBD, false);
    visual_odom_->registerKeyframeCall(&kf_updated_, &keyframe_cv_);
    visual_odom_->registerLoopCall(&lc_buf_, &loop_cv_, &loop_lock_);
    pub_path_ = nh_.advertise<nav_msgs::Path>("raw_path", 1);

    string rgb_topic, depth_topic, imu_topic;
    pnh_.param<string>("rgb_topic", rgb_topic, "/camera/color/image_raw");
    pnh_.param<string>("depth_topic", depth_topic, "/camera/aligned_depth_to_color/image_raw");
    pnh_.param<string>("imu_topic", imu_topic, "/imu/data");

    keyframe_thread_ = thread(&RawWrapper::keyframeCallback, this);
    keyframe_thread_.detach();

    loop_thread_ = thread(&RawWrapper::loopQueryCallback, this);
    loop_thread_.detach();

    //sub_yolo_  = nh_.subscribe("/side/color/image_raw/yolo", 1, &RawWrapper::detectionImageCallback, this);

    tracking_color_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, rgb_topic, 1));
    tracking_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, depth_topic, 1));
    tracking_sync_.reset(new message_filters::Synchronizer<sync_pol> (sync_pol(1000), *tracking_color_, *tracking_depth_));
    tracking_sync_->registerCallback(boost::bind(&RawWrapper::trackingImageCallback, this, _1, _2));
    record_sub_ = nh_.subscribe("record_flag", 1000, &RawWrapper::recordCallback, this);
    ocr_.reset(new OCR("/home/nuninu98/Downloads/crnn_cs.onnx", "/home/nuninu98/Downloads/alphabet_94.txt"));

}

RawWrapper::~RawWrapper(){
    kill_flag_ = true;
    keyframe_cv_.notify_all();
    visual_odom_->Shutdown();
}

void RawWrapper::trackingImageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image){
    ros::Time stamp = rgb_image->header.stamp;
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvShare(rgb_image, "bgr8");
    cv_bridge::CvImageConstPtr cv_depth_bridge = cv_bridge::toCvShare(depth_image, depth_image->encoding);
    keyframe_lock_.lock();
    Eigen::Matrix4f act_pose = visual_odom_->TrackRGBD(cv_rgb_bridge->image, cv_depth_bridge->image, stamp.toSec()).matrix().inverse();
    keyframe_lock_.unlock();
    cv::imshow("view", cv_rgb_bridge->image);
    cv::waitKey(1);
    Eigen::Matrix4f optic_in_map = act_pose;
    Eigen::Matrix4f base_in_map = optic_in_map * OPTIC_TF.inverse();
    vector<geometry_msgs::TransformStamped> tfs;
    geometry_msgs::TransformStamped tf;
    tf.header.frame_id = "map_optic";
    tf.header.stamp = ros::Time::now();
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
    tf_offset.header.stamp = ros::Time::now();
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

void RawWrapper::keyframeCallback(){
    while(true){
        unique_lock<mutex> key_lock(keyframe_lock_);
        keyframe_cv_.wait(key_lock, [this]{return this->kf_updated_ || this->kill_flag_;});
        if(kill_flag_){
            thread_killed_ = true;
            break;
        }
        vector<ORB_SLAM3::KeyFrame*> keyframes = visual_odom_->getKeyFrames();
        sort(keyframes.begin(), keyframes.end(), []( ORB_SLAM3::KeyFrame* k1,  ORB_SLAM3::KeyFrame* k2){
            return k1->mnId < k2->mnId;
        });
        stamp_.insert({keyframes.back()->mnId, ros::Time::now().toSec()});
        nav_msgs::Path path;
        path.header.frame_id = "map_optic";
        path.header.stamp = ros::Time::now();
        for(const auto& k : keyframes){
            Eigen::Matrix4f k_pose = k->GetPoseInverse().matrix();
            geometry_msgs::PoseStamped p;
            p.pose.position.x = k_pose(0, 3);
            p.pose.position.y = k_pose(1, 3);
            p.pose.position.z = k_pose(2, 3);
            path.poses.push_back(p);
        }
        pub_path_.publish(path);
        detection_lock_.lock();
        string room_number = "";
        if((ros::Time::now() - det_stamp_).toSec() < 0.1){
            room_number = det_.getClassName();
            size_t kf_id = keyframes.back()->mnId;
            kfID_room_.insert({kf_id, room_number});
            kfID_images_.insert({kf_id, det_image_});
        }
        
        detection_lock_.unlock();
        kf_updated_ = false;

    }
}

void RawWrapper::loopQueryCallback(){
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
                loops_.push_back({lq.id_query, lq.id_target});
                // if(kfID_room_[lq.id_query] != kfID_room_[lq.id_target] && (!kfID_room_[lq.id_query].empty() && !kfID_room_[lq.id_target].empty())){
                //     cout<<"DIFF ROOM: "<<kfID_room_[lq.id_query]<<" "<<kfID_room_[lq.id_target]<<endl;
                //     string folder = "/home/nuninu98/test_orbloop/"+to_string(lq.id_query)+"/";
                //     if(!boost::filesystem::exists(folder)){
                //         boost::filesystem::create_directories(folder);
                //     }
                //     cv::Mat match_image;
                //     cv::drawMatches(kfID_images_[lq.id_query], vector<cv::KeyPoint>(), kfID_images_[lq.id_target], vector<cv::KeyPoint>(), vector<cv::DMatch>(), match_image);
                //     string filename = folder + to_string(lq.id_query)+"_"+to_string(lq.id_target)+"_number_match";
                //     cv::imwrite(filename+".png", match_image);
                // }
            }
            
            lc_buf_.pop();
            
        }
    }
}

void RawWrapper::detectionImageCallback(const yolo_protocol::YoloResultConstPtr& yolo_result){
    unique_lock<mutex> lock(detection_lock_);
    for(int i = 0; i < yolo_result->detections.detections.size(); ++i){
        sensor_msgs::ImageConstPtr color_img = boost::make_shared<sensor_msgs::Image const>(yolo_result->original);
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvShare(color_img, "bgr8");
        cv::Mat image = cv_rgb_bridge->image.clone();

        auto detect = yolo_result->detections.detections[i];
        cv::Rect roi(cv::Point(detect.bbox.center.x- detect.bbox.size_x/2, detect.bbox.center.y - detect.bbox.size_y/2), cv::Size(detect.bbox.size_x, detect.bbox.size_y));
        cv::Mat mask;
        
        if(detect.header.frame_id == "room_number"){
            OCRDetection text_out;
            bool found_txt = ocr_->textRecognition(image, roi, text_out);
            if(found_txt){
                // cv::rectangle(image, roi, cv::Scalar(0, 0, 255), 2);
                // cv::putText(image, m->getClassName(), roi.tl(), 1, 2, cv::Scalar(0, 0, 255));
                det_ = Detection(roi, cv::Mat(), text_out.getContent());
                det_stamp_ = ros::Time::now();
                det_image_ = image;
                break;
            }
        }
    }
    
}

void RawWrapper::recordCallback(const std_msgs::BoolConstPtr& msg){
    string folder = "/home/nuninu98/";

    if(!boost::filesystem::exists(folder)){
        boost::filesystem::create_directories(folder);
    }
    string filename = "orbslam.txt";
    ofstream traj_file(folder + filename);
    vector<ORB_SLAM3::KeyFrame*> keyframes = visual_odom_->getKeyFrames();
    sort(keyframes.begin(), keyframes.end(), []( ORB_SLAM3::KeyFrame* k1,  ORB_SLAM3::KeyFrame* k2){
        return k1->mnId < k2->mnId;
    });
    for(size_t i = 0; i < keyframes.size(); ++i){
        if(stamp_.find(keyframes[i]->mnId) == stamp_.end()){
            cout<<"DROP? "<<endl;
            continue;
        }
        
        Eigen::Matrix4f se3 = OPTIC_TF* keyframes[i]->GetPoseInverse().matrix();
        traj_file <<to_string(stamp_[keyframes[i]->mnId])<<" "<< se3(0, 3)<<" "<<se3(1, 3)<<" "<<se3(2, 3)<<endl;
    }

    filename = "orbslam_loop.txt";
    ofstream loop_file(folder + filename);
    loop_lock_.lock();
    for(size_t i = 0; i< loops_.size(); ++i){
        loop_file << loops_[i].first<<" "<<loops_[i].second<<endl;
    }
    loop_lock_.unlock();
    cout<<"WRITTEN!"<<endl;
}
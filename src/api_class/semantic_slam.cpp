#include <semantic_slam/api_class/semantic_slam.h>
SemanticSLAM::SemanticSLAM(): pnh_("~"), kill_flag_(false), thread_killed_(false), keyframe_updated_(false){

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
    door_detector.reset(new LandmarkDetector(door_detection_onnx, door_detection_classes));

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
    obj_detector.reset(new LandmarkDetector(obj_detection_onnx, obj_detection_classes));
    

    ocr_.reset(new OCR(crnn_file, text_list));

    optic_in_base_ = Eigen::Matrix4f::Identity();
    optic_in_base_(0, 0) = 0.0;
    optic_in_base_(0, 2) = 1.0;
    optic_in_base_(1, 0) = -1.0;
    optic_in_base_(1, 1) = 0.0;
    optic_in_base_(2, 1) = -1.0;
    optic_in_base_(2, 2) = 0.0;

    pub_path_ = nh_.advertise<nav_msgs::Path>("slam_path", 1);

    visual_odom_ = new ORB_SLAM3::System(voc_file, setting_file ,ORB_SLAM3::System::RGBD, false);
    visual_odom_->registerKeyframeCall(&keyframe_updated_, &keyframe_cv_);
    keyframe_thread_ = thread(&SemanticSLAM::keyframeCallback, this);
    keyframe_thread_.detach();
    
    sub_detection_image_ = nh_.subscribe("/d435/color/image_raw", 1, &SemanticSLAM::detectionImageCallback, this);

    string rgb_topic, depth_topic, imu_topic;
    pnh_.param<string>("rgb_topic", rgb_topic, "/camera/color/image_raw");
    pnh_.param<string>("depth_topic", depth_topic, "/camera/aligned_depth_to_color/image_raw");
    pnh_.param<string>("imu_topic", imu_topic, "/imu/data");
    rgb_subscriber_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, rgb_topic, 1));
    depth_subscriber_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, depth_topic, 1));
    
    sync_.reset(new message_filters::Synchronizer<sync_pol> (sync_pol(1000), *rgb_subscriber_, *depth_subscriber_));
    sync_->registerCallback(boost::bind(&SemanticSLAM::trackingImageCallback, this, _1, _2));
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

    vector<Detection> detections;
    object_lock_.lock();
    while(!obj_detection_buf_.empty()){
        double obj_stamp = obj_detection_buf_.front().first.toSec();
        if(obj_stamp > stamp.toSec()){
            break;
        }
        if(stamp.toSec() - obj_stamp < 0.1){
            for(const auto& elem : obj_detection_buf_.front().second){
                detections.push_back(elem);
            }
        }
        obj_detection_buf_.pop();
    }
    object_lock_.unlock();
    
    keyframe_lock_.lock();
    ros::Time tic = ros::Time::now();
    Eigen::Matrix4f cam_extrinsic = visual_odom_->TrackRGBD(cv_rgb_bridge->image, cv_depth_bridge->image, stamp.toSec(), detections, imu_points).matrix();
    //cout<<(ros::Time::now() - tic).toSec()*1000.0<<"ms"<<endl;
    keyframe_lock_.unlock();
    Eigen::Matrix4f optic_in_map = cam_extrinsic.inverse();
    Eigen::Matrix4f base_in_map = optic_in_map * optic_in_base_.inverse();

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
    Eigen::Quaternionf q_tf_basecam(optic_in_base_.block<3, 3>(0, 0));
    tf_offset.transform.rotation.w = q_tf_basecam.w();
    tf_offset.transform.rotation.x = q_tf_basecam.x();
    tf_offset.transform.rotation.y = q_tf_basecam.y();
    tf_offset.transform.rotation.z = q_tf_basecam.z();
    tfs.push_back(tf_offset);

    broadcaster_.sendTransform(tfs);
    //Eigen::Matrix4d cam_extrinsic = visual_odom_->TrackRGBD(cv_rgb_bridge->image, cv_depth_bridge->image, stamp.toSec(), imu_points).matrix().cast<double>();
    //cout<<"dt: "<<(ros::Time::now() - tic).toSec()<<endl;
}

void SemanticSLAM::detectionImageCallback(const sensor_msgs::ImageConstPtr& color_image){
    cv_bridge::CvImageConstPtr bridge = cv_bridge::toCvShare(color_image, "bgr8");
    cv::Mat image = bridge->image;
    vector<Detection> detections = door_detector->detectObjectYOLO(image);
    //vector<OCRDetection> text_detections = ocr_->detect_rec(image);
    vector<Detection> doors;
    for(auto& m : detections){
        //==========Testing Room Number=========
        if(m.getClassName() == "room_number"){
            cv::Rect roi = m.getRoI();
            OCRDetection text_out;
            bool found_txt = ocr_->textRecognition(image, roi, text_out);
            if(found_txt){
                m.copyContent(text_out);
                doors.push_back(m);
                cv::rectangle(image, m.getRoI(), cv::Scalar(0, 0, 255), 2);
                cv::putText(image, text_out.getContent(), m.getRoI().tl(), 1, 2, cv::Scalar(0, 0, 255));
            }
        }
        else{
            
        }
        // if(m.getClassName() == "room_number"){
        //     cv::Rect r1 = m.getRoI();
        //     double max_iou = 0.2;
        //     int max_iou_idx = -1;
        //     for(int i = 0; i < text_detections.size(); ++i){
        //         cv::Rect r2 = text_detections[i].getRoI();
        //         cv::Rect common = (r1 & r2);
        //         double iou = (double)common.area() / (double)(r1.area() + r2.area() - common.area());
        //         if(iou > max_iou){
        //             max_iou = iou;
        //             max_iou_idx = i;
        //         }
        //     }
        //     if(max_iou_idx == -1){
        //         continue;
        //     }
        //     m.copyContent(text_detections[max_iou_idx]);
        //     doors.push_back(m);
            
        //     cv::rectangle(image, m.getRoI(), cv::Scalar(0, 0, 255), 2);
        //     cv::putText(image, text_detections[max_iou_idx].getContent(), m.getRoI().tl(), 1, 2, cv::Scalar(0, 0, 255));
        // }
        // else{

        // }
        //====================================== 
    }

    if(!doors.empty()){
        object_lock_.lock();
        obj_detection_buf_.push(make_pair(color_image->header.stamp, doors));
        object_lock_.unlock();
    }
    cv::imshow("detection", image);
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

void SemanticSLAM::keyframeCallback(){
    while(true){
        unique_lock<mutex> key_lock(keyframe_lock_);
        keyframe_cv_.wait(key_lock, [this]{return this->keyframe_updated_ || this->kill_flag_;});
        if(kill_flag_){
            thread_killed_ = true;
            break;
        }
        vector<ORB_SLAM3::KeyFrame*> keys = visual_odom_->getKeyframes();
        keyframe_updated_ = false;
        key_lock.unlock();
        sort(keys.begin(), keys.end(), [](ORB_SLAM3::KeyFrame* k1, ORB_SLAM3::KeyFrame* k2){
            return k1->mnId < k2->mnId;
        });
        nav_msgs::Path path;
        path.header.frame_id = "map_optic";
        path.header.stamp = ros::Time::now();
        for(int i = 0; i < keys.size(); ++i){
            auto k = keys[i];
            Eigen::Matrix4f pose = k->GetPose().matrix().inverse() * optic_in_base_.inverse();
            geometry_msgs::PoseStamped p;
            p.pose.position.x = pose(0, 3);
            p.pose.position.y = pose(1, 3);
            p.pose.position.z = pose(2, 3);
            //cout<<p.pose.position.x<<" "<<p.pose.position.y<<" "<<p.pose.position.z<<endl;
            path.poses.push_back(p);
        }
        pub_path_.publish(path);
        
    }
}
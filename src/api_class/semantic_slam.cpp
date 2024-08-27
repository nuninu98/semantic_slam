#include <semantic_slam/api_class/semantic_slam.h>
SemanticSLAM::SemanticSLAM(): pnh_("~"), kill_flag_(false), thread_killed_(false), keyframe_updated_(false), depth_factor_(1000.0){

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
    K_side_(0, 0) = 645.3115844726562;
    K_side_(0, 2) = 644.2869873046875;
    K_side_(1, 1) = 644.4506225585938;
    K_side_(1, 2) = 361.4469299316406;

    pub_path_ = nh_.advertise<nav_msgs::Path>("slam_path", 1);

    cv::FileStorage fsSettings(setting_file.c_str(), cv::FileStorage::READ);
    K_front_ = Eigen::Matrix3f::Identity();
    K_front_(0, 0) = static_cast<float>(fsSettings["Camera1.fx"]);
    K_front_(0, 2) = static_cast<float>(fsSettings["Camera1.cx"]);
    K_front_(1, 1) = static_cast<float>(fsSettings["Camera1.fy"]);
    K_front_(1, 2) = static_cast<float>(fsSettings["Camera1.cy"]);

    visual_odom_ = new ORB_SLAM3::System(voc_file, setting_file ,ORB_SLAM3::System::RGBD, false);
    visual_odom_->registerKeyframeCall(&keyframe_updated_, &keyframe_cv_);
    keyframe_thread_ = thread(&SemanticSLAM::keyframeCallback, this);
    keyframe_thread_.detach();
    
   
    string rgb_topic, depth_topic, imu_topic;
    pnh_.param<string>("rgb_topic", rgb_topic, "/camera/color/image_raw");
    pnh_.param<string>("depth_topic", depth_topic, "/camera/aligned_depth_to_color/image_raw");
    pnh_.param<string>("imu_topic", imu_topic, "/imu/data");

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

    //sub_frontcam_ = nh_.subscribe(rgb_topic, 1, &SemanticSLAM::frontCamCallback, this);
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

    vector<ORB_SLAM3::DetectionGroup> detection_groups;
    object_lock_.lock();
    while(!obj_detection_buf_.empty()){
        double obj_stamp = obj_detection_buf_.front().stamp();
        if(obj_stamp > stamp.toSec()){
            break;
        }
        if(stamp.toSec() - obj_stamp < 0.1){
            detection_groups.push_back(obj_detection_buf_.front());
        }
        obj_detection_buf_.pop();
    }
    object_lock_.unlock();
    keyframe_lock_.lock();
    ros::Time tic = ros::Time::now();
    Eigen::Matrix4f cam_extrinsic = visual_odom_->TrackRGBD(cv_rgb_bridge->image, cv_depth_bridge->image, stamp.toSec(), detection_groups, imu_points).matrix();
    //cout<<(ros::Time::now() - tic).toSec()*1000.0<<"ms"<<endl;
    keyframe_lock_.unlock();
    Eigen::Matrix4f optic_in_map = cam_extrinsic.inverse();
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
    //Eigen::Matrix4d cam_extrinsic = visual_odom_->TrackRGBD(cv_rgb_bridge->image, cv_depth_bridge->image, stamp.toSec(), imu_points).matrix().cast<double>();
    //cout<<"dt: "<<(ros::Time::now() - tic).toSec()<<endl;
}

void SemanticSLAM::detectionImageCallback(const sensor_msgs::ImageConstPtr& color_image, const sensor_msgs::ImageConstPtr& depth_image, const Eigen::Matrix4f& sensor_pose, const Eigen::Matrix3f& K){
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvShare(color_image, "bgr8");
    cv_bridge::CvImageConstPtr cv_depth_bridge = cv_bridge::toCvShare(depth_image, depth_image->encoding);
    cv::Mat image = cv_rgb_bridge->image.clone();
    vector<ORB_SLAM3::Detection> detections = door_detector_->detectObjectYOLO(image);
    //vector<OCRDetection> text_detections = ocr_->detect_rec(image);
    vector<ORB_SLAM3::Detection> doors;
    
    for(auto& m : detections){
        //m.sensor_pose_ = sensor_pose;
        //==========Testing Room Number=========
        if(m.getClassName() == "room_number"){
            cv::Rect roi = m.getRoI();
            ORB_SLAM3::OCRDetection text_out;
            bool found_txt = ocr_->textRecognition(image, roi, text_out);
            if(found_txt){
                m.copyContent(text_out);
                doors.push_back(m);
                cv::rectangle(image, m.getRoI(), cv::Scalar(0, 0, 255), 2);
                cv::putText(image, text_out.getContent(), m.getRoI().tl(), 1, 2, cv::Scalar(0, 0, 255));
            }
        }
        else if(m.getClassName() == "floor_sign"){
            cv::Rect roi = m.getRoI();
            cv::rectangle(image, m.getRoI(), cv::Scalar(0, 0, 255), 2);
            cv::putText(image, m.getClassName(), m.getRoI().tl(), 1, 2, cv::Scalar(0, 0, 255));
            ORB_SLAM3::OCRDetection text_out;
            bool found_txt = ocr_->textRecognition(image, roi, text_out);
            if(found_txt){
                cout<<"FLOOR SIGN: "<<text_out.getContent()<<endl;
            }
            else{
                cout<<"NOOOO"<<endl;
            }
            doors.push_back(m);
        }
    }
   

    if(!doors.empty()){
        cv::Mat color_mat = cv_rgb_bridge->image.clone();
        cv::Mat depth_mat = cv_depth_bridge->image.clone();
        cv::Mat depth_scaled;
        if((fabs(depth_factor_-1.0)>1e-5) || depth_mat.type()!=CV_32F){
            depth_mat.convertTo(depth_scaled,CV_32F, 1.0/depth_factor_);
        }

        ORB_SLAM3::DetectionGroup dg(color_mat, depth_scaled, sensor_pose, doors, K, color_image->header.stamp.toSec());
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
    auto detections = obj_detector_->detectObjectYOLO(color_mat);
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
        pcl::PointCloud<pcl::PointXYZRGB> map_cloud;
        visual_odom_->getMapPointCloud(map_cloud);
        key_lock.unlock();
        sort(keys.begin(), keys.end(), [](ORB_SLAM3::KeyFrame* k1, ORB_SLAM3::KeyFrame* k2){
            return k1->mnId < k2->mnId;
        });
        nav_msgs::Path path;
        path.header.frame_id = "map_optic";
        path.header.stamp = ros::Time::now();
        for(int i = 0; i < keys.size(); ++i){
            auto k = keys[i];
            Eigen::Matrix4f pose = k->GetPose().matrix().inverse() * OPTIC_TF.inverse();
            geometry_msgs::PoseStamped p;
            p.pose.position.x = pose(0, 3);
            p.pose.position.y = pose(1, 3);
            p.pose.position.z = pose(2, 3);
            //cout<<p.pose.position.x<<" "<<p.pose.position.y<<" "<<p.pose.position.z<<endl;
            path.poses.push_back(p);
        }
        pub_path_.publish(path);


        unordered_map<int, vector<ORB_SLAM3::Object*>> h_graph;
        visual_odom_->getHierarchyGraph(h_graph);
        pcl::PointCloud<pcl::PointXYZRGB> obj_cloud;

        // pcl::PointCloud<pcl::PointXYZLNormal>::Ptr key_poses(new pcl::PointCloud<pcl::PointXYZLNormal>());
        // pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // pcl::ModelCoefficients coeffs;
        vector<ORB_SLAM3::KeyFrame*> kf_xyzn;
        for(const auto& fo_pair : h_graph){
            for(const auto& obj : fo_pair.second){
                pcl::PointCloud<pcl::PointXYZRGB> ocl;
                obj->getCloud(ocl);
                obj_cloud += ocl;
                vector<ORB_SLAM3::KeyFrame*> kfs;
                obj->getConnectedKeyFrames(kfs);
                for(const auto& kf : kfs){
                    kf_xyzn.push_back(kf);
                }
            }
        }
        //==========Test RANSAC===============
        ORB_SLAM3::KeyFrame* plane_kf = nullptr;
        if(!kf_xyzn.empty()){
            plane_kf = normalPlaneRansac(kf_xyzn);
        }
        if(plane_kf != nullptr){
            Eigen::Matrix4f floor_pose = plane_kf->GetPoseInverse().matrix();
            geometry_msgs::TransformStamped tf_plane;
            tf_plane.header.frame_id = "map_optic";
            tf_plane.child_frame_id = "floor";
            tf_plane.header.stamp = ros::Time::now();
            tf_plane.transform.translation.x = floor_pose(0, 3);
            tf_plane.transform.translation.y = floor_pose(1, 3);
            tf_plane.transform.translation.z = floor_pose(2, 3);

            Eigen::Quaternionf q(floor_pose.block<3, 3>(0, 0));
            tf_plane.transform.rotation.w = q.w();
            tf_plane.transform.rotation.x = q.x();
            tf_plane.transform.rotation.y = q.y();
            tf_plane.transform.rotation.z = q.z();
            broadcaster_.sendTransform(tf_plane);

            //plane visualization
            visualization_msgs::Marker plane;
            plane.header.stamp = ros::Time::now();
            plane.header.frame_id = "floor";
            plane.type = visualization_msgs::Marker::CUBE;
            plane.scale.x = 30.0;
            plane.scale.y = 0.03;
            plane.scale.z = 30.0;
            plane.pose.position.x = 0.0;
            plane.pose.position.y = 0.0;
            plane.pose.position.z = 0.0;
            plane.pose.orientation.x = 0.0;
            plane.pose.orientation.y = 0.0;
            plane.pose.orientation.z = 0.0;
            plane.pose.orientation.w = 1.0;
            plane.color.a = 0.5;
            plane.color.r = 0.0;
            plane.color.g = 0.0;
            plane.color.b = 255.0;
            pub_floor_.publish(plane);
        }
        
        //====================================

        //pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> ransac;
        // ransac.setModelType(pcl::SACMODEL_NORMAL_PLANE);   
        // ransac.setOptimizeCoefficients(true);
        // ransac.setInputCloud(key_poses);
        // ransac.segment(*inliers, coeffs);

        sensor_msgs::PointCloud2 obj_cloud_ros;
        pcl::toROSMsg(obj_cloud, obj_cloud_ros);
        obj_cloud_ros.header.frame_id = "map_optic";
        obj_cloud_ros.header.stamp = ros::Time::now();
        pub_object_cloud_.publish(obj_cloud_ros);

        // sensor_msgs::PointCloud2 map_cloud_ros;
        // pcl::toROSMsg(map_cloud, map_cloud_ros);
        // map_cloud_ros.header.stamp = ros::Time::now();
        // map_cloud_ros.header.frame_id = "map_optic";
        // pub_map_cloud_.publish(map_cloud_ros);

        visualization_msgs::MarkerArray h_graph_vis;
        visualizeHGraph(h_graph, h_graph_vis);
        pub_h_graph_.publish(h_graph_vis); 
        
       
    
        
    }
}

void SemanticSLAM::visualizeHGraph(const unordered_map<int, vector<ORB_SLAM3::Object*>>& h_graph, visualization_msgs::MarkerArray& output){
    output.markers.clear();
    unordered_set<const ORB_SLAM3::KeyFrame*> keys;
    size_t id = 0;
    vector<std_msgs::ColorRGBA> colors(4);
    colors[0].a = 1.0;
    colors[0].r = 255.0;
    colors[0].g = 255.0;
    colors[0].b = 255.0;

    colors[1].a = 1.0;
    colors[1].r = 255.0;
    colors[1].g = 0.0;
    colors[1].b = 0.0;

    colors[2].a = 1.0;
    colors[2].r = 0.0;
    colors[2].g = 255.0;
    colors[2].b = 0.0;

    colors[3].a = 1.0;
    colors[3].r = 0.0;
    colors[3].g = 0.0;
    colors[3].b = 255.0;
    for(const auto& fo_pair : h_graph){
        int floor = fo_pair.first;
        if(floor == 9){ // for test
            continue;
        }
        for(const auto& obj : fo_pair.second){
            pcl::PointCloud<pcl::PointXYZRGB> ocl;
            obj->getCloud(ocl);
            pcl::PointXYZRGB centroid;
            pcl::computeCentroid(ocl, centroid);
            visualization_msgs::Marker obj_marker;
            obj_marker.type = visualization_msgs::Marker::SPHERE;
            obj_marker.id = id;
            id++;
            obj_marker.header.stamp = ros::Time::now();
            obj_marker.header.frame_id ="map_optic";
            obj_marker.color = colors[(floor + 1) % 3];
            obj_marker.pose.position.x = centroid.x;
            obj_marker.pose.position.y = centroid.y;
            obj_marker.pose.position.z = centroid.z;
            obj_marker.pose.orientation.w = 1.0;
            obj_marker.pose.orientation.x = 0.0;
            obj_marker.pose.orientation.y = 0.0;
            obj_marker.pose.orientation.z = 0.0;
            obj_marker.scale.x = 0.2;
            obj_marker.scale.y = 0.2;
            obj_marker.scale.z = 0.2;
            output.markers.push_back(obj_marker);
            
            vector< ORB_SLAM3::KeyFrame*> conn_keys;
            obj->getConnectedKeyFrames(conn_keys);
            for(int i = 0; i < conn_keys.size(); i += 3){
                auto k = conn_keys[i];
                if(keys.find(k) != keys.end()){
                    continue;
                }
                visualization_msgs::Marker kf_marker;
                kf_marker.header.stamp = ros::Time::now();
                kf_marker.header.frame_id ="map_optic";
                kf_marker.color = colors[(floor + 1) % 3];
                kf_marker.id = id;
                id++;
                Eigen::Matrix4f pose = k->GetPoseInverse().matrix();
                Eigen::Quaternionf q(pose.block<3, 3>(0, 0));
                kf_marker.type = visualization_msgs::Marker::SPHERE;
                kf_marker.scale.x = 0.1;
                kf_marker.scale.y = 0.1;
                kf_marker.scale.z = 0.1;
                kf_marker.pose.position.x = pose(0, 3);
                kf_marker.pose.position.y = pose(1, 3);
                kf_marker.pose.position.z = pose(2, 3);
                kf_marker.pose.orientation.w = q.w(); 
                kf_marker.pose.orientation.x = q.x();
                kf_marker.pose.orientation.y = q.y();
                kf_marker.pose.orientation.z = q.z();
                output.markers.push_back(kf_marker);
                keys.insert(k);

                visualization_msgs::Marker edge_marker;
                edge_marker.header = kf_marker.header;
                edge_marker.id = id;
                id++;
                edge_marker.color.a = 1.0;
                edge_marker.color.r = 255.0;
                edge_marker.color.g = 255.0;
                edge_marker.color.b = 0.0;
                edge_marker.points.push_back(obj_marker.pose.position);
                edge_marker.points.push_back(kf_marker.pose.position);
                edge_marker.scale.x = 0.1;
                edge_marker.scale.y = 0.1;
                output.markers.push_back(edge_marker);
            }
        }
    }
}

ORB_SLAM3::KeyFrame* SemanticSLAM::normalPlaneRansac(const vector<ORB_SLAM3::KeyFrame*>& xyz_norms){
    float err_thresh = 0.7;
    float inlier_r = 0.8;
    float prob = 0.99;

    int iter = log(1.0 - prob) / log(1.0 - inlier_r);
    int max_inliers = 0;
    ORB_SLAM3::KeyFrame* output = nullptr;
    for(int i = 0; i < iter; ++i){
        int id = rand() % xyz_norms.size();
        float a = xyz_norms[id]->getPoseWithNormal()(3); // ax+by+cz+d = 0
        float b = xyz_norms[id]->getPoseWithNormal()(4);
        float c = xyz_norms[id]->getPoseWithNormal()(5);
        float d = -(a*xyz_norms[id]->getPoseWithNormal()(0) + b*a*xyz_norms[id]->getPoseWithNormal()(1) + c*xyz_norms[id]->getPoseWithNormal()(2));
        int inliers = 0;
        for(int j = 0; j < xyz_norms.size(); ++j){
            float x = xyz_norms[j]->getPoseWithNormal()(0);
            float y = xyz_norms[j]->getPoseWithNormal()(1);
            float z = xyz_norms[j]->getPoseWithNormal()(2);
            float dist = abs(a*x + b*y + c*z + d) / sqrt(a*a + b*b + c*c);
            if(dist < err_thresh){
                inliers++;
            }
        }
        if(inliers > max_inliers){
            output = xyz_norms[id];
            max_inliers = inliers;
            float in_rate = (float)inliers / float(xyz_norms.size());
            if(in_rate > prob){
                break;
            }
        }
    }
    cout<<"RATE OUT: "<<((float)max_inliers / float(xyz_norms.size()))<<endl;
    return output;
}
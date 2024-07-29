#include <semantic_slam/api_class/semantic_slam.h>
SemanticSLAM::SemanticSLAM(): pnh_("~"){

    string voc_file;
    pnh_.param<string>("vocabulary_file", voc_file, "/home/nuninu98/catkin_ws/src/orb_semantic_slam/model/ORBvoc.txt");

    string setting_file;
    pnh_.param<string>("setting_file", setting_file, "/home/nuninu98/catkin_ws/src/orb_semantic_slam/setting/tum_rgbd.yaml");

    string crnn_file;
    pnh_.param<string>("crnn_file", crnn_file, "");
    
    string text_list;
    pnh_.param<string>("text_list", text_list, "");

    ocr_.reset(new OCR(crnn_file, text_list));

    string color_file;
    pnh_.param<string>("color_file", color_file, "");
    ifstream ifs(color_file.c_str());
    string line;
    while(getline(ifs, line)){
        char* pEnd;
        double r, g, b;
        r = strtod(line.c_str(), &pEnd);
        g = strtod(pEnd, &pEnd);
        b = strtod(pEnd, NULL);
        colors_.push_back(cv::Scalar(r, g, b, 255.0));
    }
    class_names_ = ld_.getClassNames();
    visual_odom_ = new ORB_SLAM3::System(voc_file, setting_file ,ORB_SLAM3::System::RGBD, false);

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
        double stamp = data.header.stamp.toSec();
        if(stamp > rgb_image->header.stamp.toSec()){
            break;
        }
        cv::Point3f accel(data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z);
        cv::Point3f gyro(data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z);
        ORB_SLAM3::IMU::Point p(accel, gyro, stamp);
        imu_points.push_back(p);
        imu_buf_.pop();
    }
    imu_lock_.unlock();
    //ros::Time tic = ros::Time::now();
    Eigen::Matrix4d cam_extrinsic = visual_odom_->TrackRGBD(cv_rgb_bridge->image, cv_depth_bridge->image, stamp.toSec(), imu_points).matrix().cast<double>();
    //cout<<"dt: "<<(ros::Time::now() - tic).toSec()<<endl;
}

void SemanticSLAM::detectionImageCallback(const sensor_msgs::ImageConstPtr& color_image){
    cv_bridge::CvImageConstPtr bridge = cv_bridge::toCvShare(color_image, "bgr8");
    cv::Mat image = bridge->image;
    vector<Detection> detections = ld_.detectObjectYOLO(image);
    vector<OCRDetection> text_detections = ocr_->detect_rec(image);
    
    for(const auto& m : detections){
        //==========Testing Room Number=========
        if(class_names_[m.getClassID()] == "room_number"){
            cv::Rect r1 = m.getRoI();
            double max_iou = 0.2;
            int max_iou_idx = -1;
            for(int i = 0; i < text_detections.size(); ++i){
                cv::Rect r2 = text_detections[i].getRoI();
                cv::Rect common = (r1 & r2);
                double iou = (double)common.area() / (double)(r1.area() + r2.area() - common.area());
                if(iou > max_iou){
                    max_iou = iou;
                    max_iou_idx = i;
                }
            }
            if(max_iou_idx == -1){
                continue;
            }
            cv::rectangle(image, m.getRoI(), colors_[m.getClassID() % 12], 2);
            cv::putText(image, to_string(text_detections[max_iou_idx].getClassID()), m.getRoI().tl(), 1, 1, colors_[m.getClassID() % 12]);
        }
        //======================================
        
    }
    cv::imshow("detection", image);
    cv::waitKey(1);
    //============TODO===============
    /*
    1. Door + Floor sign dataset (clear)
    2. Yolo training
    3. Door -> detect room number (clear)
    4. Door -> Wall plane projection (dropped)
    5. Floor, Room info to ORB SLAM
    6. Comparison 
    */
    //===============================
}

void SemanticSLAM::imuCallback(const sensor_msgs::ImuConstPtr& imu){
    imu_lock_.lock();
    imu_buf_.push(*imu);
    imu_lock_.unlock();
}

SemanticSLAM::~SemanticSLAM(){
    
}
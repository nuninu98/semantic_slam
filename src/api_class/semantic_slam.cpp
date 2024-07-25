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
    visual_odom_ = new ORB_SLAM3::System(voc_file, setting_file ,ORB_SLAM3::System::RGBD, true);

    sub_detection_image_ = nh_.subscribe("detection_camera", 1, &SemanticSLAM::detectionImageCallback, this);

    string rgb_topic, depth_topic;
    pnh_.param<string>("rgb_topic", rgb_topic, "/camera/color/image_raw");
    pnh_.param<string>("depth_topic", depth_topic, "/camera/aligned_depth_to_color/image_raw");
    rgb_subscriber_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, rgb_topic, 1));
    depth_subscriber_.reset(new message_filters::Subscriber<sensor_msgs::Image> (nh_, depth_topic, 1));
    
    sync_.reset(new message_filters::Synchronizer<sync_pol> (sync_pol(1000), *rgb_subscriber_, *depth_subscriber_));
    sync_->registerCallback(boost::bind(&SemanticSLAM::imageCallback, this, _1, _2));
}

void SemanticSLAM::imageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image){
    //ros::Time tic = ros::Time::now();
    ros::Time stamp = rgb_image->header.stamp;
    cv_bridge::CvImageConstPtr cv_rgb_bridge = cv_bridge::toCvShare(rgb_image, "bgr8");
    cv_bridge::CvImageConstPtr cv_depth_bridge = cv_bridge::toCvShare(depth_image, depth_image->encoding);
   
    Eigen::Matrix4d cam_extrinsic = visual_odom_->TrackRGBD(cv_rgb_bridge->image, cv_depth_bridge->image, stamp.toSec()).matrix().cast<double>();
    vector<Detection> detections;
    //detections = ld_.detectObjectMRCNN(cv_rgb_bridge->image);
    //detections = ld_.detectObjectYOLO(cv_rgb_bridge->image);
    cv::Mat detection_img = cv_rgb_bridge->image.clone();
    for(const auto& m : detections){
        cv::rectangle(detection_img, m.getRoI(), colors_[m.getClassID() % 12], 2);
        cv::putText(detection_img, class_names_[m.getClassID()], m.getRoI().tl(), 1, 1, colors_[m.getClassID() % 12]);
    }
    //cv::imshow("detection", detection_img);
    //cv::waitKey(1);
    // cout<<"DELAY: "<<(ros::Time::now() - tic).toSec()<<endl;
}

void SemanticSLAM::detectionImageCallback(const sensor_msgs::ImageConstPtr& color_image){
    cv_bridge::CvImageConstPtr bridge = cv_bridge::toCvShare(color_image, "bgr8");
    cv::Mat image = bridge->image;
    vector<Detection> detections = ld_.detectObjectYOLO(image);
    //============TODO===============
    /*
    1. Door + Floor sign dataset
    2. Yolo training
    3. Door -> detect room number (clear)
    4. Door -> Wall plane projection
    5. Floor, Room info to ORB SLAM
    6. Comparison
    */
    //===============================
}

SemanticSLAM::~SemanticSLAM(){
    
}
#include <semantic_slam/data_type/DataType.h>
    Detection::Detection(){

    }

    Detection::Detection(const cv::Rect& roi, const cv::Mat& mask, const string& name): roi_(roi), mask_(mask), name_(name), dg_(nullptr){

    }

    Detection::~Detection(){

    }

    cv::Rect Detection::getROI_CV() const{
        return roi_;
    }

    gtsam_quadrics::AlignedBox2 Detection::getROI() const{
        double xmin =roi_.x;
        double xmax = roi_.x + roi_.width;
        double ymin = roi_.y;
        double ymax = roi_.y + roi_.height;
        return gtsam_quadrics::AlignedBox2(xmin, ymin, xmax, ymax);
    }

    cv::Mat Detection::getMask() const{
        return mask_;
    }

    string Detection::getClassName() const{
        return name_;
    }

    // string Detection::getContent() const{
    //     return content_;
    // }

    void Detection::copyContent(const OCRDetection& ocr_output){
        name_ = name_ + ocr_output.getContent();
        //content_ = ocr_output.getContent();
    }

    void Detection::calcCentroid(const cv::Mat& color_mat, const cv::Mat& depth_mat, const Eigen::Matrix3f& K){
        // if(!cloud_.empty()){
        //     return;
        // }
        // vector<pcl::PointXYZRGB> raw_cloud;
        // for(int r = 0; r < color_mat.rows; r += 3){
        //     for(int c = 0; c < color_mat.cols; c += 3){
        //         float depth = depth_mat.at<float>(r, c);
        //         if(isnanf(depth) || depth < 1.0e-4){
        //             continue;
        //         }
        //         if(!roi_.contains(cv::Point2i(c, r))){
        //             continue;
        //         }
        //         Eigen::Vector3f pix(c, r, 1.0);
        //         float x = (c - K(0, 2)) * depth / K(0, 0);
        //         float y = (r - K(1, 2)) * depth / K(1, 1);
        //         pcl::PointXYZRGB pt;
        //         pt.x = x;
        //         pt.y = y;
        //         pt.z = depth;
        //         pt.r = color_mat.at<cv::Vec3b>(r, c)[2];
        //         pt.g = color_mat.at<cv::Vec3b>(r, c)[1];
        //         pt.b = color_mat.at<cv::Vec3b>(r, c)[0];
        //         raw_cloud.push_back(pt);
        //     }
        // }
        // sort(raw_cloud.begin(), raw_cloud.end(), [](const pcl::PointXYZRGB& pt1, const pcl::PointXYZRGB& pt2){
        //     return pt1.z < pt2.z;
        // });
        // int left = (float)raw_cloud.size() * 0.1;
        // int right = (float)raw_cloud.size() * 0.9;
        // for(int i = left; i <= min((int)raw_cloud.size()-1, right); ++i){
        //     cloud_.push_back(raw_cloud[i]);
        // }

        int cnt = 0;
        double center_depth = 0.0;
        cv::Point center_pix = roi_.tl() + cv::Point(roi_.width / 2, roi_.height / 2);
        for(int r = max(0, center_pix.y - 1); r < min(center_pix.y + 2, color_mat.rows); ++r){
            for(int c = max(0, center_pix.x - 1); c < min(center_pix.x + 2, color_mat.cols); ++c){
                float depth = depth_mat.at<float>(r, c);
                if(isnanf(depth) || depth < 1.0e-4){
                    continue;
                }
                center_depth += depth;
                cnt++;
            }
        }
        center_depth /= cnt;
        if(cnt == 0){
            centroid_(0) = -1.0;
            centroid_(1) = -1.0;
            centroid_(2) = -1.0;
            return; 
        }
        float x = (center_pix.x - K(0, 2)) * center_depth / K(0, 0);
        float y = (center_pix.y - K(1, 2)) * center_depth / K(1, 1);
        float z = center_depth;

        centroid_(0) = x;
        centroid_(1) = y;
        centroid_(2) = z;
    }

    // void Detection::getCloud(pcl::PointCloud<pcl::PointXYZRGB>& output) const{
    //     output.clear();
    //     output = cloud_;
    // }

    void Detection::setDetectionGroup(DetectionGroup* dg){
        dg_ = dg;
    }

    const DetectionGroup* Detection::getDetectionGroup() const{
        return dg_;
    }

    Eigen::Vector3f Detection::center3D() const{
        return centroid_;
    }

    void Detection::setCorrespondence(Object* obj){
        matched_obj_ = obj;
    }

    Object* Detection::getCorrespondence() const{
        return matched_obj_;
    }
    //=====================OCR DETECTION======================
    OCRDetection::OCRDetection(){
        
    }

    OCRDetection::OCRDetection(const OCRDetection& ocr){
        content_ = ocr.content_;
        roi_ = ocr.roi_;
    }

    OCRDetection::OCRDetection(const cv::Rect& roi, const string& content): roi_(roi){
        
        content_ = content;
    }


    string OCRDetection::getContent() const{
        return content_;
    }

    cv::Rect OCRDetection::getRoI() const{
        return roi_;
    }

    OCRDetection& OCRDetection::operator=(const OCRDetection& ocr){
        content_ = ocr.content_;
        roi_ = ocr.roi_;
        return *this;
    }
    //======================OBJECT==============================

    Object::Object() : id_(0){

    }

    Object::Object(const string& name, size_t id, const gtsam_quadrics::ConstrainedDualQuadric& Q): name_(name), id_(id), Q_(Q){

    }

    Object::Object(const Object& obj): name_(obj.name_), Q_(obj.Q_), id_(obj.id_){
    }

    string Object::getClassName() const{
        return name_;
    }

    void Object::addDetection(const Detection* det){
        seens_.push_back(det);
    }

    void Object::getConnectedKeyFrames(vector<KeyFrame*>& output) const{
        output.clear();
        for(const auto& s : seens_){
            output.push_back(s->getDetectionGroup()->getKeyFrame());
        }
    }

    // Eigen::Vector3f Object::getCentroid() const{
    //     // Eigen::Vector3f centroid;
    //     // pcl::PointCloud<pcl::PointXYZRGB> cloud;
    //     // getCloud(cloud);

    //     // pcl::PointXYZRGB cent;
    //     // pcl::computeCentroid(cloud, cent);

    //     // centroid(0) = cent.x;
    //     // centroid(1) = cent.y;
    //     // centroid(2) = cent.z;
    //     // return centroid;
    //     return centroid_;
    // }

    size_t Object::id() const{
        return id_;
    }

    void Object::merge(Object* obj){
        for(const auto& elem : obj->seens_){
            seens_.push_back(elem);
        }
    }

    gtsam_quadrics::ConstrainedDualQuadric Object::Q() const{
        return Q_;
    }

    void Object::setQ(const gtsam_quadrics::ConstrainedDualQuadric& Q){
        Q_ = Q;
    }

    //=====================DetectionGroup================
    DetectionGroup::DetectionGroup(){}

    DetectionGroup::DetectionGroup(const DetectionGroup& dg) : sensor_pose_(dg.sensor_pose_), detections_(dg.detections_), K_(dg.K_), stamp_(dg.stamp_), kf_(dg.kf_), sid_(dg.sid_), gray_(dg.gray_){
        for(auto& elem : detections_){
            elem->setDetectionGroup(this);
        }
    }

    DetectionGroup::DetectionGroup(const Eigen::Matrix4f& sensor_pose, const vector<Detection*>& detections, const Eigen::Matrix3f& K, double stamp, char sid): sensor_pose_(sensor_pose), stamp_(stamp), K_(K), detections_(detections), kf_(nullptr), sid_(sid)
    {
    // detection sibal
        for(auto& elem : detections_){
            //elem->generateCloud(color, depth, K);
            elem->setDetectionGroup(this);
        }
    }

    DetectionGroup::~DetectionGroup(){ //fix this!!!!

    }

    double DetectionGroup::stamp() const{
        return stamp_;
    }

    void DetectionGroup::detections(vector<Detection*>& output) const{
        output.clear();
        for(int i = 0; i < detections_.size(); ++i){
            output.push_back(detections_[i]);
        }
    }

    Eigen::Matrix4f DetectionGroup::getSensorPose() const{
        return sensor_pose_;
    }

    Eigen::Matrix3f DetectionGroup::getIntrinsic() const{
        return K_;
    }

    void DetectionGroup::setKeyFrame(KeyFrame* kf){
        kf_ = kf;    
    }

    KeyFrame* DetectionGroup::getKeyFrame() const{
        return kf_;
    }

    char DetectionGroup::sID() const{
        return sid_;
    }

    DetectionGroup& DetectionGroup::operator=(const DetectionGroup& dg){        
        detections_=dg.detections_;
        K_=dg.K_;
        stamp_ = dg.stamp_;
        kf_ =dg.kf_;
        sid_ = dg.sid_;
        gray_ = dg.gray_;
        //sensor_pose_ = dg.sensor_pose_;
        for(auto& elem : detections_){
            elem->setDetectionGroup(this);
        }
        return *this;
    }

    // cv::Mat DetectionGroup::getColorImage() const{
    //     return color_img;
    // }

    // cv::Mat DetectionGroup::getDepthImage() const{
    //     return depth_img;
    // }

    //=================Floor RANSAC================
    Floor::Floor(int label, KeyFrame* kf): label_(label), plane_kf_(kf){
        kfs_.push_back(kf);
    }   

    Floor::~Floor(){

    }

    void Floor::addKeyFrame(KeyFrame* kf){
        if(kf == nullptr){
            cout<<"INSERT NULL"<<endl;
        }
        kf->setFloor(this);
        kfs_.push_back(kf);
        if(kfs_.size() > 500){
            kfs_.pop_front();
        }
        refine();
        
    }

    void Floor::refine(){
        if(kfs_.empty()){
            return;
        }
        float err_thresh = 0.5;
        float inlier_r = 0.8;
        float prob = 0.95;

        int iter = 300;
        //int iter = log(1.0 - prob) / log(1.0 - inlier_r);
        int max_inliers = 0;

        for(int i = 0; i < iter; ++i){
            int id = rand() % kfs_.size();
            if(kfs_[id] == nullptr){
                cout<<"XXXX"<<endl;
            }
            Eigen::Matrix4f plane_se3 = kfs_[id]->getPose();
            // float a = kfs_[id]->getPoseWithNormal()(3); // ax+by+cz+d = 0
            // float b = kfs_[id]->getPoseWithNormal()(4);
            // float c = kfs_[id]->getPoseWithNormal()(5);
            // float d = -(a*kfs_[id]->getPoseWithNormal()(0) + b*kfs_[id]->getPoseWithNormal()(1) + c*kfs_[id]->getPoseWithNormal()(2));
            int inliers = 0;
            for(int j = 0; j < kfs_.size(); ++j){
                if(kfs_[j] == nullptr){
                    cout<<"YYYY"<<endl;
                }
                Eigen::Matrix4f kf_se3 = kfs_[j]->getPose();
                Eigen::Matrix4f delta = plane_se3.inverse() * kf_se3;
                float dist = abs(delta(1, 3));
                // float x = kfs_[j]->getPoseWithNormal()(0);
                // float y = kfs_[j]->getPoseWithNormal()(1);
                // float z = kfs_[j]->getPoseWithNormal()(2);
                // float dist = abs(a*x + b*y + c*z + d) / sqrt(a*a + b*b + c*c);
                if(dist < err_thresh){
                    inliers++;
                }
            }
            if(inliers > max_inliers){
                plane_kf_ = kfs_[id];
                max_inliers = inliers;
                float in_rate = (float)inliers / float(kfs_.size());
                if(in_rate > prob){
                    break;
                }
            }
        }
        //cout<<"RATE OUT: "<<((float)max_inliers / float(kfs_.size()))<<endl;

    }

    bool Floor::isInlier(KeyFrame* kf){
        
        // Eigen::VectorXf plane = plane_kf_->getPoseWithNormal();
        // float a = plane(3); // ax+by+cz+d = 0
        // float b = plane(4);
        // float c = plane(5);
        // float d = -(a*plane(0) + b*plane(1) + c*plane(2));

        // Eigen::VectorXf point = kf->getPoseWithNormal();
        // float x = point(0);
        // float y = point(1);
        // float z = point(2);
        Eigen::Matrix4f plane_se3 = plane_kf_->getPose();
        Eigen::Matrix4f kf_se3 = kf->getPose();
        Eigen::Matrix4f delta = plane_se3.inverse() * kf_se3;
        float dist = abs(delta(1, 3));//abs(a*x + b*y + c*z + d) / sqrt(a*a + b*b + c*c);
        bool res = dist < 1.5;
        if(!res){
            cout<<"DIST ERR: "<<dist<<" kf: "<<kf->id()<<endl;
        }
        return res;
    }


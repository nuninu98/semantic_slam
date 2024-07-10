#include <semantic_slam/algorithm/landmark_detector.h>
#include <opencv2/dnn/dnn.hpp>
LandmarkDetector::LandmarkDetector(): pnh_("~"){
    string text_graph = "/home/nuninu98/backup/Mask-R-CNN/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    string model_weights = "/home/nuninu98/backup/Mask-R-CNN/mask-rcnn-coco/frozen_inference_graph.pb";
    string class_file = "/home/nuninu98/backup/Mask-R-CNN/mask-rcnn-coco/object_detection_classes_coco.txt";
    pnh_.param<string>("text_graph", text_graph, "");
    pnh_.param<string>("model_weights", model_weights, "");
    pnh_.param<string>("class_file", class_file, "");
    ifstream ifs(class_file.c_str());
    string line;
    while(getline(ifs, line)){
        class_names_.push_back(line);
    }
    network_ = cv::dnn::readNetFromTensorflow(model_weights, text_graph);
    //network_ = cv::dnn::readNetFromDarknet("/home/nuninu98/catkin_ws/src/orb_semantic_slam/model/yolov3.cfg", "/home/nuninu98/catkin_ws/src/orb_semantic_slam/model/yolov3.weights");
    network_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    network_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    // vocabulary_.reset(new DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>());
    // vocabulary_->loadFromTextFile(voc_file_path);
    //last_layer_names_ = network_.getUnconnectedOutLayersNames();    
    last_layer_names_ = {"detection_out_final", "detection_masks"};

}

LandmarkDetector::~LandmarkDetector(){
}

void LandmarkDetector::detectObjectYOLO(const cv::Mat& rgb_image){
    cv::Mat blob = cv::dnn::blobFromImage(rgb_image, 1.0 /255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    network_.setInput(blob);
    //========================Simple detection=======================
    vector<cv::Mat> outputs;

    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    network_.forward(outputs, last_layer_names_);
    for(size_t i = 0; i < outputs.size(); i++){
        float* data = (float*)outputs[i].data;
        for(size_t r = 0; r < outputs[i].rows; r++, data += outputs[i].cols){
            cv::Mat scores = outputs[i].row(r).colRange(5, outputs[i].cols);
            cv::Point class_id_pt;
            double conf = 0.0;
            cv::minMaxLoc(scores, 0, &conf, 0, &class_id_pt);
            if(conf > CONFIDENCE_THRESHOLD){
                int center_x = data[0] * rgb_image.cols;
                int center_y = data[1] * rgb_image.rows;
                int width = data[2] * rgb_image.cols;
                int height = data[3] * rgb_image.rows;
                int left = center_x - width / 2;
                int top = center_y - height / 2;
                if(width * height < 50){
                    continue;
                }
                class_ids.push_back(class_id_pt.x);
                confidences.push_back(float(conf));
                cv::Rect roi = cv::Rect(left, top, width, height);
                boxes.push_back(roi);
               
            }
        }
    }
    // cv::Mat detection_image = rgb_image.clone();
    // vector<int> indices;
    // cv::Rect image_size(0, 0, rgb_image.cols, rgb_image.rows);
    // cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4, indices);
    // for(const auto& id : indices){
    //     cv::Rect roi = boxes[id] & image_size;
    //     cv::rectangle(detection_image, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width, roi.y + roi.height), cv::Scalar(0, 255, 0), 2);
    //     cv::putText(detection_image, COCO_NAMES[class_ids[id]]+" : "+to_string(confidences[id]), cv::Point(roi.x, roi.y), 1, 2, cv::Scalar(255, 255, 0));
    //     SemanticMeasurement obj(roi);
    //     obj.score = confidences[id];
    //     //obj.id = class_ids[id];
    //     obj.class_name = COCO_NAMES[class_ids[id]];
    //     cv::Mat crop_img = rgb_image(roi).clone();
    //     //(*semantic_vocabulary_)[obj.class_name].transform(obj.descriptor, obj.bow_vector);

    //     objects.push_back(obj);
    // }
}


vector<Detection> LandmarkDetector::detectObjectMRCNN(const cv::Mat& rgb_image){
    cv::Mat blob;
    cv::dnn::blobFromImage(rgb_image, blob, 1.0, cv::Size(rgb_image.cols, rgb_image.rows), cv::Scalar(), true, false);
    network_.setInput(blob);
    vector<cv::Mat> outputs;
    network_.forward(outputs, last_layer_names_);
    vector<Detection> objects;
    cv::Mat out_detections = outputs[0];
    cv::Mat out_masks = outputs[1];
    int num_detections = out_detections.size[2];
    int num_classes = out_masks.size[1];

    out_detections = out_detections.reshape(1, out_detections.total() / 7);
    cv::Rect image_size(0, 0, rgb_image.cols, rgb_image.rows);

    for(int i = 0; i < num_detections; ++i){
        float score = out_detections.at<float>(i, 2);
        if(score > CONFIDENCE_THRESHOLD){

            int class_id = static_cast<int>(out_detections.at<float>(i, 1));
            int left = static_cast<int>(rgb_image.cols * out_detections.at<float>(i, 3));
            int top = static_cast<int>(rgb_image.rows * out_detections.at<float>(i, 4));
            int right = static_cast<int>(rgb_image.cols * out_detections.at<float>(i, 5));
            int bottom = static_cast<int>(rgb_image.rows * out_detections.at<float>(i, 6));

            left = max(0, min(left, rgb_image.cols - 1));
            top = max(0, min(top, rgb_image.rows - 1));
            right = max(0, min(right, rgb_image.cols - 1));
            bottom = max(0, min(bottom, rgb_image.rows - 1));
            cv::Rect box(left, top, right - left + 1, bottom - top + 1);
            cv::Rect roi = box & image_size;

            cv::Mat mask_resize;
            cv::Mat object_mask(out_masks.size[2], out_masks.size[3], CV_32F, out_masks.ptr<float>(i, class_id));
            cv::resize(object_mask, mask_resize, cv::Size(roi.width, roi.height)); 
            cv::Mat mask = mask_resize > 0.6;
            mask.convertTo(mask, CV_8U);
            
            Detection det(roi, mask, class_id);

            objects.push_back(det);
        }
    }
    return objects;
}


// vector<SemanticMeasurement> LandmarkDetector::detectObject(const cv::Mat& rgb_image){
//     vector<SemanticMeasurement> objects;
//     //======================Using rcnn===================================================
//     cv::Mat blob;
//     cv::dnn::blobFromImage(rgb_image, blob, 1.0, cv::Size(rgb_image.cols, rgb_image.rows), cv::Scalar(), true, false);
//     network_.setInput(blob);
//     vector<cv::Mat> outputs;
//     network_.forward(outputs, last_layer_names_);
//     cv::Mat out_detections = outputs[0];
//     cv::Mat out_masks = outputs[1];
//     int num_detections = out_detections.size[2];
//     int num_classes = out_masks.size[1];
//     out_detections = out_detections.reshape(1, out_detections.total() / 7);
//     cv::Rect image_size(0, 0, rgb_image.cols, rgb_image.rows);
//     //#pragma omp parallel for
//     for(int i = 0; i < num_detections; ++i){
//         float score = out_detections.at<float>(i, 2);
//         if(score > CONFIDENCE_THRESHOLD){

//             int class_id = static_cast<int>(out_detections.at<float>(i, 1));
//             int left = static_cast<int>(rgb_image.cols * out_detections.at<float>(i, 3));
//             int top = static_cast<int>(rgb_image.rows * out_detections.at<float>(i, 4));
//             int right = static_cast<int>(rgb_image.cols * out_detections.at<float>(i, 5));
//             int bottom = static_cast<int>(rgb_image.rows * out_detections.at<float>(i, 6));

//             left = max(0, min(left, rgb_image.cols - 1));
//             top = max(0, min(top, rgb_image.rows - 1));
//             right = max(0, min(right, rgb_image.cols - 1));
//             bottom = max(0, min(bottom, rgb_image.rows - 1));
//             cv::Rect box(left, top, right - left + 1, bottom - top + 1);
//             cv::Rect roi = box & image_size;
//             SemanticMeasurement obj(roi);

//             cv::Mat mask_resize;
//             cv::Mat object_mask(out_masks.size[2], out_masks.size[3], CV_32F, out_masks.ptr<float>(i, class_id));
//             cv::resize(object_mask, mask_resize, cv::Size(obj.box.width(), obj.box.height())); 
//             cv::Mat mask = mask_resize > 0.6;
//             mask.convertTo(mask, CV_8U);
            
//             //==============Color Histogram===========
//             cv::Mat img_crop = rgb_image(roi);
//             // cv::Mat img_hsv;
//             // cv::cvtColor(img_crop, img_hsv, CV_BGR2HSV);
//             // cv::calcHist(&img_hsv, 1, channel_, mask, obj.histogram, 2, hist_size, channel_ranges);
//             // cv::calcHist(&img_crop, 1, channel_B, mask, obj.histogram[0], 1, hist_size, channel_ranges);
//             // cv::calcHist(&img_crop, 1, channel_G, mask, obj.histogram[1], 1, hist_size, channel_ranges);
//             // cv::calcHist(&img_crop, 1, channel_R, mask, obj.histogram[2], 1, hist_size, channel_ranges);
//             //========================================
//             obj.class_id = class_id;
//             obj.score = score;
//             obj.mask = mask.clone();
//             //================Test==========
//             obj.image = img_crop.clone();
//             // cv::Mat crop_gray;
//             // cv::cvtColor(img_crop, crop_gray, CV_BGR2GRAY);
//             // cv::bitwise_and(obj.mask, crop_gray, obj.image);
//             //==============================
//             //#pragma omp critical
//             objects.push_back(obj);
//         }
//     }
  
    
    
//     return objects;
// }

vector<string> LandmarkDetector::getClassNames() const{
    return class_names_;
}

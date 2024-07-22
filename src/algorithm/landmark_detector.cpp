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
    
    //===========MASK RCNN=============
    // network_ = cv::dnn::readNetFromTensorflow(model_weights, text_graph);
    // last_layer_names_ = {"detection_out_final", "detection_masks"};
    //============YOLOv8===============
    network_ = cv::dnn::readNetFromONNX("/home/nuninu98/Downloads/yolov8n.onnx");
    last_layer_names_ = network_.getUnconnectedOutLayersNames();

    //=================================

    network_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    network_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // vocabulary_.reset(new DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>());
    // vocabulary_->loadFromTextFile(voc_file_path);
    

}

LandmarkDetector::~LandmarkDetector(){
}

cv::Mat LandmarkDetector::formatToSquare(const cv::Mat& source){
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

vector<Detection> LandmarkDetector::detectObjectYOLO(const cv::Mat& rgb_image){
    vector<Detection> objects;
    cv::Mat model_input = rgb_image;

    model_input = formatToSquare(model_input);
    cv::Size model_shape = cv::Size(640, 640); // model size
    cv::Mat blob = cv::dnn::blobFromImage(model_input, 1.0 /255.0, model_shape, cv::Scalar(), true, false);
    
    // cv::Size model_shape = rgb_image.size();
    // cv::Mat blob = cv::dnn::blobFromImage(model_input, 1.0 /255.0, model_shape, cv::Scalar(), true, false);
    network_.setInput(blob);
    //========================Simple detection=======================
    vector<cv::Mat> outputs;
    ros::Time tic = ros::Time::now();
    network_.forward(outputs, last_layer_names_);
    cout<<"TIME: "<<(ros::Time::now() - tic).toSec()<<endl;
    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];
    bool yolov8 = false;
    if(dimensions > rows){
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];
        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }

    float* data = (float*)outputs[0].data;
    float x_factor = model_input.cols / model_shape.width;
    float y_factor = model_input.rows / model_shape.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for(int i = 0; i < rows; ++i){
        if(yolov8){
            float* classes_scores = data + 4;
            int classes_num = 80;
            cv::Mat scores(1, classes_num, CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if(max_class_score > CONFIDENCE_THRESHOLD){
                confidences.push_back(max_class_score);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else{

        }
        data += dimensions;
    }
    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_TRESHOLD, nms_result);
    for(int i = 0; i < nms_result.size(); ++i){
        int idx = nms_result[i];
        Detection detection(boxes[idx], cv::Mat(), class_ids[idx]);
        objects.push_back(detection);
    }
  
    return objects;
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

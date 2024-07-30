#include <semantic_slam/algorithm/landmark_detector.h>
#include <opencv2/dnn/dnn.hpp>
LandmarkDetector::LandmarkDetector(): pnh_("~"){
    class_names_ = {"floor_sign", "room_number"};
    
    //============YOLOv8===============
    string yolo_model;
    pnh_.param<string>("yolo_model", yolo_model, "");
    network_ = cv::dnn::readNetFromONNX(yolo_model);
    last_layer_names_ = network_.getUnconnectedOutLayersNames();
    //=================================

    network_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    network_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

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
            int classes_num = 2;
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
        Detection detection(boxes[idx], cv::Mat(), class_names_[class_ids[idx]]);
        objects.push_back(detection);
    }
  
    return objects;
}




vector<string> LandmarkDetector::getClassNames() const{
    return class_names_;
}

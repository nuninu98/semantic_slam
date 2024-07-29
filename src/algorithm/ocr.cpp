#include <semantic_slam/algorithm/ocr.h>

OCR::OCR(string modelRecognition, string alphabet)
{
	this->confThreshold = 0.5;
	this->nmsThreshold = 0.4;
	this->inpHeight = 320;
	this->inpWidth = 320;
    this->alphabet = alphabet;
    string model = "/home/nuninu98/Downloads/frozen_east_text_detection.pb";
	this->detector = cv::dnn::readNet(model);
	this->modelRecognition = modelRecognition;
	detector.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    detector.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    if (!modelRecognition.empty())
	{
        ifstream vocFile;
        vocFile.open(alphabet);
        string vocLine;
        while(getline(vocFile, vocLine)){
            vocabulary.push_back(vocLine);
        }
        recognizer.reset(new cv::dnn::TextRecognitionModel(modelRecognition));
        recognizer->setVocabulary(vocabulary);
        recognizer->setDecodeType("CTC-greedy");
        recognizer->setInputParams(1.0/127.5, cv::Size(100, 32), cv::Scalar(127.5));
        recognizer->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        recognizer->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	}	
}

void OCR::decodeBoundingBoxes(const cv::Mat& scores, const cv::Mat& geometry, vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
{
    detections.clear();
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < this->confThreshold)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}

void OCR::fourPointsTransform(const cv::Mat& frame, cv::Point2f vertices[4], cv::Mat& result)
{
    const cv::Size outputSize = cv::Size(100, 32);

    cv::Point2f targetVertices[4] = { cv::Point(0, outputSize.height - 1),
                                  cv::Point(0, 0), cv::Point(outputSize.width - 1, 0),
                                  cv::Point(outputSize.width - 1, outputSize.height - 1),
                                };
    cv::Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);

    warpPerspective(frame, result, rotationMatrix, outputSize);
}

void OCR::decodeText(const cv::Mat& scores, string& text)
{
    cv::Mat scoresMat = scores.reshape(1, scores.size[0]);

    vector<char> elements;
    elements.reserve(scores.size[0]);

    for (int rowIndex = 0; rowIndex < scoresMat.rows; ++rowIndex)
    {
        cv::Point p;
        minMaxLoc(scoresMat.row(rowIndex), 0, 0, 0, &p);
        if (p.x > 0 && static_cast<size_t>(p.x) <= this->alphabet.size())
        {
            elements.push_back(this->alphabet[p.x - 1]);
        }
        else
        {
            elements.push_back('-');
        }
    }

    if (elements.size() > 0 && elements[0] != '-')
        text += elements[0];

    for (size_t elementIndex = 1; elementIndex < elements.size(); ++elementIndex)
    {
        if (elementIndex > 0 && elements[elementIndex] != '-' &&
            elements[elementIndex - 1] != elements[elementIndex])
        {
            text += elements[elementIndex];
        }
    }
}

void OCR::expandRectangle(const cv::Rect& input, double rate, cv::Rect& output){
    double w = input.width;
    double h = input.height;
    cv::Point center = input.tl() + cv::Point(w/2, h/2);
    w *= (1.0 + rate);
    h *= (1.0 + rate);
    cv::Point topleft = center - cv::Point(w/2 , h/2);
    output = cv::Rect(topleft.x, topleft.y, w, h);
}


vector<OCRDetection> OCR::detect_rec(cv::Mat& frame)
{
    vector<OCRDetection> output;

    vector<cv::Mat> outs;
    vector<string> outNames(2);
    outNames[0] = "feature_fusion/Conv_7/Sigmoid";
    outNames[1] = "feature_fusion/concat_3";
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(123.68, 116.78, 103.94), true, false);
    this->detector.setInput(blob);
    //this->detector.forward(outs, this->detector.getUnconnectedOutLayersNames());   ////ÔËÐÐ»á³ö´í
    this->detector.forward(outs, outNames);

    cv::Mat scores = outs[0];
    cv::Mat geometry = outs[1];
    // Decode predicted bounding boxes.
    vector<cv::RotatedRect> boxes;
    vector<float> confidences;
    this->decodeBoundingBoxes(scores, geometry, boxes, confidences);

    // Apply non-maximum suppression procedure.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

    cv::Point2f ratio((float)frame.cols / this->inpWidth, (float)frame.rows / this->inpHeight);
    // Render text.
    cv::Rect image_size = cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Mat frame_copy = frame.clone();
    #pragma omp parallel for
    for (size_t i = 0; i < indices.size(); ++i)
    {
        cv::RotatedRect& box = boxes[indices[i]];

        cv::Point2f vertices[4];
        box.points(vertices);

        for (int j = 0; j < 4; ++j)
        {
            vertices[j].x *= ratio.x;
            vertices[j].y *= ratio.y;
        }

        if (!this->modelRecognition.empty())
        {   
            bool valid = true;
            vector<cv::Point2f> vertices_arr(4);
            for (int j = 0; j < 4; ++j)
            {
                if(vertices[j].x < 0 || vertices[j].y < 0 || vertices[j].x >= frame.cols || vertices[j].y >= frame.rows){
                    valid = false;
                    break;
                }
                vertices_arr[j] = vertices[j];
            }
            if(!valid){
                continue;
            }
            cv::Mat cropped;
            cv::Rect roi = minAreaRect(vertices_arr).boundingRect();
            cv::Rect roi_expand;
            expandRectangle(roi, 0.4, roi_expand);
            //this->fourPointsTransform(frame, vertices, cropped);
            #pragma omp critical
            {
                cropped = frame((roi_expand & image_size));
                cv::Mat crop_hsv;
                cv::cvtColor(cropped, crop_hsv, cv::COLOR_BGR2HSV);
                vector<cv::Mat> hsv_hist;
                cv::split(crop_hsv, hsv_hist);
                cv::equalizeHist(hsv_hist[2], hsv_hist[2]);
                cv::Mat hsv_eq;
                cv::merge(hsv_hist, hsv_eq);
                cv::Mat crop_v_eq;
                cv::cvtColor(hsv_eq, crop_v_eq, cv::COLOR_HSV2BGR);
                
                string wordRecognized = recognizer->recognize(crop_v_eq);
                if(all_of(wordRecognized.begin(), wordRecognized.end(), ::isdigit) && !wordRecognized.empty()){
                    if(wordRecognized.size() == 4){
                        OCRDetection text(roi, stoi(wordRecognized));
                        output.push_back(text);
                    }
                    // cv::putText(frame_copy, wordRecognized, vertices[1], cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255));
                    // cv::rectangle(frame_copy, roi_expand, cv::Scalar(0, 0, 255), 2);
                    // cv::imshow("Detected", crop_v_eq);
                    // cv::waitKey(1);
                }
                
            }
            
        }

        // for (int j = 0; j < 4; ++j)
        //     line(frame_copy, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
    }
    

    // string kWinName = "Deep learning object detection in OpenCV";
    // cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
    // cv::imshow(kWinName, frame_copy);
    // cv::waitKey(1);

    return output;
}
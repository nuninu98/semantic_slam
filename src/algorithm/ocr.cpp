#include <semantic_slam/algorithm/ocr.h>

OCR::OCR(string modelRecognition, string alphabet)
{
	this->confThreshold = 0.5;
	this->nmsThreshold = 0.4;
	this->inpHeight = 320;
	this->inpWidth = 320;
    this->alphabet = alphabet;
    String model = "/home/nuninu98/Downloads/frozen_east_text_detection.pb";
	this->detector = readNet(model);
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
        recognizer.reset(new TextRecognitionModel(modelRecognition));
        recognizer->setVocabulary(vocabulary);
        recognizer->setDecodeType("CTC-greedy");
        recognizer->setInputParams(1.0/127.5, Size(100, 32), Scalar(127.5));
        recognizer->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        recognizer->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		// this->recognizer = readNet(modelRecognition);
        // recognizer.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        // recognizer.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}	
}

void OCR::decodeBoundingBoxes(const Mat& scores, const Mat& geometry, std::vector<RotatedRect>& detections, std::vector<float>& confidences)
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

            Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
            Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
            RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}

void OCR::fourPointsTransform(const Mat& frame, Point2f vertices[4], Mat& result)
{
    const Size outputSize = Size(100, 32);

    Point2f targetVertices[4] = { Point(0, outputSize.height - 1),
                                  Point(0, 0), Point(outputSize.width - 1, 0),
                                  Point(outputSize.width - 1, outputSize.height - 1),
                                };
    Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);

    warpPerspective(frame, result, rotationMatrix, outputSize);
}

void OCR::decodeText(const Mat& scores, std::string& text)
{
    Mat scoresMat = scores.reshape(1, scores.size[0]);

    std::vector<char> elements;
    elements.reserve(scores.size[0]);

    for (int rowIndex = 0; rowIndex < scoresMat.rows; ++rowIndex)
    {
        Point p;
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

void OCR::expandRectangle(const Rect& input, double rate, Rect& output){
    double w = input.width;
    double h = input.height;
    Point center = input.tl() + Point(w/2, h/2);
    w *= (1.0 + rate);
    h *= (1.0 + rate);
    Point topleft = center - Point(w/2 , h/2);
    output = Rect(topleft.x, topleft.y, w, h);
}


void OCR::detect_rec(Mat& frame)
{
    Mat frame_copy = frame.clone();
    std::vector<Mat> outs;
    std::vector<String> outNames(2);
    outNames[0] = "feature_fusion/Conv_7/Sigmoid";
    outNames[1] = "feature_fusion/concat_3";
    Mat blob;
    blobFromImage(frame, blob, 1.0, Size(this->inpWidth, this->inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
    this->detector.setInput(blob);
    //this->detector.forward(outs, this->detector.getUnconnectedOutLayersNames());   ////ÔËÐÐ»á³ö´í
    this->detector.forward(outs, outNames);

    Mat scores = outs[0];
    Mat geometry = outs[1];
    // Decode predicted bounding boxes.
    std::vector<RotatedRect> boxes;
    std::vector<float> confidences;
    this->decodeBoundingBoxes(scores, geometry, boxes, confidences);

    // Apply non-maximum suppression procedure.
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

    Point2f ratio((float)frame.cols / this->inpWidth, (float)frame.rows / this->inpHeight);
    // Render text.
    Rect image_size = Rect(0, 0, frame.cols, frame.rows);
    #pragma omp parallel for
    for (size_t i = 0; i < indices.size(); ++i)
    {
        RotatedRect& box = boxes[indices[i]];

        Point2f vertices[4];
        box.points(vertices);

        for (int j = 0; j < 4; ++j)
        {
            vertices[j].x *= ratio.x;
            vertices[j].y *= ratio.y;
        }

        if (!this->modelRecognition.empty())
        {   
            bool valid = true;
            vector<Point2f> vertices_arr(4);
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
            Mat cropped;
            Rect roi = minAreaRect(vertices_arr).boundingRect();
            Rect roi_expand;
            expandRectangle(roi, 0.4, roi_expand);
            //this->fourPointsTransform(frame, vertices, cropped);
            #pragma omp critical
            {
                cropped = frame((roi_expand & image_size));
                //=============Test Heq============
                Mat crop_hsv;
                cvtColor(cropped, crop_hsv, COLOR_BGR2HSV);
                vector<Mat> hsv_hist;
                split(crop_hsv, hsv_hist);
                equalizeHist(hsv_hist[2], hsv_hist[2]);
                Mat hsv_eq;
                merge(hsv_hist, hsv_eq);
                Mat crop_v_eq;
                cvtColor(hsv_eq, crop_v_eq, COLOR_HSV2BGR);
                
                //=================================
                string wordRecognized = recognizer->recognize(crop_v_eq);
                if(all_of(wordRecognized.begin(), wordRecognized.end(), ::isdigit) && !wordRecognized.empty()){
                    putText(frame_copy, wordRecognized, vertices[1], FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255));
                    rectangle(frame_copy, roi_expand, Scalar(0, 0, 255), 2);
                    imshow("Equalized", crop_v_eq);
                    waitKey(1);
                }
                
            }
            
            // cvtColor(cropped, cropped, cv::COLOR_BGR2GRAY);

            // Mat blobCrop = blobFromImage(cropped, 1.0 / 127.5, Size(), Scalar::all(127.5));
            // this->recognizer.setInput(blobCrop);

            // Mat result = this->recognizer.forward();

            // std::string wordRecognized = "";
            // this->decodeText(result, wordRecognized);
            // putText(frame, wordRecognized, vertices[1], FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255));
        }

        // for (int j = 0; j < 4; ++j)
        //     line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 1);
    }

    string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    imshow(kWinName, frame_copy);
    waitKey(1);

}
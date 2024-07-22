#ifndef __SEMANTIC_SLAM_OCR_H__
#define __SEMANTIC_SLAM_OCR_H__
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace dnn;
using namespace std;

class OCR
{
	public:
		float confThreshold;
		float nmsThreshold;
		int inpWidth;
		int inpHeight;
		string modelRecognition;
		Net detector;
		Net recognizer;
        string alphabet;
        OCR(string modelRecognition, string alphabet);
		void decodeBoundingBoxes(const Mat& scores, const Mat& geometry, std::vector<RotatedRect>& detections, std::vector<float>& confidences);
        void fourPointsTransform(const Mat& frame, Point2f vertices[4], Mat& result);
        void decodeText(const Mat& scores, std::string& text);
        void detect_rec(Mat& frame);
};

#endif
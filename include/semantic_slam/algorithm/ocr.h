#ifndef __SEMANTIC_SLAM_OCR_H__
#define __SEMANTIC_SLAM_OCR_H__
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
using namespace cv;
using namespace dnn;
using namespace std;

class OCR
{
	private:
		void expandRectangle(const Rect& input, double rate, Rect& output);
	public:
		float confThreshold;
		float nmsThreshold;
		int inpWidth;
		int inpHeight;
		string modelRecognition;
		Net detector;
		string alphabet;

		//=========Testing============
		shared_ptr<TextRecognitionModel> recognizer;
		vector<string> vocabulary;
		//=============================
		//Net recognizer;
        OCR(string modelRecognition, string alphabet);
		void decodeBoundingBoxes(const Mat& scores, const Mat& geometry, std::vector<RotatedRect>& detections, std::vector<float>& confidences);
        void fourPointsTransform(const Mat& frame, Point2f vertices[4], Mat& result);
        void decodeText(const Mat& scores, std::string& text);
        void detect_rec(Mat& frame);
};

#endif
#ifndef __SEMANTIC_SLAM_OCR_H__
#define __SEMANTIC_SLAM_OCR_H__
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <omp.h>
#include <opencv2/ximgproc.hpp>
#include <semantic_slam/data_types/data_types.h>
using namespace std;

class OCR
{
	private:
		float confThreshold;
		float nmsThreshold;
		int inpWidth;
		int inpHeight;
		string modelRecognition;
		cv::dnn::Net detector;
		string alphabet;
		//=========Testing============
		shared_ptr<cv::dnn::TextRecognitionModel> recognizer;
		vector<string> vocabulary;
		//=============================
		void decodeBoundingBoxes(const cv::Mat& scores, const cv::Mat& geometry, vector<cv::RotatedRect>& detections, vector<float>& confidences);
        void fourPointsTransform(const cv::Mat& frame, cv::Point2f vertices[4], cv::Mat& result);
        void decodeText(const cv::Mat& scores, string& text);
		void expandRectangle(const cv::Rect& input, double rate, cv::Rect& output);
	public:
		

		
		//Net recognizer;
        OCR(string modelRecognition, string alphabet);
		
        void detect_rec(cv::Mat& frame);
};

#endif
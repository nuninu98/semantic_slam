#ifndef __SEMANTIC_SLAM_DATA_TYPES_HEADER__
#define __SEMANTIC_SLAM_DATA_TYPES_HEADER__
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

class Detection{
    protected:
        cv::Rect roi_;
        cv::Mat mask_;
        size_t id_;
    public: 
        Detection();

        Detection(const cv::Rect& roi, const cv::Mat& mask, const size_t& id);

        ~Detection();   

        cv::Rect getRoI() const;

        cv::Mat getMask() const;

        size_t getClassID() const;
};


class OCRDetection: public Detection{
    public:
        OCRDetection();

        OCRDetection(const cv::Rect& roi, const size_t& id);
};

#endif
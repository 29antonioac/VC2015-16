#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using std::string;

string SplitFilename (const string& str);
Mat gaussianMask(float sigma);
Mat convolution1D(Mat &input, Mat &mask, bool reflected);


#endif

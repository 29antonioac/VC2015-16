#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using cv::Mat;
using std::string;

string SplitFilename (const string& str);
Mat gaussianMask(float sigma);


#endif

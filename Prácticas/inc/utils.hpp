#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using std::string;
using std::vector;

string SplitFilename (const string& str);

Mat homograpyEquationMatrix(vector<Point> origin, vector<Point> destination);
Mat homography(vector<Point> origin, vector<Point> destination);

#endif

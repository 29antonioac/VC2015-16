#ifndef HOMOGRAPHY_HPP
#define HOMOGRAPHY_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using std::vector;
using namespace cv;

class Homography
{
private:
  Mat homography;

  Mat homographyEquationMatrix(vector<Point> origin, vector<Point> destination);
  Mat calcHomography(vector<Point> origin, vector<Point> destination);
public:
  Homography(vector<Point> origin, vector<Point> destination);
  Mat getHomography();

};


#endif

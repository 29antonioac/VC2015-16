#ifndef __IMAGEN_HPP__
#define __IMAGEN_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include "utils.hpp"
#include "homography.hpp"

using namespace cv;
using std::string;
using std::vector;

class Image
{
private:
  Mat image;
  string name;
  static int num_images;
  int ID;
  static const int DISPLAY_WIDTH, DISPLAY_HEIGHT;

  Mat gaussianMask(float sigma);
  Mat convolution1D1C(Mat &input, Mat &mask, bool reflected);
  Mat convolution1D(Mat &input, Mat &mask, bool reflected);
  Mat convolution2D(Mat &input, Mat &mask, bool reflected);

  vector<Mat> MaskFirstDerivative(float sigma, char axis);
  vector<Mat> MaskSecondDerivative(float sigma, char axis);
public:
  Image(string path, bool flag);
  Image(int rows, int cols, string name = "Mat");
  Image(const Mat& input, string name = "Mat");
  Image(const Image& img);
  Image(const vector<Image*> & sequence, unsigned int rows, unsigned int cols);
  Image(vector<Image*> & images, string name);

  void paint();
  void getPixel(int x, int y);

  void setPixels(const vector<Point> & pixel_list, const vector<Vec3b> & value_list);
  void setPixels(const vector<Point> & pixel_list, const vector<uchar> & value_list);

  Image GaussConvolution(const float sigma, bool reflected = false);
  Image highFrecuencies(const float sigma, bool reflected = false);
  Image createHybrid(const Image &another, bool reflected, float sigma_1, float sigma_2);
  Image convolution(const Mat &mask);
  Image downsample();

  /* Bonus 1 */

  Image calcFirstDerivative(float sigma, char axis, bool reflected = false);
  Image calcSecondDerivative(float sigma, char axis, bool reflected = false);
  Image detectEdges(double threshold1, double threshold2);

  /* Work 2 */

  Image warpPerspective(Homography hom);
  void drawCircle(Point p, int radius, Scalar color, int thickness = 1);

  // Image makePanorama(vector<Image*> & images, string name);



};

#endif

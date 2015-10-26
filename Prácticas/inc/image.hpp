#ifndef __IMAGEN_HPP__
#define __IMAGEN_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include "utils.hpp"

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

  Mat gaussianMask(float sigma);
  Mat convolution1D1C(Mat &input, Mat &mask, bool reflected);
  Mat convolution1D(Mat &input, Mat &mask, bool reflected);
  Mat convolution2D(Mat &input, Mat &mask, bool reflected);
  Mat flip(Mat &input);
public:
  Image(string path, bool flag);
  Image(int rows, int cols);
  Image(const Mat& input);
  Image(const Image& img);
  Image(const vector<Image*> & sequence, unsigned int rows, unsigned int cols);

  void paint();
  void getPixel(int x, int y);

  void setPixels(const vector<Point> & pixel_list, const vector<Vec3b> & value_list);
  void setPixels(const vector<Point> & pixel_list, const vector<uchar> & value_list);

  Image GaussConvolution(const float sigma, bool reflected = false);
  Image createHybrid(const Image &another, bool reflected, float sigma_1, float sigma_2);



};

#endif

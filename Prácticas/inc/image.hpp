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
public:
  Image(string path, bool flag);
  Image(int rows, int cols);
  Image(const Image& img);
  Image(const vector<Image*> & sequence, unsigned int rows, unsigned int cols);

  void paint();
  void getPixel(int x, int y);

  void setPixels(const vector<Point> & pixel_list, const vector<Vec3b> & value_list);
  void setPixels(const vector<Point> & pixel_list, const vector<uchar> & value_list);

  void convolution(const float sigma);



};

#endif

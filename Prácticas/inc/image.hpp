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
  Image(string path, int flag);
  Image(int rows, int cols);
  Image(const Image& img);

  void paint();
  void getPixel(int x, int y);
  void setPixels();

  Image createCanvas(vector<Image> sequence, int rows, int cols);

};

#endif

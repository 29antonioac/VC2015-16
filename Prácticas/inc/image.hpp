#ifndef __IMAGEN_HPP__
#define __IMAGEN_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include "utils.hpp"

using namespace cv;
using std::string;

class Image
{
private:
  Mat image;
  string name;
public:
  Image(string path, int flag);
  Image(const Image& img);

  void paint();

};

#endif

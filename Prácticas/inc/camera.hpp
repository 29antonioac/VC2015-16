#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <ostream>

using namespace cv;

class Camera
{
private:
  Mat camera;


public:
  Camera(float low = 0.0, float high = 1.0);

  bool isFinite();

  Point2f project(Point3f input);

  void print();






};

#endif

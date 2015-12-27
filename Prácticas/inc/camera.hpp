#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace cv;

class Camera
{
private:
  Mat camera;


public:
  Camera();

  bool isFinite();
  void makeRandomFinite(float low, float high);

  Point2f project(Point3f input);




};

#endif

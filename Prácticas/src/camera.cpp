#include "../inc/camera.hpp"

Camera::Camera()
{
  // Fill 3x4 matrix with zeros
  this->camera = Mat::zeros(3,4,CV_32FC1);
}

bool Camera::isFinite()
{
  // Check if the first 3x3 submatrix is regular
  Mat M = this->camera(Rect(0,0,3,3));
  return determinant(M) != 0.0;
}

void Camera::makeRandomFinite(float low, float high)
{
  while (!this->isFinite())
  {
    // Set this->camera to randon matrix. Random values are uniform from low to high
    randu(this->camera,low,high);
  }
}

Point2f Camera::project(Point3f input)
{
  // Take input with homogeneous coordinates
  Vec4f homogeneous (input.x, input.y, input.z, 1.0);

  // Product with the camera
  Mat result = this->camera * Mat(homogeneous);

  // Quotient two first coordinates with last one
  Point2f projection (result.at<float>(0) / result.at<float>(2), result.at<float>(1) / result.at<float>(2));

  return projection;



}

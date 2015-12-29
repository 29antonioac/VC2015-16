#include "../inc/camera.hpp"
#include <cmath>
#include <iostream>

Camera::Camera(float low, float high)
{
  // Fill 3x4 matrix with zeros
  this->camera = Mat::zeros(3, 4, CV_32FC1);

  while (!this->isFinite())
  {
    // Set this->camera to randon matrix. Random values are uniformly distributed from low to high
    randu(this->camera,low,high);
  }
}

Camera::Camera(vector< pair<Point3f, Point2f> > correspondences)
{
  this->camera = Mat::zeros(3, 4, CV_32FC1);
  Mat A = Mat::zeros(2 * correspondences.size(), 12, CV_32FC1);

  for (unsigned i = 0; i < correspondences.size(); i++)
  {
    A.at<float>(2 * i, 4) = -correspondences[i].first.x;
    A.at<float>(2 * i, 5) = -correspondences[i].first.y;
    A.at<float>(2 * i, 6) = -correspondences[i].first.z;
    A.at<float>(2 * i, 7) = 1.0;
    A.at<float>(2 * i, 8) = correspondences[i].second.y * correspondences[i].first.x;
    A.at<float>(2 * i, 9) = correspondences[i].second.y * correspondences[i].first.y;
    A.at<float>(2 * i,10) = correspondences[i].second.y * correspondences[i].first.z;
    A.at<float>(2 * i,11) = correspondences[i].second.y;

    A.at<float>(2 * i + 1, 0) = correspondences[i].first.x;
    A.at<float>(2 * i + 1, 1) = correspondences[i].first.y;
    A.at<float>(2 * i + 1, 2) = correspondences[i].first.z;
    A.at<float>(2 * i + 1, 3) = 1.0;
    A.at<float>(2 * i + 1, 8) = -correspondences[i].second.x * correspondences[i].first.x;
    A.at<float>(2 * i + 1, 9) = -correspondences[i].second.x * correspondences[i].first.y;
    A.at<float>(2 * i + 1,10) = -correspondences[i].second.x * correspondences[i].first.z;
    A.at<float>(2 * i + 1,11) = -correspondences[i].second.x;
  }

  Mat w, u ,vt;
  SVD::compute(A, w, u, vt);

  for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			this->camera.at<float>(i, j) = vt.at<float>(11, i * 4 + j);
}


bool Camera::isFinite()
{
  // Check if the first 3x3 submatrix is regular
  Mat M = this->camera(Rect(0,0,3,3));
  return determinant(M) != 0.0;
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

float Camera::error(Camera other)
{
  Mat one = this->camera / this->camera.at<float>(2,2);
  Mat two = other.camera / other.camera.at<float>(2,2);

  Mat diff = one - two;
  float error = 0.0;

  for (int i = 0; i < diff.rows; i++)
    for (int j = 0; j < diff.cols; j++)
      error += diff.at<float>(i,j) * diff.at<float>(i,j);

  error = sqrt(error);

  return error;
}

void Camera::print()
{
  std::cout << this->camera << std::endl;
}

#include "../inc/homography.hpp"

Mat Homography::homographyEquationMatrix(vector<Point> origin, vector<Point> destination)
{
  // Check input size
  if (origin.size() != destination.size())
  {
    return Mat::zeros(3,3,CV_32FC1);
  }

  int num_points = origin.size();
  int num_equations = 2 * num_points;

  // Fill new matrix with zeros
  Mat coeffs = Mat::zeros(num_equations, 9, CV_32FC1);

  Point ori, dest;

  // Set coeffs of the equation matrix as slides
  for (int i = 0; i < num_points; i++)
  {
    ori = origin[i];
    dest = destination[i];

    int actual_row = 2 * i;
    int next_row = actual_row + 1;

    coeffs.at<float>(Point(0,actual_row)) = ori.x;
    coeffs.at<float>(Point(1,actual_row)) = ori.y;
    coeffs.at<float>(Point(2,actual_row)) = 1;
    coeffs.at<float>(Point(6,actual_row)) = -dest.x * ori.x;
    coeffs.at<float>(Point(7,actual_row)) = -dest.x * ori.y;
    coeffs.at<float>(Point(8,actual_row)) = -dest.x;

    coeffs.at<float>(Point(3,next_row)) = ori.x;
    coeffs.at<float>(Point(4,next_row)) = ori.y;
    coeffs.at<float>(Point(5,next_row)) = 1;
    coeffs.at<float>(Point(6,next_row)) = -dest.y * ori.x;
    coeffs.at<float>(Point(7,next_row)) = -dest.y * ori.y;
    coeffs.at<float>(Point(8,next_row)) = -dest.y;
  }

  return coeffs;
}

Mat Homography::calcHomography(vector<Point> origin, vector<Point> destination)
{
  // Compute matrix with equation coefficients
  Mat A = homographyEquationMatrix(origin,destination);

  // Compute SVD of A (http://docs.opencv.org/master/df/df7/classcv_1_1SVD.html#a76f0b2044df458160292045a3d3714c6)
  Mat w, u, vt;
	SVD::compute(A, w, u, vt);

  // Take last col of vt matrix and reorder as 3x3 matrix
  Mat homography_matrix = Mat(3, 3, CV_32FC1);
	for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
		  homography_matrix.at<float>(Point(j, i)) = vt.at<float>(Point(3*i + j, vt.cols -1 ));
    }
	}

  return homography_matrix;
}

Homography::Homography(vector<Point> origin, vector<Point> destination)
{
  this->homography = this->calcHomography(origin,destination);
}

Mat Homography::getHomography()
{
  return homography;
}

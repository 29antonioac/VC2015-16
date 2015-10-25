#include "../inc/utils.hpp"

#include <cstddef>        // std::size_t
#include <cmath>
#include <iostream>

using namespace std;

string SplitFilename (const string& str)
{
  std::size_t found = str.find_last_of("/\\");
  return str.substr(found+1);
}

Mat gaussianMask(float sigma)
{
  unsigned int mask_size = 2 * round(3 * sigma) + 1; // +-3*sigma plus the zero
  unsigned int mask_center = round(3 * sigma);

  float value = 0, sum_values = 0;
  unsigned int mask_index;

  Mat mask = Mat(1, mask_size, CV_32FC1);

  for (unsigned int i = 0; i < mask_size; i++)
  {
    mask_index = i - mask_center;
    value = exp(-0.5 * (mask_index * mask_index) / (sigma * sigma));

    mask.at<float>(mask_index,0) = value;
    sum_values += value;
  }

  mask *= 1.0 / sum_values;

  return mask;
}

Mat convolution1D(Mat &input, Mat &mask, bool reflected)
{
  // Expand the matrix
  Mat expanded;
  int borderType = BORDER_CONSTANT;
  int offset = (mask.cols - 1) / 2;

  if (reflected)
    borderType = BORDER_REFLECT;

  copyMakeBorder(input,expanded,0,0,offset,offset,borderType,0);

  // Convolution!
  Mat ROI;
  Mat output = Mat::zeros(1, input.cols, CV_32FC1);

  for (int i = 0; i < input.cols; i++) // Index are OK
  {
    ROI = Mat(expanded, Rect(i,0,mask.cols,1));
    output.at<float>(Point(i,0)) = ROI.dot(mask);
  }

  return output;

}

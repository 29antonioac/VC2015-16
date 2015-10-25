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
  int mask_size = 2 * round(3 * sigma) + 1; // +-3*sigma plus the zero
  int mask_center = round(3 * sigma);

  float value = 0, sum_values = 0;
  int mask_index;

  Mat mask = Mat::zeros(1, mask_size, CV_32FC1);

  for (int i = 0; i < mask_size; i++)
  {
    mask_index = i - mask_center;
    value = exp(-0.5 * (mask_index * mask_index) / (sigma * sigma));

    mask.at<float>(Point(i,0)) = value;
    sum_values += value;
  }

  mask *= 1.0 / sum_values;

  return mask;
}

Mat convolution1D1C(Mat &input, Mat &mask, bool reflected)
{
  // Expand the matrix
  Mat expanded, copy_input;
  int borderType = BORDER_CONSTANT;
  int offset = (mask.cols - 1) / 2;
  bool is_column = false;

  if (reflected)
    borderType = BORDER_REFLECT;

  if (input.rows > 1 && input.cols == 1)
    is_column = true;

  if (is_column)
    transpose(input,copy_input);
  else
    copy_input = input.clone();

  copyMakeBorder(copy_input,expanded,0,0,offset,offset,borderType,0);

  // Convolution!
  Mat ROI;
  Mat output = Mat::zeros(1, copy_input.cols, CV_32FC1);
  expanded.convertTo(expanded,CV_32FC1);

  for (int i = 0; i < copy_input.cols; i++) // Index are OK
  {
    ROI = Mat(expanded, Rect(i,0,mask.cols,1));
    output.at<float>(Point(i,0)) = ROI.dot(mask);
  }

  Mat copy_output;
  if (is_column)
    transpose(output,copy_output);
  else
    output.copyTo(copy_output);

  return copy_output;
}

Mat convolution1D(Mat &input, Mat &mask, bool reflected)
{
  Mat output;
  if (input.channels() == 1)
    output = convolution1D1C(input,mask,reflected);
  else
  {
    cout << "3 canales" << endl;
    Mat input_channels[3], output_channels[3];
    split(input, input_channels);

    for (int i = 0; i < input.channels(); i++)
    {
      output_channels[i] = convolution1D1C(input_channels[i],mask,reflected);
    }
    merge(output_channels, input.channels(), output);
  }

  return output;
}

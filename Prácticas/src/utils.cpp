#include "../inc/utils.hpp"

#include <cstddef>        // std::size_t
#include <cmath>

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

#include "../inc/image.hpp"

#include <opencv2/imgproc.hpp>

#include <iostream>
using std::cout;
using std::endl;

int Image::num_images = 0;

/* Private methods */

Mat Image::gaussianMask(float sigma)
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

Mat Image::convolution1D1C(Mat &input, Mat &mask, bool reflected)
{
  // Expand the matrix
  Mat expanded, copy_input;
  int borderType = BORDER_CONSTANT;
  int offset = (mask.cols - 1) / 2;

  if (reflected)
    borderType = BORDER_REFLECT;


  copyMakeBorder(input,expanded,0,0,offset,offset,borderType,0);

  // Convolution!
  Mat ROI;
  Mat output = Mat::zeros(1, input.cols, CV_32FC1);
  expanded.convertTo(expanded,CV_32FC1);

  for (int i = 0; i < input.cols; i++) // Index are OK
  {
    ROI = Mat(expanded, Rect(i,0,mask.cols,1));
    output.at<float>(Point(i,0)) = ROI.dot(mask);
  }

  return output;
}

Mat Image::convolution1D(Mat &input, Mat &mask, bool reflected)
{
  Mat output;
  if (input.channels() == 1)
    output = convolution1D1C(input,mask,reflected);
  else
  {
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

Mat Image::convolution2D(Mat &input, Mat &mask, bool reflected)
{

  mask = flip(mask);
  Mat output = Mat(input);

  // Convolution
  for (int i = 0; i < image.rows; i++)
  {
    Mat row = output.row(i).clone();
    convolution1D(row,mask,reflected).copyTo(output.row(i));
  }
  // Transposing for convolution by cols
  output = output.t();

  for (int i = 0; i < input.rows; i++)
  {
    Mat row = output.row(i).clone();
    convolution1D(row,mask,reflected).copyTo(output.row(i));
  }
  // Re-transposing to make the correct image
  output = output.t();

  return output;
}

Mat Image::flip(Mat &input) {
	Mat flipped = Mat::zeros(input.rows, input.cols, input.type());

	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			flipped.at<float>(Point(j,i)) = input.at<float>(Point(input.cols - 1 - j, input.rows - i - 1));

	return flipped;
}
/* ---------------------------------------------------- */

Image::Image(string path, bool flag)
{
  num_images++;
  ID = num_images;
  int cv_flag = (flag) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
  image = imread(path,cv_flag);
  name = SplitFilename(path);
}

Image::Image(int rows, int cols)
{
  num_images++;
  ID = num_images;
  image = Mat::zeros(rows,cols,CV_8UC3);
  name = "zeros";
}

Image::Image(const Mat& input)
{
  num_images++;
  ID = num_images;
  image = input.clone();
  name = "Mat";
}

Image::Image(const Image& img)
{
  num_images++;
  ID = num_images;
  image = img.image.clone();
  name = img.name;
}

Image::Image(const vector<Image*> & sequence, unsigned int rows, unsigned int cols)
{
  int max_rows = 0;
  int sum_rows = 0;
  int max_cols = 0;
  int sum_cols = 0;

  int result_rows = 0;
  int result_cols = 0;

  if (sequence.size() == rows * cols)
  {
    // Convert grayscale images to color
    for (Image* pimg: sequence)
    {
      if (pimg->image.channels() == 1)
      {
        Mat out;
        cvtColor(pimg->image,out,CV_GRAY2RGB);
        pimg->image = out;
      }
    }

    // Compute the resulting canvas
    max_rows = 0;
    sum_rows = 0;
    max_cols = 0;
    sum_cols = 0;


    vector<int> row_offsets; // Max number of rows from each image row

    for (int row_image = 0; row_image < rows; row_image++)
    {
      sum_cols = 0;
      for (int col_image = 0; col_image < cols; col_image++)
      {
        max_rows = std::max(sequence.at(row_image*cols+col_image)->image.rows,max_rows);
        sum_cols += sequence.at(row_image*cols+col_image)->image.cols;
      }
      max_cols = std::max(max_cols,sum_cols);
      sum_rows += max_rows;
      row_offsets.push_back(sum_rows);
    }


    result_rows = sum_rows;
    result_cols = max_cols;

    // Fill the canvas
    Mat canvas = Mat::zeros(result_rows,result_cols,CV_8UC3);
    int canvas_initial_col = 0;
    int canvas_initial_row = 0;

    int row_image, col_image;
    int image_index, image_index_prev_row;

    for (row_image = 0; row_image < rows; row_image++)
    {
      canvas_initial_col = 0;

      if (row_image > 0)
      {
        canvas_initial_row = row_offsets.at(row_image-1);
      }
      else
      {
        canvas_initial_row = 0;
      }

      for (col_image = 0; col_image < cols; col_image++)
      {
        image_index = row_image*cols+col_image;
        image_index_prev_row = image_index - cols;
        Mat actual_image = sequence.at(image_index)->image;

        for (int actual_row = 0; actual_row < actual_image.rows; actual_row++)
        {
          for (int actual_col = 0; actual_col < actual_image.cols; actual_col++)
          {
            canvas.at<Vec3b>(actual_row + canvas_initial_row,actual_col + canvas_initial_col) = actual_image.at<Vec3b>(actual_row,actual_col);
          }
        }
        canvas_initial_col += actual_image.cols;
      }
    }
    image = canvas;
  }
  else
  {
    image = Mat::zeros(rows,cols,CV_8UC3);
  }

  // Construct the object
  num_images++;
  ID = num_images;
  name = "Canvas";

}

void Image::paint()
{
  string window_name = std::to_string(ID) + "-" + name;
  namedWindow(window_name,WINDOW_AUTOSIZE);
  imshow(window_name,image);
}

void Image::setPixels(const vector<Point> & pixel_list, const vector<Vec3b> & value_list)
{
  for (int i = 0; i < pixel_list.size(); i++)
  {
    image.at<Vec3b>(pixel_list.at(i)) = value_list.at(i % value_list.size());
  }
}

void Image::setPixels(const vector<Point> & pixel_list, const vector<uchar> & value_list)
{
  for (int i = 0; i < pixel_list.size(); i++)
  {
    image.at<uchar>(pixel_list.at(i)) = value_list.at(i % value_list.size());
  }
}

Image Image::GaussConvolution(const float sigma, bool reflected)
{
  Mat mask = gaussianMask(sigma);
  Mat convolution = convolution2D(this->image, mask, reflected);

  return Image(convolution);
}

Image Image::createHybrid(const Image &another, bool reflected, float sigma_1, float sigma_2)
{
  Mat low, high;
  Mat output;

  Mat mask_1 = gaussianMask(sigma_1);
  Mat mask_2 = gaussianMask(sigma_2);

  Mat copy_image = another.image.clone();

  low = convolution2D(this->image, mask_1, reflected);
  high = convolution2D(copy_image, mask_2, reflected);

  high = another.image - high;

	output = low + high;

  return Image(output);
}

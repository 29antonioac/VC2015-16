#include "../inc/image.hpp"

#include <opencv2/imgproc.hpp>

// #include <iostream>
// using std::cout;
// using std::endl;

int Image::num_images = 0;


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

    vector<int> row_offsets;

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

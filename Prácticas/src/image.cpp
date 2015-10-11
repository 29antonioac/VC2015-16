#include "../inc/image.hpp"

int Image::num_images = 0;


Image::Image(string path, int flag)
{
  num_images++;
  ID = num_images;
  image = imread(path,flag);
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

void Image::paint()
{
  string window_name = std::to_string(ID) + "-" + name;
  namedWindow(window_name,WINDOW_AUTOSIZE);
  imshow(window_name,image);
}

Image Image::createCanvas(vector<Image> sequence, int rows, int cols)
{
  return Image(rows,cols);
}

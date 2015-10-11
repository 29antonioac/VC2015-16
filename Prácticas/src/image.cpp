#include "../inc/image.hpp"


Image::Image(string path, int flag)
{
  image = imread(path,flag);
  name = SplitFilename(path);
}

Image::Image(const Image& img)
{
  image = img.image.clone();
  name = img.name;
}

void Image::paint()
{
  namedWindow(name,WINDOW_AUTOSIZE);
  imshow(name,image);
}

#include "../headers/image.hpp"

Image::Image(string path, int flag)
{
  image = readImage(path,flag);
  name = SplitFilename(path);
}

Image::Image(const Image::Image& img)
{
  image = img.clone();
  name = img.name;
}

void Image::paint()
{
  namedWindow(name,WINDOW_AUTOSIZE);
  imshow(name,image);
}

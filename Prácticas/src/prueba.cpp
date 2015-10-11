#include "../inc/image.hpp"
#include <iostream>



using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
  Image prueba(string("../Imágenes/lena.jpg"),CV_LOAD_IMAGE_COLOR);
  prueba.paint();


  waitKey(0);

  return 0;
}

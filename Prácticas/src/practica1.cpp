#include "../inc/image.hpp"
#include <iostream>


int main(int argc, char const *argv[]) {
  Image prueba(string("Imágenes/lena.jpg"),CV_LOAD_IMAGE_COLOR);
  Image prueba2(string("Imágenes/lena.jpg"),CV_LOAD_IMAGE_COLOR);
  prueba.paint();
  prueba2.paint();


  waitKey(0);

  return 0;
}

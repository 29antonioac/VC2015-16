#include "../inc/image.hpp"
#include <iostream>


int main(int argc, char const *argv[]) {
  Image prueba(string("Imágenes/lena.jpg"),CV_LOAD_IMAGE_COLOR);
  Image prueba2(string("Imágenes/bird.bmp"),CV_LOAD_IMAGE_COLOR);
  prueba.paint();
  prueba2.paint();

  vector<Image> secuencia;
  secuencia.push_back(prueba);
  secuencia.push_back(prueba2);
  secuencia.push_back(prueba);
  secuencia.push_back(prueba2);
  secuencia.push_back(prueba);
  secuencia.push_back(prueba2);
  secuencia.push_back(prueba2);
  secuencia.push_back(prueba2);



  Image canvas(secuencia,2,4);
  canvas.paint();


  waitKey(0);

  return 0;
}

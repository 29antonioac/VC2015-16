#include "../inc/image.hpp"
#include <iostream>


int main(int argc, char const *argv[]) {
  Image prueba(string("Imágenes/lena.jpg"),false);
  Image prueba2(string("Imágenes/bird.bmp"),true);
  prueba.paint();
  prueba2.paint();

  waitKey(0);
  destroyAllWindows();


  vector<Image*> secuencia;
  secuencia.push_back(&prueba);
  secuencia.push_back(&prueba2);
  secuencia.push_back(&prueba);


  Image canvas(secuencia,1,3);
  canvas.paint();

  waitKey(0);
  destroyAllWindows();

  Image prueba3(string("Imágenes/lena.jpg"),true);
  vector<Point> pixels;
  for (int i = 0; i < 256; i+=4)
  {
    for (int j = 0; j < 256; j++)
      pixels.push_back(Point(j,i));
  }

  vector<Vec3b> values;
  values.push_back(0);
  prueba.setPixels(pixels,values);
  prueba.paint();


  waitKey(0);
  destroyAllWindows();

  return 0;
}

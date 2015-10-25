#include "../inc/image.hpp"
#include "../inc/utils.hpp"
#include <iostream>

using namespace std;
using namespace cv;


int main(int argc, char const *argv[]) {
  Image prueba(string("Imágenes/lena.jpg"),false);
  Image prueba2(string("Imágenes/einstein.bmp"),true);
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

  float data[5] = {1,2,3,2,1};
  float data2[3] = {1,2,1};
  Mat A (1, 5, CV_32FC1, &data);
  Mat B (1, 3, CV_32FC1, &data2);
  cout << A << endl;

  cout << convolution1D(A,B,false);

  return 0;
}

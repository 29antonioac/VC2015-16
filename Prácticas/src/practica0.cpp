#include "../inc/image.hpp"
#include "../inc/utils.hpp"
#include <iostream>

using namespace std;
using namespace cv;


int main(int argc, char const *argv[]) {


  Image lena(string("Imágenes/lena.jpg"),false);
  Image lena_conv = lena.GaussConvolution(3);
  lena.paint();
  lena_conv.paint();

  waitKey(0);
  destroyAllWindows();

  Image einstein(string("Imágenes/einstein.bmp"),true);
  Image marilyn(string("Imágenes/marilyn.bmp"),true);

  Image hybrid = einstein.createHybrid(marilyn,true,6,6);

  vector<Image*> secuencia;
  secuencia.push_back(&einstein);
  secuencia.push_back(&hybrid);
  secuencia.push_back(&marilyn);

  Image tira(secuencia,1,3);
  tira.paint();

  waitKey(0);
  destroyAllWindows();

  vector<Image*> pyramid;
  Image hybrid_d1 = hybrid.downsample();
  Image hybrid_d2 = hybrid_d1.downsample();
  Image hybrid_d3 = hybrid_d2.downsample();
  Image hybrid_d4 = hybrid_d3.downsample();
  pyramid.push_back(&hybrid);
  pyramid.push_back(&hybrid_d1);
  pyramid.push_back(&hybrid_d2);
  pyramid.push_back(&hybrid_d3);
  pyramid.push_back(&hybrid_d4);

  Image pyramidImage(pyramid, 1,5);

  pyramidImage.paint();



  waitKey(0);
  destroyAllWindows();

  Image derivative_x = einstein.calcFirstDerivative(3,'x');
  Image derivative_y = einstein.calcFirstDerivative(3,'y');
  derivative_x.paint();
  derivative_y.paint();

  waitKey(0),
  destroyAllWindows();

  return 0;
}

#include "../inc/image.hpp"
#include "../inc/utils.hpp"
#include <iostream>

using namespace std;
using namespace cv;


int main(int argc, char const *argv[]) {
  Image input(string("imagenes/Tablero1.jpg"),true);
  Image dest(string("imagenes/Tablero2.jpg"),true);

	vector<Point> origin, destination;

	origin.push_back(Point(156,  49));
	origin.push_back(Point(530,  13));
	origin.push_back(Point(215,  90));
	origin.push_back(Point(474, 103));
	origin.push_back(Point(362, 134));
	origin.push_back(Point(305, 330));
	origin.push_back(Point(184, 348));
	origin.push_back(Point(438, 397));
	origin.push_back(Point(137, 422));
	origin.push_back(Point(527, 464));

	destination.push_back(Point(148,  14));
	destination.push_back(Point(502,  96));
	destination.push_back(Point(212,  80));
	destination.push_back(Point(446, 157));
	destination.push_back(Point(347, 160));
	destination.push_back(Point(264, 322));
	destination.push_back(Point(138, 323));
	destination.push_back(Point(374, 386));
	destination.push_back(Point(76 , 387));
	destination.push_back(Point(431, 443));

  Homography hom(origin, destination);
	Image output_good = input.warpPerspective(hom);

  vector<Point> another_origin, another_destination;

	another_origin.push_back(Point(156,  47));
	another_origin.push_back(Point(177,  46));
	another_origin.push_back(Point(155,  72));
	another_origin.push_back(Point(175,  70));
	another_origin.push_back(Point(198,  43));
	another_origin.push_back(Point(197,  67));
	another_origin.push_back(Point(219,  66));
	another_origin.push_back(Point(153,  95));
	another_origin.push_back(Point(152, 120));
	another_origin.push_back(Point(173, 119));

	another_destination.push_back(Point(150, 14));
	another_destination.push_back(Point(174, 19));
	another_destination.push_back(Point(142, 40));
	another_destination.push_back(Point(167, 46));
	another_destination.push_back(Point(198, 24));
	another_destination.push_back(Point(191, 51));
	another_destination.push_back(Point(217, 56));
	another_destination.push_back(Point(137, 63));
	another_destination.push_back(Point(131, 90));
	another_destination.push_back(Point(157, 95));

  Homography hom2(another_origin, another_destination);
	Image output_bad = input.warpPerspective(hom2);

  int circle_radius = 10;
  int circle_thickness = 3;

  for (int i = 0; i < origin.size(); i++)
  {
    input.drawCircle(origin[i], circle_radius, Scalar(255,0,0), circle_thickness);
    dest.drawCircle(destination[i], circle_radius, Scalar(255,0,0), circle_thickness);
  }

  for (int i = 0; i < another_origin.size(); i++)
  {
    input.drawCircle(another_origin[i], circle_radius, Scalar(0,0,255), circle_thickness);
    dest.drawCircle(another_destination[i], circle_radius, Scalar(0,0,255), circle_thickness);
  }

  vector<Image*> sequence;
  sequence.push_back(&input);
  sequence.push_back(&dest);
  sequence.push_back(&output_good);
  sequence.push_back(&output_bad);

  Image canvas(sequence, 2, 2);
  canvas.paint();

  waitKey(0);
  destroyAllWindows();

  return 0;
}

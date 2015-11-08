#include "../inc/image.hpp"
#include "../inc/utils.hpp"
#include <iostream>

using namespace std;
using namespace cv;


int main(int argc, char const *argv[]) {
  Mat input = imread("imagenes/Tablero1.jpg", 1);
	Mat dest = imread("imagenes/Tablero2.jpg", 1);
	Mat output_good, output_bad;

  Mat input_copy = input.clone();

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

	Mat hom = homography(origin, destination);
	warpPerspective(input_copy,output_good,hom,Size(output_good.cols,output_good.rows));

  int circle_radius = 10;
  int circle_thickness = 3;

  for (int i = 0; i < origin.size(); i++)
  {
    circle(input,origin[i],circle_radius,Scalar(0,0,255),circle_thickness);
    circle(dest,destination[i],circle_radius,Scalar(0,0,255),circle_thickness);
  }

	origin.clear();
	destination.clear();

	origin.push_back(Point(156,  47));
	origin.push_back(Point(177,  46));
	origin.push_back(Point(155,  72));
	origin.push_back(Point(175,  70));
	origin.push_back(Point(198,  43));
	origin.push_back(Point(197,  67));
	origin.push_back(Point(219,  66));
	origin.push_back(Point(153,  95));
	origin.push_back(Point(152, 120));
	origin.push_back(Point(173, 119));

	destination.push_back(Point(150, 14));
	destination.push_back(Point(174, 19));
	destination.push_back(Point(142, 40));
	destination.push_back(Point(167, 46));
	destination.push_back(Point(198, 24));
	destination.push_back(Point(191, 51));
	destination.push_back(Point(217, 56));
	destination.push_back(Point(137, 63));
	destination.push_back(Point(131, 90));
	destination.push_back(Point(157, 95));

  hom = homography(origin, destination);
	warpPerspective(input_copy,output_bad,hom,Size(output_bad.cols,output_bad.rows));

  for (int i = 0; i < origin.size(); i++)
  {
    circle(input,origin[i],circle_radius,Scalar(255,0,0),circle_thickness);
    circle(dest,destination[i],circle_radius,Scalar(255,0,0),circle_thickness);
  }

  imshow("Original", input);
	imshow("Output_good", output_good);
  imshow("Output_bad",output_bad);
  imshow("Destination", dest);

  waitKey(0);
  destroyAllWindows();

  return 0;
}

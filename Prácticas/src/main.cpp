#include "../inc/image.hpp"
#include "../inc/camera.hpp"
#include "../inc/utils.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <ctime>

using namespace std;

int main(int argc, char const *argv[]) {
  theRNG().state = clock();
  vector<Point3f> worldPoints;

  for (int x1 = 1; x1 <= 10; x1++)
    for (int x2 = 1; x2 <= 10; x2++)
    {
      worldPoints.push_back(Point3f(0.0, x1/10.0, x2/10.0));
      worldPoints.push_back(Point3f(x2/10.0, x1/10.0, 0.0));
    }

  Camera randomFinite;
  vector < pair <Point3f, Point2f> > correspondences;

  for (unsigned int i = 0; i < worldPoints.size(); i++)
  {
    Point3f actualPoint = worldPoints[i];
    correspondences.push_back(pair<Point3f, Point2f>(actualPoint,randomFinite.project(actualPoint)));
  }

  Camera estimated(correspondences);

  randomFinite.print();
  cout << endl;
  estimated.print();

  cout << "Error = " << randomFinite.error(estimated) << endl;


  return 0;
}

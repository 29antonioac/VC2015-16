#include "../inc/image.hpp"
#include "../inc/camera.hpp"
#include "../inc/utils.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <ctime>

int main(int argc, char const *argv[]) {
  theRNG().state = clock();
  vector<Point3f> worldPoints;

  for (float x1 = 0.1; x1 <= 1.0; x1 += 0.1)
    for (float x2 = 0.1; x2 <= 1.0; x2 += 0.1)
    {
      worldPoints.push_back(Point3f(0.0, x1, x2));
      worldPoints.push_back(Point3f(x2, x1, 0.0));
    }

  Camera randomFinite;
  randomFinite.print();


  return 0;
}

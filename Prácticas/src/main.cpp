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
  vector<Point2f> projectedPoints;
  vector < pair <Point3f, Point2f> > correspondences;

  for (int x1 = 1; x1 <= 10; x1++)
    for (int x2 = 1; x2 <= 10; x2++)
    {
      worldPoints.push_back(Point3f(0.0, x1/10.0, x2/10.0));
      worldPoints.push_back(Point3f(x2/10.0, x1/10.0, 0.0));
    }

  Camera randomFinite;

  for (unsigned int i = 0; i < worldPoints.size(); i++)
  {
    Point3f actualPoint = worldPoints[i];
    Point2f projectedPoint = randomFinite.project(actualPoint);
    projectedPoints.push_back(projectedPoint);
    correspondences.push_back(pair<Point3f, Point2f>(actualPoint,projectedPoint));
  }

  Camera estimated(correspondences);

  cout << "Error = " << randomFinite.error(estimated) << endl;

  vector<Point2f> newProjectedPoints;

  for (unsigned int i = 0; i < worldPoints.size(); i++)
  {
    newProjectedPoints.push_back(estimated.project(worldPoints[i]));
  }

  Image canvas(600,800,"Points");

  for (int i = 0; i < projectedPoints.size(); i++)
  {
    canvas.drawCircle(projectedPoints[i], 5, Scalar(255,0,0), 3);
    canvas.drawCircle(newProjectedPoints[i], 5, Scalar(0,0,255), 3);
  }

  canvas.paint();

  waitKey(0);
  destroyAllWindows();

  return 0;
}

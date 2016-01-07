#include "../inc/image.hpp"
#include "../inc/camera.hpp"
#include "../inc/utils.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <ctime>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
  theRNG().state = clock();
  vector<Point3f> worldPoints;
  vector<Point2f> projectedPoints;
  vector < pair<Point3f, Point2f> > correspondences;

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

  for (unsigned i = 0; i < projectedPoints.size(); i++)
  {
    canvas.drawCircle(projectedPoints[i] * 100, 5, Scalar(255,0,0), 3);
    canvas.drawCircle(newProjectedPoints[i] * 100, 5, Scalar(0,0,255), 3);
  }

  canvas.paint();

  waitKey(0);
  destroyAllWindows();

  /* Exercise 2 */


  const int CHESS_IMAGES = 25;
  bool valid;
  vector<Point2f> corners;
  vector< vector<Point2f> > imagePoints;
  vector<Mat> images(CHESS_IMAGES);
  vector<Mat> valid_images;
  Size patternSize(12,13);

  for (int i = 0; i < CHESS_IMAGES; i++)
  {
    images[i] = imread("imagenes/Image" + to_string(i+1) + ".tif", CV_LOAD_IMAGE_GRAYSCALE);
    valid = cv::findChessboardCorners(images[i], patternSize, corners);

    if (valid)
    {
      cornerSubPix(images[i], corners, Size(5, 5), Size(-1, -1),
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
      valid_images.push_back(images[i]);
      imagePoints.push_back(corners);
      cv::drawChessboardCorners(images[i], patternSize, corners, valid);
    }
  }

  for (int i = 0; i < CHESS_IMAGES; i++)
  {
    imshow("Imagen " + to_string(i+1), images[i]);
    waitKey();
    destroyAllWindows();
  }

  vector< vector<Point3f> > objectPoints;
  vector<Point3f> points;
  for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 13; j++) {
			Point3f p = Point3f(j,i,0);
			points.push_back(p);
		}
	}

  for (unsigned i = 0; i < valid_images.size(); i++)
  {
    objectPoints.push_back(points);
  }

  Mat cameraMatrix = Mat(3, 3, CV_32F);
	Size imageSize(valid_images[0].cols, valid_images[0].rows);
	Mat distCoeffs = Mat(8, 1, CV_32F);
	vector< Mat > rotationVectors;
	vector< Mat > translationVectors;

  bool distorsion = false;

  int flags = 0;
  if (distorsion)
    flags = CV_CALIB_RATIONAL_MODEL;
  else
    flags = CV_CALIB_ZERO_TANGENT_DIST;

	double error = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rotationVectors, translationVectors, flags);

  cout << "Error calibrando = " << error << endl;

  return 0;
}

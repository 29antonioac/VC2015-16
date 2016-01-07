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
  RNG rng;
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

/*
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
  */

  /* Excercise 3 */

  int Threshl=65;
  int Octaves=3;
  float PatternScales=1.0f;

  vector<Mat> vmort;
  vector<KeyPoint> keypoints[2];
  Mat descriptors[2];

  for (int i = 1; i <= 2; i++)
  {
    vmort.push_back(imread("imagenes/Vmort" + to_string(i) + ".pgm",CV_LOAD_IMAGE_COLOR));
  }

  // Detect and compute descriptors
  Ptr<BRISK> ptrBrisk = BRISK::create(Threshl,Octaves,PatternScales);

  ptrBrisk->detect(vmort[0], keypoints[0]);
  ptrBrisk->compute(vmort[0], keypoints[0],descriptors[0]);

  ptrBrisk->detect(vmort[1], keypoints[1]);
  ptrBrisk->compute(vmort[1], keypoints[1],descriptors[1]);

  BFMatcher matcher(NORM_HAMMING, true);

  vector<DMatch> matches;

  // Match!
  vector<Point2f> corresp[2];
  matcher.match(descriptors[0], descriptors[1], matches);

  // Get correspondence points
  for (int i = 0; i < matches.size(); i++)
  {
		corresp[0].push_back(keypoints[0][ matches[i].queryIdx ].pt);
		corresp[1].push_back(keypoints[1][ matches[i].trainIdx ].pt);
	}

  // Compute fundamental matrix
  Mat F = cv::findFundamentalMat(corresp[0], corresp[1], CV_FM_RANSAC, 0.001);
  cout << F << endl;

  // Compute epilines
  vector<Vec3f> epilines[2];

  computeCorrespondEpilines(corresp[0], 1, F, epilines[1]);
  computeCorrespondEpilines(corresp[1], 2, F, epilines[0]);

  double distance = 0.0;
  int epilineIndex;

  for (epilineIndex = 0; epilineIndex < epilines[0].size() && epilineIndex < 200; epilineIndex++)
  {
    Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
    for (int image = 0; image < 2; image++)
    {
  		Vec3f epiline = epilines[image].at(epilineIndex);
  		Point p = Point(0, -epiline[2] / epiline[1]);
  		Point q = Point(vmort[image].cols, (-epiline[2] - epiline[0] * vmort[image].cols) / epiline[1]);

      distance += fabs(epiline[0] * corresp[image][epilineIndex].x + epiline[1] * corresp[image][epilineIndex].y + epiline[2]) / sqrt(epiline[0]*epiline[0] + epiline[1]*epiline[1]);

  		line(vmort[image], p, q, color);
  		circle(vmort[image], corresp[image][epilineIndex], 5, color);
  	}
  }

  // Compute average of distance between correspondences and epilines
  distance /= 2 * (epilineIndex - 1);

  cout << "Average distance = " << distance << endl;

  imshow("Vmort 1", vmort[0]);
  imshow("Vmort 2", vmort[1]);

  waitKey();
  destroyAllWindows();


  return 0;
}

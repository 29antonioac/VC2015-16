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

RNG rng;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void exercise1()
{
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
}

void exercise2()
{
  const int CHESS_IMAGES = 25;
  bool valid;
  vector<Point2f> corners;
  vector< vector<Point2f> > imagePoints;
  vector<Mat> images(CHESS_IMAGES);
  vector<Mat> valid_images;
  Size patternSize(13,12);

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

  cout << "CameraMatrix = " << cameraMatrix << endl;
  cout << "Error calibrando = " << error << endl;
}

void exercise3()
{
  vector<Mat> vmort;
  vector<KeyPoint> keypoints[2];
  Mat descriptors[2];

  for (int i = 1; i <= 2; i++)
  {
    vmort.push_back(imread("imagenes/Vmort" + to_string(i) + ".pgm",CV_LOAD_IMAGE_COLOR));
  }

  // Detect and compute descriptors
  Ptr<BRISK> ptrBrisk = BRISK::create(65);

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
  vector<unsigned char> taken;
  Mat F = cv::findFundamentalMat(corresp[0], corresp[1], CV_FM_8POINT | CV_FM_RANSAC, 1, 0.99, taken);
  cout << F << endl;

  vector<Point2f> right_corresp[2];
  for (int i = 0; i < corresp[0].size(); i++)
  {
    if ((int)taken[i] == 1)
    {
      right_corresp[0].push_back(corresp[0][i]);
      right_corresp[1].push_back(corresp[1][i]);
    }
  }

  // Compute epilines
  vector<Vec3f> epilines[2];

  computeCorrespondEpilines(right_corresp[0], 1, F, epilines[1]);
  computeCorrespondEpilines(right_corresp[1], 2, F, epilines[0]);

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

      distance += fabs(epiline[0] * right_corresp[image][epilineIndex].x + epiline[1] * right_corresp[image][epilineIndex].y + epiline[2]) / sqrt(epiline[0]*epiline[0] + epiline[1]*epiline[1]);

      line(vmort[image], p, q, color);
      circle(vmort[image], right_corresp[image][epilineIndex], 5, color);
    }
  }

  // Compute average of distance between correspondences and epilines
  distance /= 2 * (epilineIndex - 1);

  cout << "Average distance = " << distance << endl;

  imshow("Vmort 1", vmort[0]);
  imshow("Vmort 2", vmort[1]);

  waitKey();
  destroyAllWindows();
}

void exercise4()
{
  const unsigned IMAGES = 3;
  vector<Mat> reconstruccion;
  Mat cameraMatrix(Mat(3, 3, CV_64FC1, new double[3][3] {{1839.6300000000001091, 0.0, 1024.2000000000000455 },
                                                              {0.0, 1848.0699999999999363, 686.5180000000000291},
                                                              {0.0, 0.0, 1.0} } ));;

  double f_i = 1839.6300000000001091;
  double f_d = f_i;

  // Load images
  reconstruccion.push_back(imread("imagenes/rdimage.000.ppm", CV_LOAD_IMAGE_COLOR));
  reconstruccion.push_back(imread("imagenes/rdimage.001.ppm", CV_LOAD_IMAGE_COLOR));
  reconstruccion.push_back(imread("imagenes/rdimage.004.ppm", CV_LOAD_IMAGE_COLOR));

  // Detect correspondences between pairs
  vector<KeyPoint> keypoints[IMAGES];
  Mat descriptors[IMAGES];

  Ptr<BRISK> ptrBrisk = BRISK::create(35);

  for (unsigned i = 0; i < IMAGES; i++)
  {
    ptrBrisk->detect(reconstruccion[i], keypoints[i]);
    ptrBrisk->compute(reconstruccion[i], keypoints[i],descriptors[i]);
  }

  BFMatcher matcher(NORM_HAMMING, true);
  vector<DMatch> matches[IMAGES];

  // Match!
  vector<Point2f> corresp[IMAGES][2];
  for (unsigned i = 0; i < IMAGES - 1; i++)
    matcher.match(descriptors[i], descriptors[i+1], matches[i]);
  matcher.match(descriptors[0], descriptors[IMAGES - 1], matches[IMAGES - 1]);

  // Get correspondence points
  for (int image = 0; image < IMAGES - 1; image++)
    for (int i = 0; i < matches[image].size(); i++)
    {
      corresp[image][0].push_back(keypoints[0][ matches[image][i].queryIdx ].pt);
      corresp[image][1].push_back(keypoints[1][ matches[image][i].trainIdx ].pt);
    }
  for (int i = 0; i < matches[IMAGES - 1].size(); i++)
  {
    corresp[IMAGES - 1][0].push_back(keypoints[0][ matches[IMAGES - 1][i].queryIdx ].pt);
    corresp[IMAGES - 1][1].push_back(keypoints[1][ matches[IMAGES - 1][i].trainIdx ].pt);
  }

  // Get fundamental matrix between cameras
  vector<Mat> fundamentals;
  for (unsigned i = 0; i < IMAGES - 1; i++)
    fundamentals.push_back(cv::findFundamentalMat(corresp[i][0], corresp[i][1], CV_FM_8POINT | CV_FM_RANSAC, 1));
  fundamentals.push_back(cv::findFundamentalMat(corresp[IMAGES - 1][0], corresp[IMAGES - 1][1], CV_FM_8POINT | CV_FM_RANSAC, 1));

  // Get essential matrix for cameras
  vector<Mat> essentials;
  for (unsigned i = 0; i < IMAGES - 1; i++)
    essentials.push_back(cameraMatrix.t() * fundamentals[i] * cameraMatrix);
  essentials.push_back(cameraMatrix.t() * fundamentals[IMAGES - 1] * cameraMatrix);

  vector< pair<Mat, Vec3d> > Rt;

  // Algorithm!
  for (unsigned i = 0; i < essentials.size(); i++)
  {
    cout << "Fundamental " << fundamentals[i] << endl;
    Mat E = essentials[i];

    Mat eet = E.t() * E;
    eet /= trace(eet).val[0] / 2;
    eet = Mat::eye(3,3,CV_64FC1) - eet;
    E /= sqrt(trace(eet).val[0] / 2);


    // Getting translation
    int max_row = 0;
    if(eet.at<double>(1,1) > eet.at<double>(0,0))
		  max_row = 1;
    if(eet.at<double>(2,2) > eet.at<double>(max_row,max_row))
		  max_row = 2;

    Vec3d T(eet.row(max_row));
    T /= sqrt(eet.at<double>(max_row,max_row));

    Mat R, R1, R2, R3;

    for (unsigned p = 0; p < corresp[i][0].size(); p++)
    {
      Point2d p_i = corresp[i][0][p];
      Point2d p_d = corresp[i][1][p];

      double x_d = p_d.x;

      double Z_i = -1.0, Z_d = 1.0;

      while (Z_i <= 0.0 || Z_d <= 0.0)
      {
        // cout << "Z_i = " << Z_i << ", Z_d = " << Z_d << endl;
        if ((Z_i < 0.0 && Z_d > 0.0) || (Z_i > 0.0 && Z_d < 0.0))
        {
          // cout << "DISTINTOS" << endl;
          E = -E;

          // Get w
          Mat w[3];
          // cout << type2str(E.row(0).type()) << "," << E.row(0).rows << "x" << E.row(0).cols << endl;
          Mat tmp(T, CV_64FC1);
          tmp = tmp.t();

          w[0] = E.row(0).cross(tmp);
          w[1] = E.row(1).cross(tmp);
          w[2] = E.row(2).cross(tmp);

          // cout << type2str(w[0].type()) << "," << w[0].rows << "x" << w[0].cols << endl;

          R1 = w[0] + w[1].cross(w[2]);
          R2 = w[1] + w[2].cross(w[0]);
          R3 = w[2] + w[0].cross(w[1]);

          R = Mat(3,3, CV_64FC1);
          R1.copyTo(R.row(0));
          R2.copyTo(R.row(1));
          R3.copyTo(R.row(2));

          // Z_i = Z_d = -1.0;
          Mat p_hom = Mat(Vec3d(p_i.x, p_i.y, 1.0));
          Mat T_mat = Mat(T);

          // Calcular Z_i y Z_d
          Mat aux = (f_d * R1 - x_d*R3);

          Mat num = aux * T_mat;
          Mat den = aux * p_hom;

          Mat m_Z_i = f_i * num / den;

          Z_i = m_Z_i.at<double>(0,0);

          Mat pt_3D_i = Z_i * p_hom / f_i;

          Mat m_Z_d = R2 * (pt_3D_i - T_mat);
          Z_d = m_Z_d.at<double>(0,0);

        }

        if (Z_i < 0.0 && Z_d < 0.0)
        {
          // cout << "Iguales" << endl;
          T = -T;
          Mat p_hom = Mat(Vec3d(p_i.x, p_i.y, 1.0));
          Mat T_mat = Mat(T);

          // Calcular Z_i y Z_d
          Mat aux = (f_d * R1 - x_d*R3);

          Mat num = aux * T_mat;
          Mat den = aux * p_hom;

          Mat m_Z_i = f_i * num / den;

          Z_i = m_Z_i.at<double>(0,0);

          Mat pt_3D_i = Z_i * p_hom / f_i;

          Mat m_Z_d = R2 * (pt_3D_i - T_mat);
          Z_d = m_Z_d.at<double>(0,0);
        }
      }
    }

    Rt.push_back(pair<Mat,Vec3d>(R,T));
  }



  for (unsigned i = 0; i < Rt.size(); i++)
  {
    cout << "R = " << Rt[i].first << "\nT = " << Rt[i].second << "\n" << endl;
  }


}


int main(int argc, char const *argv[]) {
  theRNG().state = clock();

  // exercise1();
  // exercise2();
  // exercise3();
  exercise4();

  return 0;
}

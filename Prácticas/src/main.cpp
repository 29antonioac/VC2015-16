#include "../inc/image.hpp"
#include "../inc/utils.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;


int main(int argc, char const *argv[]) {

  /* Excersice 1 */
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

  // waitKey(0);
  // destroyAllWindows();

  /* Excersice 2 */
  Mat im = imread("imagenes/yosemite1.jpg");
  Mat im_2 = imread("imagenes/yosemite2.jpg");

  Mat im_copy = im.clone();
  Mat im_2_copy = im_2.clone();

  std::vector<cv::KeyPoint> keypointsA[2], keypointsB[2];
  Mat descriptorsA[2], descriptorsB[2];

  int Threshl=65;
  int Octaves=3;
  float PatternScales=1.0f;

  circle_radius = 5;
  circle_thickness = 1;

  /* BRISK */
  cv::Ptr<cv::BRISK> ptrBrisk = cv::BRISK::create(Threshl,Octaves,PatternScales);

  ptrBrisk->detect(im, keypointsA[0]);
  ptrBrisk->compute(im, keypointsA[0],descriptorsA[0]);

  ptrBrisk->detect(im_2, keypointsA[1]);
  ptrBrisk->compute(im_2, keypointsA[1],descriptorsA[1]);

  for (int i = 0; i < keypointsA[0].size(); i++)
  {
    circle(im,keypointsA[0][i].pt, circle_radius, Scalar(255,0,0), circle_thickness);
  }

  for (int i = 0; i < keypointsA[1].size(); i++)
  {
    circle(im_2,keypointsA[1][i].pt, circle_radius, Scalar(255,0,0), circle_thickness);
  }

  /* ORB */
  cv::Ptr<cv::ORB> ptrORB = cv::ORB::create();

  ptrORB->detect(im_copy, keypointsB[0]);
  ptrORB->compute(im_copy, keypointsB[0],descriptorsB[0]);

  ptrORB->detect(im_2_copy, keypointsB[1]);
  ptrORB->compute(im_2_copy, keypointsB[1],descriptorsB[1]);



  for (int i = 0; i < keypointsB[0].size(); i++)
  {
    circle(im_copy,keypointsB[0][i].pt, circle_radius, Scalar(0,0,255), circle_thickness);
  }

  for (int i = 0; i < keypointsB[1].size(); i++)
  {
    circle(im_2_copy,keypointsB[1][i].pt, circle_radius, Scalar(0,0,255), circle_thickness);
  }

  imshow("Brisk", im);
  imshow("Brisk2", im_2);
  imshow("ORB1",im_copy);
  imshow("ORB2",im_2_copy);
  waitKey(0);
  destroyAllWindows();

  /* Excersice 3 */

  // Neccesary for FlannBasedMatcher!
  for (int i = 0; i < 2; i++)
  {
    if (descriptorsA[i].type() != CV_32F)
      descriptorsA[i].convertTo(descriptorsA[i], CV_32F);
  }

  for (int i = 0; i < 2; i++)
  {
    if (descriptorsB[i].type() != CV_32F)
      descriptorsB[i].convertTo(descriptorsB[i], CV_32F);
  }
  // -------------------------------

  /* BFMatcher */
  BFMatcher BFmatcherBRISK(NORM_L2, true);
  BFMatcher BFmatcherORB(NORM_L2, true);

  vector<DMatch> BFmatchesBRISK, BFmatchesORB;
  BFmatcherBRISK.match(descriptorsA[0], descriptorsA[1], BFmatchesBRISK);
  BFmatcherORB.match(descriptorsB[0], descriptorsB[1], BFmatchesORB);

  Mat BFall_matchesBRISK, BFall_matchesORB;
  drawMatches( im, keypointsA[0], im_2, keypointsA[1],
                       BFmatchesBRISK, BFall_matchesBRISK, Scalar::all(-1), Scalar::all(-1),
                       vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  drawMatches( im_copy, keypointsB[0], im_2_copy, keypointsB[1],
                      BFmatchesORB, BFall_matchesORB, Scalar::all(-1), Scalar::all(-1),
                      vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  imshow( "BRISK All Matches BFMatcher", BFall_matchesBRISK );
  imshow( "ORB All Matches BFMatcher", BFall_matchesORB );
  waitKey(0);
  destroyAllWindows();

  /* FlannBasedMatcher */

  FlannBasedMatcher FlmatcherBRISK;
  FlannBasedMatcher FlmatcherORB;

  vector<DMatch> FlmatchesBRISK, FlmatchesORB;

  FlmatcherBRISK.match(descriptorsA[0], descriptorsA[1], FlmatchesBRISK);

  FlmatcherORB.match(descriptorsB[0], descriptorsB[1], FlmatchesORB);

  Mat Flall_matchesBRISK, Flall_matchesORB;
  drawMatches( im, keypointsA[0], im_2, keypointsA[1],
                       FlmatchesBRISK, Flall_matchesBRISK, Scalar::all(-1), Scalar::all(-1),
                       vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  drawMatches( im_copy, keypointsB[0], im_2_copy, keypointsB[1],
                      FlmatchesORB, Flall_matchesORB, Scalar::all(-1), Scalar::all(-1),
                      vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  imshow( "BRISK All Matches FlannMatcher", Flall_matchesBRISK );
  imshow( "ORB All Matches FlannMatcher", Flall_matchesORB );
  waitKey(0);
  destroyAllWindows();

  /* After this point I implement a function that takes two images and gets
   * their correspondece points using BRISK and BruteForceMatcher + crossCheck
   */
  /* Exercise 4 */

  Mat another_yosemite = imread("imagenes/yosemite1.jpg");
  Mat another_yosemite_2 = imread("imagenes/yosemite2.jpg");

  vector<Point2f> pointsOrigin, pointsDestination;

  Mat new_output = Mat::zeros(500,1000,CV_32FC3);
  warpPerspective(another_yosemite, new_output, Mat::eye(3, 3, CV_32F), Size(new_output.cols, new_output.rows),INTER_LINEAR,BORDER_CONSTANT);

  pointsOrigin.clear();
  pointsDestination.clear();

  for( int i = 0; i < BFmatchesBRISK.size(); i++ )
 {
   // Get the keypoints from the matches
   pointsOrigin.push_back( keypointsA[0][ BFmatchesBRISK[i].queryIdx ].pt );
   pointsDestination.push_back( keypointsA[1][ BFmatchesBRISK[i].trainIdx ].pt );
 }

  Mat homography_1 = findHomography(pointsDestination, pointsOrigin, CV_RANSAC);
  homography_1.convertTo(homography_1, CV_32F);


  warpPerspective(another_yosemite_2, new_output,  homography_1, Size(new_output.cols, new_output.rows),INTER_LINEAR,BORDER_TRANSPARENT);

  new_output.convertTo(new_output, CV_8U);


  imshow("Yosemite1", another_yosemite);
  imshow("Yosemite2", another_yosemite_2);
  imshow("RANSAC output", new_output);
  waitKey(0);
  destroyAllWindows();
  // ------------------------------------------------------------------------------------
  /* Excersice 5 */

  const int YOSEMITE_IMAGES = 10;
  Mat yosemite[YOSEMITE_IMAGES];
  vector<KeyPoint> keypointsYosemite[YOSEMITE_IMAGES];
  Mat descriptorsYosemite[YOSEMITE_IMAGES];
  vector<DMatch> BFMatchesYosemite[YOSEMITE_IMAGES-1];
  Mat actual_homography, first_homography, final_homography;

  // Using previous ptrBrisk and BFmatcherBRISK

  std::ostringstream ss;


  cout << "Loading yosemite images..." << endl;
  for (int i = 0; i < YOSEMITE_IMAGES; i++)
  {
    ss << setw(3) << setfill('0') << (i+2);
    // string filename = string("imagenes/yosemite") + std::to_string(i+1) + string(".jpg");
    string filename = string("imagenes/mosaico") + ss.str() + string(".jpg");
    cout << filename << endl;
    ss.str(string());
    ss.clear();
    yosemite[i] = imread(filename, CV_LOAD_IMAGE_COLOR);
    ptrBrisk->detect(yosemite[i], keypointsYosemite[i]);
    ptrBrisk->compute(yosemite[i], keypointsYosemite[i], descriptorsYosemite[i]);

    descriptorsYosemite[i].convertTo(descriptorsYosemite[i], CV_32F);
  }

  cout << "Getting matches..." << endl;
  for (int i = 0; i < YOSEMITE_IMAGES - 1; i++)
  {
    BFmatcherBRISK.match(descriptorsYosemite[i], descriptorsYosemite[i+1], BFMatchesYosemite[i]);
  }

  cout << "Initializing output..." << endl;
  Mat yosemite_output = Mat::zeros((int)3*yosemite[0].rows, 7 * yosemite[0].cols, CV_32FC3);
  cout << "HOla" << endl;

  int offset_x = 600;
  int offset_y = 200;

  pointsOrigin.clear();
  pointsDestination.clear();

  pointsOrigin.push_back(Point2f(0, 0));
	pointsOrigin.push_back(Point2f(yosemite[YOSEMITE_IMAGES/2].cols, 0));
	pointsOrigin.push_back(Point2f(0, yosemite[YOSEMITE_IMAGES/2].rows));
	pointsOrigin.push_back(Point2f(yosemite[YOSEMITE_IMAGES/2].cols, yosemite[YOSEMITE_IMAGES/2].rows));
	pointsDestination.push_back(Point2f(offset_x, offset_y));
	pointsDestination.push_back(Point2f(offset_x+yosemite[YOSEMITE_IMAGES/2].cols, offset_y));
	pointsDestination.push_back(Point2f(offset_x, yosemite[YOSEMITE_IMAGES/2].rows+offset_y));
	pointsDestination.push_back(Point2f(yosemite[YOSEMITE_IMAGES/2].cols+offset_x, yosemite[YOSEMITE_IMAGES/2].rows+offset_y));

	//hom_zero = getHomography(p1, p2);
	first_homography = findHomography(pointsOrigin, pointsDestination);

	first_homography.convertTo(first_homography, CV_32FC1);
  final_homography = first_homography.clone();
  // final_homography = Mat::eye(3,3, CV_32F);
  cout << "HOla2" << endl;
  warpPerspective(yosemite[YOSEMITE_IMAGES/2], yosemite_output, first_homography, Size(yosemite_output.cols, yosemite_output.rows), INTER_LINEAR, BORDER_CONSTANT);

  cout << "Getting output..." << endl;

  for (int i = YOSEMITE_IMAGES/2+1; i < YOSEMITE_IMAGES; i++)
  {
    cout << "Mixing " << i << " with " << i+1 << endl;
    pointsOrigin.clear();
    pointsDestination.clear();

    for( int j = 0; j < BFMatchesYosemite[i-1].size(); j++ )
    {
     // Get the keypoints from the matches
     pointsOrigin.push_back( keypointsYosemite[i-1][ BFMatchesYosemite[i-1][j].queryIdx ].pt );
     pointsDestination.push_back( keypointsYosemite[i][ BFMatchesYosemite[i-1][j].trainIdx ].pt );
    }
    actual_homography = findHomography(pointsDestination, pointsOrigin, CV_RANSAC);
    actual_homography.convertTo(actual_homography, CV_32F);

    final_homography = final_homography * actual_homography;

    warpPerspective(yosemite[i], yosemite_output, final_homography, Size(yosemite_output.cols, yosemite_output.rows), INTER_LINEAR, BORDER_TRANSPARENT);
  }

  final_homography = first_homography.clone();

  for (int i = YOSEMITE_IMAGES/2 - 1; i >= 0; i--)
  {
    cout << "Mixing " << i+1 << " with " << i << endl;
    pointsOrigin.clear();
    pointsDestination.clear();

    for( int j = 0; j < BFMatchesYosemite[i].size(); j++ )
    {
     // Get the keypoints from the matches
     pointsOrigin.push_back( keypointsYosemite[i][ BFMatchesYosemite[i][j].queryIdx ].pt );
     pointsDestination.push_back( keypointsYosemite[i+1][ BFMatchesYosemite[i][j].trainIdx ].pt );
    }
    actual_homography = findHomography(pointsOrigin, pointsDestination, CV_RANSAC);
    actual_homography.convertTo(actual_homography, CV_32F);

    final_homography = final_homography * actual_homography;

    warpPerspective(yosemite[i], yosemite_output, final_homography, Size(yosemite_output.cols, yosemite_output.rows), INTER_LINEAR, BORDER_TRANSPARENT);
  }

  yosemite_output.convertTo(yosemite_output, CV_8U);

  namedWindow("Yosemite Panorama", WINDOW_NORMAL);
  // resizeWindow("Yosemite Panorama", 768, 1366);
  imshow("Yosemite Panorama", yosemite_output);

  waitKey(0);
  destroyAllWindows();

  return 0;
}

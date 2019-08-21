#include "calibrator.h"

void CameraCalibrator::addPoints(std::vector<cv::Point2f>& imgCorners, std::vector<cv::Point3f>& objCorners)
{
	imagePoints.push_back(imgCorners);
	objectPoints.push_back(objCorners);

	return;
}

int CameraCalibrator::addChessboardPoints(const std::vector<std::string>& filelist,
	cv::Size& boardSize, bool visualize) {
	// the points on the chessboard
	std::vector<cv::Point2f> imageCorners;
	std::vector<cv::Point3f> objectCorners;

	// 3D Scene Points:
	// Initialize the chessboard corners
	// in the chessboard reference frame
	// The corners are at 3D location (X,Y,Z)= (i,j,0)
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			objectCorners.push_back(cv::Point3f(i, j, 0.0f));

	// 2D Image Points:
	cv::Mat image; // to contain chessboard image
	cv::Mat gray;
	int successes = 0;
	
	// for all viewpoints
	for (int i = 0; i < filelist.size(); i++) {
		// open the image
		image = cv::imread(filelist[i]);
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

		// get the chessboard corners
		bool found = cv::findChessboardCorners(image, boardSize, imageCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		// get subpixel accuracy on the corners
		if (found) {
			cv::cornerSubPix(gray, imageCorners, cv::Size(5, 5),// half size of search window
				cv::Size(-1, -1),
				cv::TermCriteria(cv::TermCriteria::MAX_ITER +
					cv::TermCriteria::EPS, 30,// max number of iterations
					0.1)); //min accuracy
		}

		// if we have a good board, add it to our data
		if (imageCorners.size() == boardSize.area()) {
			// add image and scene points from one view
			addPoints(imageCorners, objectCorners);
			successes++;
		}

		//Draw the corners
		if (visualize) {
			cv::drawChessboardCorners(gray, boardSize, imageCorners, found);
			cv::imshow("Corners on Chessboard", image);
			cv::waitKey(1000);
		}
	}

	return successes;
}

double CameraCalibrator::calibrate(cv::Size& imageSize)
{
	// Calibrate the camera
	// return the re-projection error
	// output rotations and translations
	std::vector<cv::Mat> rvecs, tvecs;

	// start calibration
	return cv::calibrateCamera(objectPoints,
		imagePoints,
		imageSize,
		cameraMatrix,
		distCoeffs,
		rvecs, tvecs,
		flag);
}

// Set the calibration options
// 8radialCoeffEnabled should be true if 8 radial coefficients are required (5 is default)
// tangentialParamEnabled should be true if tangeantial distortion is present
void CameraCalibrator::setCalibrationFlag(bool radial8CoeffEnabled, bool tangentialParamEnabled) {

	// Set the flag used in cv::calibrateCamera()
	flag = 0;
	if (!tangentialParamEnabled) flag += CV_CALIB_ZERO_TANGENT_DIST;
	if (radial8CoeffEnabled) flag += CV_CALIB_RATIONAL_MODEL;
}

cv::Mat CameraCalibrator::remap(const cv::Mat& image)
{
	cv::Mat undistorted;
	if (mustInitUndistort) { // called once per calibration
		cv::initUndistortRectifyMap(
			cameraMatrix, // computed camera matrix
			distCoeffs, // computed distortion matrix
			cv::Mat(), // optional rectification (none)
			cv::Mat(), // camera matrix to generate undistorted
			image.size(), // size of undistorted
			CV_32FC1, // type of output map
			map1, map2); // the x and y mapping functions
		mustInitUndistort = false;
	}
	// Apply mapping functions
	cv::remap(image, undistorted, map1, map2,
		cv::INTER_LINEAR); // interpolation type
	return undistorted;
}

void CameraCalibrator::showUndistortedImages(const std::vector<std::string>& filelist)
{
	cv::Mat image, undist_image;

	for (int i = 0; i < filelist.size(); i++) {
		// open the image
		image = cv::imread(filelist[i]);
		undist_image = remap(image);
		cv::imshow("Undistorted Image", undist_image);
		cv::waitKey(15000);
	}

	return;
}
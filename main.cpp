#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d/calib3d_c.h>

#include <filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;

class CameraCalibrator {
	//input points;
	//the points in world coordinates
	vector<vector<cv::Point3f>> objectPoints;
	// the image pint positions in pixels
	vector<vector<cv::Point2f>> imagePoints;
	//output matrices
	cv::Mat cameraMatrix, distCoeffs;
	// used in image undistortion 
	cv::Mat map1, map2;

	// flag to specify how calibration is done
	int flag;
	bool mustInitUndistort;

	void addPoints(vector<cv::Point2f> &imgCorners, vector<cv::Point3f> &objCorners);

public:
	CameraCalibrator() : flag(0), mustInitUndistort(true) {};
	int addChessboardPoints(const vector<string>& filelist,
		cv::Size& boardSize,bool visualize=false);
	double calibrate(cv::Size& imageSize);
	void setCalibrationFlag(bool radial8CoeffEnabled = false, bool tangentialParamEnabled = false);
	// Remove distortion in an image (after calibration)
	cv::Mat remap(const cv::Mat& image);
	void showUndistortedImages(const vector<string>& filelist);
	// Getters
	cv::Mat getCameraMatrix() { return cameraMatrix; }
	cv::Mat getDistCoeffs() { return distCoeffs; }
};

void CameraCalibrator::addPoints(vector<cv::Point2f>& imgCorners, vector<cv::Point3f>& objCorners)
{
	imagePoints.push_back(imgCorners);
	objectPoints.push_back(objCorners);

	return;
}

int CameraCalibrator::addChessboardPoints(const vector<string> &filelist,
	cv::Size & boardSize,bool visualize){
	// the points on the chessboard
	vector<cv::Point2f> imageCorners;
	vector<cv::Point3f> objectCorners;

	// 3D Scene Points:
	// Initialize the chessboard corners
	// in the chessboard reference frame
	// The corners are at 3D location (X,Y,Z)= (i,j,0)
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			objectCorners.push_back(cv::Point3f(i, j, 0.0f));

	// 2D Image Points:
	cv::Mat image; // to contain chessboard image
	int successes = 0;

	// for all viewpoints
	for (int i = 0; i < filelist.size(); i++) {
		// open the image
		image = cv::imread(filelist[i], 0);

		// get the chessboard corners
		bool found = cv::findChessboardCorners(image, boardSize, imageCorners);

		// get subpixel accuracy on the corners
		if (found) {
			cv::cornerSubPix(image,imageCorners,cv::Size(5,5),// half size of search window
				cv::Size(-1,-1),
				cv::TermCriteria(cv::TermCriteria::MAX_ITER +
				cv::TermCriteria::EPS,30,// max number of iterations
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
			cv::drawChessboardCorners(image, boardSize, imageCorners, found);
			cv::imshow("Corners on Chessboard", image);
			cv::waitKey(100);
		}
	}

	return successes;
}

double CameraCalibrator::calibrate(cv::Size& imageSize)
{
	// Calibrate the camera
	// return the re-projection error
	// output rotations and translations
	vector<cv::Mat> rvecs, tvecs;

	// start calibration
	return cv::calibrateCamera(objectPoints,
		imagePoints,
		imageSize,
		cameraMatrix,
		distCoeffs,
		rvecs,tvecs,
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

void CameraCalibrator::showUndistortedImages(const vector<string>& filelist)
{
	cv::Mat image, undist_image;

	for (int i = 0; i < filelist.size(); i++) {
		// open the image
		image = cv::imread(filelist[i], 0);
		undist_image = remap(image);
		cv::imshow("Undistorted Image", undist_image);
		cv::waitKey(15000);
	}

	return;
}

int main()
{
	string path_to_files = "C:\\Users\\PC\\Documents\\KakaoTalk Downloads\\camera calibre";
	vector<string> files;

	CameraCalibrator camCalibrator;
	

	cout << "Start Calibration ..." << endl;
	for (const auto& entry : fs::directory_iterator(path_to_files)) {
		fs::path path = entry.path();
		files.push_back(path.u8string());
	}

	cv::Mat image = cv::imread(files[0], 0);
	cv::Size imgSize = image.size();
	cv::Size boardCells(9, 6);

	int successes = camCalibrator.addChessboardPoints(files, boardCells,true);
	double error = camCalibrator.calibrate(imgSize);
	cv::Mat cameraMatrix = camCalibrator.getCameraMatrix();
	cv::Mat  distCoeffs = camCalibrator.getDistCoeffs();

	cout << "------------------------ Calibration Log ------------------------" << endl;
	cout << "Image Size: " << imgSize << endl;
	cout << "Calibration Error: " << error << endl;
	cout << "Camera Matrix: " << cameraMatrix << endl;
	cout << "Dist Matrix: " << distCoeffs << endl;
	cout << "------------------------ end ------------------------" << endl;

	camCalibrator.showUndistortedImages(files);

	return 0;
}
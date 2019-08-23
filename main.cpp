#include <iostream>
#include <filesystem>
#include "calibrator.h"

namespace fs = std::experimental::filesystem;
using namespace std;

void undistToHLS(cv::Mat& src,cv::Mat& dest, CameraCalibrator& calibrator)
{
	cv::Mat undist_img = calibrator.remap(src);
	cv::cvtColor(undist_img, dest, cv::COLOR_BGR2HLS);
	
	return;
}

void absSobelThresh(cv::Mat& src, cv::Mat& dest, char orient = 'x', int kernel_size = 3, int thresh_min = 0, int thresh_max = 255)
{
	int dx, dy;
	int ddepth = CV_32FC1;

	cv::Mat grad_img, scaled;

	if (orient == 'x') {
		dy = 0;
		dx = 1;
	}
	else {
		dy = 1;
		dx = 0;
	}

	cv::Sobel(src, grad_img, ddepth, dx, dy, kernel_size);
	grad_img = cv::abs(grad_img);
	
	// Scaling 
	double min, max;
	cv::minMaxLoc(grad_img, &min, &max);
	scaled = grad_img.mul(255 / max);
	scaled.convertTo(scaled, CV_8UC1);

	assert(scaled.type() == CV_8UC1);
	cv::inRange(scaled, cv::Scalar(thresh_min), cv::Scalar(thresh_max), dest);

	return;
}
int main()
{
	string path_to_files = "C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\camera_cal";
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

	int successes = camCalibrator.addChessboardPoints(files, boardCells);
	double error = camCalibrator.calibrate(imgSize);
	cv::Mat cameraMatrix = camCalibrator.getCameraMatrix();
	cv::Mat  distCoeffs = camCalibrator.getDistCoeffs();

	cout << "------------------------ Calibration Log ------------------------" << endl;
	cout << "Image Size: " << imgSize << endl;
	cout << "Calibration Error: " << error << endl;
	cout << "Camera Matrix: " << cameraMatrix << endl;
	cout << "Dist Matrix: " << distCoeffs << endl;
	cout << " Success " << successes << endl;
	cout << "------------------------ end ------------------------" << endl;

	//camCalibrator.showUndistortedImages(files);

	string path_to_imgs = "C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\test_images";
	vector<string> images;
	
	bool first = true;
	for (const auto& ent : fs::directory_iterator(path_to_imgs)) {
		fs::path pathe = ent.path();
		if (!first) images.push_back(pathe.u8string());
		first = false;
	}

	cv::Mat sample_img = cv::imread("C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\test_images\\straight_lines1.jpg");

	// undistort and convert to HLS color space
	cv::Mat undist_hls;
	undistToHLS(sample_img, undist_hls, camCalibrator);


	// S channel of HLS space

	cv::Mat hls_channels[3];
	cv::split(undist_hls, hls_channels);

	// absolute Sobel Threshold
	cv::Mat sobel_x, sobel_y, combined;
	absSobelThresh(hls_channels[2], sobel_x, 'x', 3, 10, 160);
	absSobelThresh(hls_channels[2], sobel_y, 'y', 3, 10, 160);
	cv::bitwise_and(sobel_x, sobel_y, combined); // combine gradient images
	

	// Perspective Transform

	int x_size, y_size;
	x_size = undist_hls.size().width;
	y_size = undist_hls.size().height;

	int bottom_y = 720;
	int top_y = 425;

	cv::Scalar color(0, 0, 255);
	cv::Point left1(115, bottom_y), left2(560, top_y), right1(695, top_y), right2(1170, bottom_y);

	int w = 2;
	cv::Mat undistorted = camCalibrator.remap(sample_img);

	cv::line(undistorted, left1, left2, color, w);
	cv::line(undistorted, left2, right1, color, w);
	cv::line(undistorted, right1, right2, color, w);
	cv::line(undistorted, right2, left1, color, w);

	cv::Mat gray, M, Minv, warped;
	cv::cvtColor(undistorted, gray, cv::COLOR_BGR2GRAY);

	int offset = 200, nX, nY;
	nX = gray.size().width;
	nY = gray.size().height;

	cv::Point2f src[4], dst[4];

	src[0] = left2;
	src[1] = right1;
	src[2] = right2;
	src[3] = left1;

	dst[0] = cv::Point2f(offset, 0);
	dst[1] = cv::Point2f(nX - offset, 0);
	dst[2] = cv::Point2f(nX - offset, nY);
	dst[3] = cv::Point2f(offset, nY);

	M = cv::getPerspectiveTransform(src, dst);
	Minv = cv::getPerspectiveTransform(dst, src);
	cv::warpPerspective(undistorted, warped, M, gray.size());

	imshow("Undistorted", undistorted);
	imshow("Perspective Transform", warped);

	cv::waitKey(0);

	return 0;
}
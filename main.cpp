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
	int ddepth = CV_64F;

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
	scaled = 255 * (grad_img / max);
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
	absSobelThresh(hls_channels[2], sobel_x, 'x', 3, 10, 170);
	absSobelThresh(hls_channels[2], sobel_y, 'y', 3, 10, 170);
	combined = sobel_x & sobel_y; // combine gradient images
	

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
	cv::warpPerspective(combined, warped, M, gray.size());

	// Histogram 
	cv::Mat cropped_warped = warped(cv::Rect(0, warped.rows / 2, warped.cols, warped.rows / 2));
	cv::Mat histogram;
	cv::reduce(cropped_warped, histogram, 0, cv::REDUCE_SUM, CV_32S);

	cv::Mat out_img;
	auto channels = vector<cv::Mat>{ cropped_warped,cropped_warped,cropped_warped };
	cv::merge(channels, out_img);

	cv::Mat right_half, left_half;
	cv::Point leftx_base, rightx_base, temp;
	double min, max;
	int midpoint;
	midpoint = histogram.cols / 2;

	left_half = histogram.colRange(0, midpoint);
	right_half = histogram.colRange(midpoint, histogram.cols);
	
	cv::minMaxLoc(left_half, &min, &max, &temp, &leftx_base);
	cv::minMaxLoc(right_half, &min, &max, &temp, &rightx_base);
	rightx_base = rightx_base + cv::Point(midpoint,0);

	// Window  height
	int nwindows = 9;
	int window_height = cropped_warped.rows / nwindows;

	vector<cv::Point> nonzero, nonzero_y, nonzero_x;
	cv::findNonZero(cropped_warped, nonzero);
	
	//cv::findNonZero(nonzero, nonzero_x);
	//cv::findNonZero(nonzero, nonzero_y);


	cv::Point leftx_current, rightx_current;
	leftx_current = leftx_base;
	rightx_current = rightx_base;

	int margin = 110, minpix = 50;
	int win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high;

	vector<cv::Point> good_left_inds, good_right_inds, left_lane_inds, right_lane_inds;

	for (int window = 0; window < nwindows; window++) {

		// Identify window boundaries in x and y (and right and left)
		win_y_low = cropped_warped.rows - (window + 1) * window_height;
		win_y_high = cropped_warped.rows - window * window_height;
		win_xleft_low = leftx_current.x - margin;
		win_xleft_high = leftx_current.x + margin;
		win_xright_low = rightx_current.x - margin;
		win_xright_high = rightx_current.x + margin;

		// Draw the windows on the visualization image
		cv::rectangle(out_img, cv::Point(win_xleft_low, win_y_low), cv::Point(win_xleft_high, win_y_high), cv::Scalar(0, 255, 0), 2);
		cv::rectangle(out_img, cv::Point(win_xright_low, win_y_low), cv::Point(win_xright_high, win_y_high), cv::Scalar(0, 255, 0), 2);

		// Identify the nonzero pixels in x and y within the window
		cv::Rect rect_left(cv::Point(win_xleft_low, win_y_low), cv::Point(win_xleft_high, win_y_high));
		cv::Rect rect_right(cv::Point(win_xright_low, win_y_low), cv::Point(win_xright_high, win_y_high));

		cv::Mat win_left = cropped_warped(rect_left);
		cv::Mat win_right = cropped_warped(rect_right);

		cv::findNonZero(win_left, good_left_inds);
		cv::findNonZero(win_right, good_right_inds);

		// Append these indices to the vector
		// reserve() is optional - just to improve performance

		left_lane_inds.reserve(left_lane_inds.size() + good_left_inds.size());
		left_lane_inds.insert(left_lane_inds.end(), good_left_inds.begin(), good_left_inds.end());

		right_lane_inds.reserve(right_lane_inds.size() + good_right_inds.size());
		right_lane_inds.insert(right_lane_inds.end(), good_right_inds.begin(), good_right_inds.end());

		// If you found > minpix pixels, recenter next window on their mean position
		if ((good_left_inds.size() > minpix) || (good_left_inds.size() > (win_left.cols * win_left.rows) * 0.75)) {
			double sum = 0;
			for (int i = 0; i < good_left_inds.size(); i++) {
				sum += (good_left_inds[i].x + win_xleft_low);

			}
			int mean = sum / good_left_inds.size();
			leftx_current.x = mean;

		}
		// If you found > minpix pixels, recenter next window on their mean position
		if ((good_right_inds.size() > minpix) || ( good_right_inds.size() > (win_right.cols * win_right.rows) * 0.75)) {
			double sum = 0;
			
			for (int i = 0; i < good_right_inds.size(); i++) {
				sum += (good_right_inds[i].x + win_xright_low);

			}
			int mean = sum / good_right_inds.size();
			rightx_current.x = mean;

		}
	}
	imshow("Output", out_img);
	cv::waitKey(0);

	return 0;
}
#include <iostream>
#include <filesystem>
#include <string>
#include "calibrator.h"
#include "utils.h"
#include "windowbox.h"

namespace fs = std::experimental::filesystem;

float calc_curvature(cv::Mat& poly, int height = 1280)
{
	if (poly.empty()) return .0;

	std::vector<double> fity = linspace<double>(0, height - 1, height);
	float y_eval = *(max_element(fity.begin(), fity.end()));

	// Define conversions in xand y from pixels space to meters
	int lane_px_height = 720;
	int lane_px_width = 700;
	float ym_per_pix = (30. / lane_px_height);
	float xm_per_pix = 3.7 / lane_px_width;

	std::vector<double> xs;
	poly_fitx(fity, xs, poly);
	//std::reverse(xs.begin(), xs.end()); // Reverse to match top-to-bottom in y
	cv::Mat x(xs), y(fity);
	x.convertTo(x, CV_32F);
	y.convertTo(y, CV_32F);
	cv::Mat poly_cr = cv::Mat::zeros(3, 1, CV_32F);
	polyfit(y * ym_per_pix, x * xm_per_pix, poly_cr, 2);

	float derivative_1 = 2 * poly_cr.at<float>(2,0) * y_eval * ym_per_pix + poly_cr.at<float>(1,0); // f'(y) = dx/dy = 2Ay + B
	float derivative_2 = 2 * poly_cr.at<float>(2,0); // f''(y) = d^2x/dy^2 = 2A
	float curveradm = pow((1 + pow(derivative_1, 2)), 1.5) / abs(derivative_2);
	
	return curveradm;
}



void drawLine(cv::Mat& img, cv::Mat& left_fit, cv::Mat& right_fit, cv::Mat Minv, cv::Mat& out_img)
{
	int y_max = img.rows;
	std::vector<double> fity = linspace<double>(0, y_max - 1, y_max);
	cv::Mat color_warp = cv::Mat::zeros(img.size(), CV_8UC3);

	// Calculate Points
	std::vector<double> left_fitx, right_fitx;
	poly_fitx(fity, left_fitx, left_fit);
	poly_fitx(fity, right_fitx, right_fit);

	int npoints = fity.size();
	std::vector<cv::Point> pts_left(npoints), pts_right(npoints), pts;
	for (int i = 0; i < npoints; i++) {
		pts_left[i] = cv::Point(left_fitx[i], fity[i]);
		pts_right[i] = cv::Point(right_fitx[i], fity[i]);
	}
	pts.reserve(2 * npoints);
	pts.insert(pts.end(), pts_left.begin(), pts_left.end());
	pts.insert(pts.end(), pts_right.rbegin(), pts_right.rend());
	std::vector<std::vector<cv::Point>> ptsarray{ pts };
	cv::fillPoly(color_warp, ptsarray,cv::Scalar(0, 255, 0));

	cv::Mat new_warp;
	perspective_warp(color_warp, new_warp, Minv);
	cv::addWeighted(img, 1, new_warp, 0.3, 0, out_img);

	return;
}

int main()
{
	std::string filename = "camera.yml";
	bool exists = std::experimental::filesystem::exists(filename);
	CameraCalibrator calibrator;

	if (exists) {
		calibrator.load_settings(filename);
	}
	else {
		std::string path_to_files = "C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\camera_cal";
		std::vector<std::string> chessboard_imgs;
		read_imgs(path_to_files, chessboard_imgs);
		start_calibration(chessboard_imgs, calibrator);
		calibrator.save_as(filename);
	}

	cv::Mat sample_img = cv::imread("faulty.jpg");
	//imshow("faulty", sample_img);
	cv::Mat un = calibrator.remap(sample_img);

	cv::VideoCapture cap("C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\project_video.mp4");
	cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, un.size());
	// Check if camera opened successfully
	if (!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << std::endl;
		exit(EXIT_FAILURE);
	}

	cv::Mat warped, Minv, M;
	// Window  height
	int nwindows = 9;
	int width = 100;
	int nframe = 0;
	while (1) {
		
		cv::Mat frame;
		// Capture frame-by-frame
		cap >> frame;
		nframe++;

		std::cout << nframe << std::endl;
		if (nframe==763) imwrite("faulty.jpg", frame);
		// If the frame is empty, break immediately
		if (frame.empty())
			break;

		// undistort the image
		cv::Mat undistorted = calibrator.remap(frame);
		binary_topdown(undistorted, warped, M, Minv);
		int height = undistorted.rows;

		// Histogram 
		cv::Mat histogram;
		lane_histogram(warped, histogram);

		// Peaks
		cv::Point leftx_base, rightx_base;
		lane_peaks(histogram, leftx_base, rightx_base);

		std::vector<WindowBox> left_boxes, right_boxes;
		calc_lane_windows(warped, nwindows, width, left_boxes, right_boxes);

		cv::Mat left_fit = calc_fit_from_boxes(left_boxes);
		cv::Mat right_fit = calc_fit_from_boxes(right_boxes);

		// generate x and values for plotting
		std::vector<double> fitx, fity, left_fitx, right_fitx, hist_fity, new_left_fitx, new_right_fitx;
		fity = linspace<double>(0, warped.rows - 1, warped.rows);
		fitx = linspace<double>(0, warped.cols - 1, warped.cols);

		for (int i = 0; i < histogram.cols; i++)
			hist_fity.push_back(height - histogram.at<int>(0, i));

		poly_fitx(fity, left_fitx, left_fit);
		poly_fitx(fity, right_fitx, right_fit);

		cv::Mat warp_back = cv::Mat::zeros(undistorted.size(), CV_8UC3);

		drawLine(undistorted, left_fit, right_fit, Minv, warp_back);

		// Write the frame into the file 'outcpp.avi'
		//video.write(warp_back);
	}


	// When everything done, release the video capture object
	cap.release();
	video.release();


	//cv::Mat sample_img = cv::imread("C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\test_images\\test2.jpg");

	// undistort the image
	//cv::Mat undistorted = calibrator.remap(sample_img);
	//cv::Mat warped, Minv, M;
	//binary_topdown(undistorted, warped, M, Minv);
	//imshow("warped", warped);
	//// Histogram 
	//cv::Mat histogram;
	//lane_histogram(warped, histogram);

	//// Peaks
	//cv::Point leftx_base, rightx_base;
	//lane_peaks(histogram, leftx_base, rightx_base);


	//std::vector<WindowBox> left_boxes, right_boxes;
	//calc_lane_windows(warped, nwindows,width,left_boxes,right_boxes);

	//// ------------------------------
	//int height = undistorted.rows;

	//// create output image
	//cv::Mat out_img;
	//auto channels = std::vector<cv::Mat>{ warped,warped,warped };
	//cv::merge(channels, out_img);

	//// Draw the windows on the visualization image
	//cv::Point pnt1, pnt2;
	//for (const auto& box : left_boxes) {
	//	pnt1 = box.get_bottom_left_point();
	//	pnt2 = box.get_top_right_point();
	//	cv::rectangle(out_img, pnt1, pnt2, cv::Scalar(0, 255, 0), 2);
	//}

	//for (const auto& box : right_boxes) {
	//	pnt1 = box.get_bottom_left_point();
	//	pnt2 = box.get_top_right_point();
	//	cv::rectangle(out_img, pnt1, pnt2, cv::Scalar(0, 255, 0), 2);
	//}

	//cv::Mat left_fit = calc_fit_from_boxes(left_boxes);
	//cv::Mat right_fit = calc_fit_from_boxes(right_boxes);

	//// generate x and values for plotting
	//std::vector<double> fitx, fity, left_fitx, right_fitx, hist_fity,new_left_fitx,new_right_fitx;
	//fity = linspace<double>(0, warped.rows - 1, warped.rows);
	//fitx = linspace<double>(0, warped.cols - 1, warped.cols);
	//
	//for (int i = 0; i < histogram.cols; i++)
	//	hist_fity.push_back(height - histogram.at<int>(0, i));

	//poly_fitx(fity, left_fitx, left_fit);
	//poly_fitx(fity, right_fitx, right_fit);
	//
	//cv::Scalar red(0, 0, 255), blue(255, 0, 0), yellow(153, 255, 255);
	//// draw polynomial curve
	//draw_polyline(out_img, left_fitx, fity, blue);
	//draw_polyline(out_img, right_fitx, fity, blue);
	//draw_polyline(out_img, fitx, hist_fity, red);

	//std::cout << calc_curvature(left_fit) << std::endl;
	//std::cout << calc_curvature(right_fit) << std::endl;
	///*int margin = 100;
	//cv::Mat new_left_fit, new_right_fit;
	//calc_lr_fit_from_polys(warped, left_fit, right_fit, new_left_fit, new_right_fit, margin);
	//poly_fitx(fity, new_left_fitx, new_left_fit);
	//poly_fitx(fity, new_right_fitx, new_right_fit);*/


	//cv::namedWindow("Undistorted Image", cv::WINDOW_NORMAL);
	//cv::resizeWindow("Undistorted Image", 600,400);
	//imshow("Undistorted Image", undistorted);
	//cv::Mat comb;
	//combined_threshold(undistorted, comb);
	//cv::namedWindow("Binary Image", cv::WINDOW_NORMAL);
	//cv::resizeWindow("Binary Image", 600, 400);
	//imshow("Binary Image", comb);
	//
	//cv::Mat warp_back = cv::Mat::zeros(undistorted.size(),CV_8UC3);

	//drawLine(undistorted, left_fit, right_fit, Minv, warp_back);

	//imshow("Warped back", warp_back);
	//cv::waitKey(0);

	return 0;
}
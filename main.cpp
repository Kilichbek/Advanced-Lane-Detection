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
	std::reverse(xs.begin(), xs.end()); // Reverse to match top-to-bottom in y
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

int main()
{
	std::string path_to_files = "C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\camera_cal";
	std::vector<std::string> chessboard_imgs;
	read_imgs(path_to_files, chessboard_imgs);

	CameraCalibrator calibrator;
	start_calibration(chessboard_imgs, calibrator);

	cv::Mat sample_img = cv::imread("C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\test_images\\test1.jpg");

	// undistort the image
	cv::Mat undistorted = calibrator.remap(sample_img);
	cv::Mat warped;
	binary_topdown(undistorted, warped);

	// Histogram 
	cv::Mat histogram;
	lane_histogram(warped, histogram);

	// Peaks
	cv::Point leftx_base, rightx_base;
	lane_peaks(histogram, leftx_base, rightx_base);

	// Window  height
	int nwindows = 9;
	int width = 220;
	std::vector<WindowBox> left_boxes, right_boxes;
	calc_lane_windows(warped, nwindows,width,left_boxes,right_boxes);

	// ------------------------------
	int height = undistorted.rows;

	// create output image
	cv::Mat out_img;
	auto channels = std::vector<cv::Mat>{ warped,warped,warped };
	cv::merge(channels, out_img);

	// Draw the windows on the visualization image
	cv::Point pnt1, pnt2;
	for (const auto& box : left_boxes) {
		pnt1 = box.get_bottom_left_point();
		pnt2 = box.get_top_right_point();
		cv::rectangle(out_img, pnt1, pnt2, cv::Scalar(0, 255, 0), 2);
	}

	for (const auto& box : right_boxes) {
		pnt1 = box.get_bottom_left_point();
		pnt2 = box.get_top_right_point();
		cv::rectangle(out_img, pnt1, pnt2, cv::Scalar(0, 255, 0), 2);
	}

	cv::Mat left_fit = calc_fit_from_boxes(left_boxes);
	cv::Mat right_fit = calc_fit_from_boxes(right_boxes);

	// generate x and values for plotting
	std::vector<double> fitx, fity, left_fitx, right_fitx, hist_fity,new_left_fitx,new_right_fitx;
	fity = linspace<double>(0, warped.rows - 1, warped.rows);
	fitx = linspace<double>(0, warped.cols - 1, warped.cols);
	
	for (int i = 0; i < histogram.cols; i++)
		hist_fity.push_back(height - histogram.at<int>(0, i));

	poly_fitx(fity, left_fitx, left_fit);
	poly_fitx(fity, right_fitx, right_fit);
	
	cv::Scalar red(0, 0, 255), blue(255, 0, 0), yellow(153, 255, 255);
	// draw polynomial curve
	draw_polyline(out_img, left_fitx, fity, blue);
	draw_polyline(out_img, right_fitx, fity, blue);
	draw_polyline(out_img, fitx, hist_fity, red);

	std::cout << calc_curvature(left_fit) << std::endl;
	std::cout << calc_curvature(right_fit) << std::endl;
	/*int margin = 100;
	cv::Mat new_left_fit, new_right_fit;
	calc_lr_fit_from_polys(warped, left_fit, right_fit, new_left_fit, new_right_fit, margin);
	poly_fitx(fity, new_left_fitx, new_left_fit);
	poly_fitx(fity, new_right_fitx, new_right_fit);*/


	cv::namedWindow("Undistorted Image", cv::WINDOW_NORMAL);
	cv::resizeWindow("Undistorted Image", 600,400);
	imshow("Undistorted Image", undistorted);
	cv::Mat comb;
	combined_threshold(undistorted, comb);
	cv::namedWindow("Binary Image", cv::WINDOW_NORMAL);
	cv::resizeWindow("Binary Image", 600, 400);
	imshow("Binary Image", comb);
	cv::waitKey(0);

	return 0;
}
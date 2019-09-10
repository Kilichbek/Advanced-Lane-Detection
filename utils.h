#ifndef __UTILS_H__
#define __UTILS_H__

#include "windowbox.h"

void polyfit(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst, int order)
{
	CV_Assert((src_x.rows > 0) && (src_y.rows > 0) && (src_x.cols == 1) && (src_y.cols == 1)
		&& (dst.cols == 1) && (dst.rows == (order + 1)) && (order >= 1));
	cv::Mat X;
	X = cv::Mat::zeros(src_x.rows, order + 1, CV_32FC1);
	cv::Mat copy;
	for (int i = 0; i <= order; i++)
	{
		copy = src_x.clone();
		pow(copy, i, copy);
		cv::Mat M1 = X.col(i);
		copy.col(0).copyTo(M1);
	}
	cv::Mat X_t, X_inv;
	transpose(X, X_t);
	cv::Mat temp = X_t * X;
	cv::Mat temp2;
	invert(temp, temp2);
	cv::Mat temp3 = temp2 * X_t;
	cv::Mat W = temp3 * src_y;
	W.copyTo(dst);
}


template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{
	std::vector<double> linspaced;
	double start = static_cast<double>(start_in);
	double end = static_cast<double>(end_in);
	double num = static_cast<double>(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1) {
		linspaced.push_back(start);
		return linspaced;
	}

	double delta = (end - start) / (num - 1);

	for (int i = 0; i < num - 1; ++i) {
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end);

	return linspaced;
}

void undistort_to_HLS(const cv::Mat& src, cv::Mat& dest, CameraCalibrator& calibrator)
{
	cv::Mat undist_img = calibrator.remap(src);
	cv::cvtColor(undist_img, dest, cv::COLOR_BGR2HLS);

	return;
}

void abs_sobel_thresh(cv::Mat const& src, cv::Mat& dest, char orient = 'x', int kernel_size = 3, int thresh_min = 0, int thresh_max = 255)
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


inline void lane_histogram(cv::Mat const& img, cv::Mat& histogram)
{
	// Histogram 
	cv::Mat cropped = img(cv::Rect(0, img.rows / 2, img.cols, img.rows / 2));
	cv::reduce(cropped / 255, histogram, 0, cv::REDUCE_SUM, CV_32S);

	return;
}

void lane_peaks(cv::Mat const& histogram, cv::Point& left_max_loc, cv::Point& right_max_loc)
{
	// TODO: find a method to handle shadows
	cv::Point temp;
	double min, max;
	int midpoint = histogram.cols / 2;

	cv::Mat left_half = histogram.colRange(0, midpoint);
	cv::Mat right_half = histogram.colRange(midpoint, histogram.cols);

	cv::minMaxLoc(left_half, &min, &max, &temp, &left_max_loc);
	cv::minMaxLoc(right_half, &min, &max, &temp, &right_max_loc);
	right_max_loc = right_max_loc + cv::Point(midpoint, 0);

	return;
}

void calc_warp_points(const cv::Mat& img,
	std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst,
	int y_bottom, int y_top, int offset = 200)
{
	int nX, nY;
	nX = img.cols;
	nY = img.rows;

	// calculate the vertices of the Region of Interest
	src.push_back(cv::Point2f(560, y_top));
	src.push_back(cv::Point2f(695, y_top));
	src.push_back(cv::Point2f(1170, y_bottom));
	src.push_back(cv::Point2f(115, y_bottom));

	// calculate the destination points of the warp
	dst.push_back(cv::Point2f(offset, 0));
	dst.push_back(cv::Point2f(nX - offset, 0));
	dst.push_back(cv::Point2f(nX - offset, nY));
	dst.push_back(cv::Point2f(offset, nY));

	return;
}

inline void draw_lines(cv::Mat& img, const std::vector<cv::Point2f>& vertices)
{
	std::vector<cv::Point> contour(vertices.begin(), vertices.end());

	// create a pointer to the data as an array of points 
	// (via a conversion to a Mat() object)
	const cv::Point* points = (const cv::Point*) cv::Mat(contour).data;
	int npts = cv::Mat(contour).rows;

	// draw the polygon 
	cv::polylines(img, &points, &npts, 1,
		true, 			// draw closed contour (i.e. joint end to start) 
		cv::Scalar(0, 0, 255),// colour RGB ordering (here = RED) 
		2, 		        // line thickness
		cv::LINE_AA, 0);

	imshow("Region of Interest", img);
	return;
}

inline void perspective_transforms(std::vector<cv::Point2f> const& src, std::vector<cv::Point2f>  const& dst,
	cv::Mat& M, cv::Mat& Minv)
{
	M = cv::getPerspectiveTransform(src, dst);
	Minv = cv::getPerspectiveTransform(dst, src);

	return;
}

inline void perspective_warp(const cv::Mat& img, cv::Mat& dst, const cv::Mat& M)
{
	cv::warpPerspective(img, dst, M, img.size(), cv::INTER_LINEAR);
	return;
}

void read_imgs(const std::string& path_to_imgs, std::vector<std::string>& imgs)
{
	for (const auto& entry : std::experimental::filesystem::directory_iterator(path_to_imgs)) {
		std::experimental::filesystem::path path = entry.path();
		imgs.push_back(path.u8string());
	}

	return;
}

void combined_threshold(cv::Mat const& img, cv::Mat& dst)
{
	// convert to HLS color space
	cv::Mat undist_hls;
	cv::cvtColor(img, undist_hls, cv::COLOR_BGR2HLS);

	// split into H,L,S channels
	cv::Mat hls_channels[3];
	cv::split(undist_hls, hls_channels);

	// apply Absolute Sobel Threshold
	cv::Mat sobel_x, sobel_y, combined;
	abs_sobel_thresh(hls_channels[2], sobel_x, 'x', 3, 10, 170);
	abs_sobel_thresh(hls_channels[2], sobel_y, 'y', 3, 10, 170);
	dst = sobel_x & sobel_y; // combine gradient images

	return;
}


void binary_topdown(const cv::Mat& undistorted, cv::Mat& warped,cv::Mat& M,cv::Mat& Minv)
{

	// top down view warp of the undistorted binary image
	int y_bottom = 720;
	int y_top = 425;
	std::vector<cv::Point2f> src, dst;

	calc_warp_points(undistorted, src, dst, y_bottom, y_top);

	// calculate matrix for perspective warp
	perspective_transforms(src, dst, M, Minv);

	// TODO: handle daytime shadow images
	// convert to HLS color space
	cv::Mat combined;
	combined_threshold(undistorted, combined);

	// get a warped image
	perspective_warp(combined, warped, M);
}

void start_calibration(const std::vector<std::string>& imgs, CameraCalibrator& calibrator)
{

	std::cout << "Start Calibration ..." << std::endl;

	cv::Mat image = cv::imread(imgs[0], 0);
	cv::Size imgSize = image.size();
	cv::Size boardCells(9, 6);

	int successes = calibrator.addChessboardPoints(imgs, boardCells);
	double error = calibrator.calibrate(imgSize);
	cv::Mat cameraMatrix = calibrator.getCameraMatrix();
	cv::Mat  distCoeffs = calibrator.getDistCoeffs();

	std::cout << "------------------------ Calibration Log ------------------------" << std::endl;
	std::cout << "Image Size: " << imgSize << std::endl;
	std::cout << "Calibration Error: " << error << std::endl;
	std::cout << "Camera Matrix: " << cameraMatrix << std::endl;
	std::cout << "Dist Matrix: " << distCoeffs << std::endl;
	std::cout << " Success " << successes << std::endl;
	std::cout << "------------------------ end ------------------------" << std::endl;

	return;
}

void find_lane_windows(cv::Mat& binary_img, WindowBox& window_box, std::vector<WindowBox>& wboxes)
{
	bool continue_lane_search = true;
	int contiguous_box_no_line_count = 0;

	// keep searching up the image for a lane lineand append the boxes
	while (continue_lane_search && window_box.y_top > 0) {
		if (window_box.has_line())
			wboxes.push_back(window_box);
		window_box = window_box.get_next_windowbox(binary_img);

		// if we've found the lane and can no longer find a box with a line in it
		// then its no longer worth while searching
		if (window_box.has_lane())
			if (window_box.has_line())
				contiguous_box_no_line_count = 0;
			else {
				contiguous_box_no_line_count += 1;
				if (contiguous_box_no_line_count >= 4)
					continue_lane_search = false;
			}
	}

	return;
}

void calc_lane_windows(cv::Mat& binary_img, int nwindows, int width,
	std::vector<WindowBox>& left_boxes, std::vector<WindowBox>& right_boxes)
{
	// calc height of each window
	int ytop = binary_img.rows;
	int height = ytop / nwindows;

	// find leftand right lane centers to start with
	cv::Mat histogram;
	lane_histogram(binary_img, histogram); // Histogram 

	cv::Point peak_left, peak_right;
	lane_peaks(histogram, peak_left, peak_right); // Peaks

	// Initialise left and right window boxes
	WindowBox wbl(binary_img, peak_left.x, ytop, width, height);
	WindowBox wbr(binary_img, peak_right.x, ytop, width, height);

	// TODO: Parallelize searching

	find_lane_windows(binary_img, wbl, left_boxes);
	find_lane_windows(binary_img, wbr, right_boxes);

	return;
}


void draw_polyline(cv::Mat& out_img, std::vector<double> const& fitx, std::vector<double> const& fity, cv::Scalar& color)
{
	assert(fitx.size() == fity.size());

	std::vector<cv::Point2f> points;
	for (int i = 0; i < fity.size(); i++)
		points.push_back(cv::Point2f(fitx[i], fity[i]));
	cv::Mat curve(points, true);
	curve.convertTo(curve, CV_32S); //adapt type for polylines
	cv::polylines(out_img, curve, false, color, 2);

}

void poly_fitx(std::vector<double> const& fity, std::vector<double>& fitx, cv::Mat const& line_fit)
{
	for (auto const& y : fity) {
		double x = line_fit.at<float>(2, 0) * y * y + line_fit.at<float>(1, 0) * y + line_fit.at<float>(0, 0);
		fitx.push_back(x);
	}

	return;
}

cv::Mat calc_fit_from_boxes(std::vector<WindowBox> const& boxes)
{
	int n = boxes.size();
	std::vector<cv::Mat> xmatrices, ymatrices;
	xmatrices.reserve(n);
	ymatrices.reserve(n);

	cv::Mat xtemp, ytemp;
	for (auto const& box : boxes) {
		// get matpoints
		box.get_indices(xtemp, ytemp);
		xmatrices.push_back(xtemp);
		ymatrices.push_back(ytemp);
	}
	cv::Mat xs, ys;
	cv::vconcat(xmatrices, xs);
	cv::vconcat(ymatrices, ys);

	// Fit a second order polynomial to each
	cv::Mat fit = cv::Mat::zeros(3, 1, CV_32F);
	polyfit(ys, xs, fit, 2);

	return fit;
}

void window_lane(std::vector<cv::Point> const& nonzero, cv::Mat const& poly,
	std::vector<double>& xs, std::vector<double>& ys, int margin)
{
	float left_x, right_x, y, x;
	for (auto const& point : nonzero) {
		y = point.y;
		x = point.x;
		left_x = poly.at<float>(2, 0) * y * y + poly.at<float>(1, 0) * y + poly.at<float>(0, 0) - margin;
		right_x = poly.at<float>(2, 0) * y * y + poly.at<float>(1, 0) * y + poly.at<float>(0, 0) + margin;
		if (x > left_x && x < right_x) {
			xs.push_back(x);
			ys.push_back(y);
		}
	}

	return;
}

void calc_lr_fit_from_polys(cv::Mat& binary_img, cv::Mat const& left_fit, cv::Mat const& right_fit,
	cv::Mat& new_left_fit, cv::Mat& new_right_fit, int margin)
{
	std::vector<cv::Point> nonzero;
	cv::findNonZero(binary_img, nonzero);

	std::vector<double> left_xs, left_ys, right_xs, right_ys;
	window_lane(nonzero, left_fit, left_xs, left_ys, margin);
	window_lane(nonzero, right_fit, right_xs, right_ys, margin);

	new_left_fit = left_fit;
	new_right_fit = right_fit;
	if (!left_fit.empty()) {
		new_left_fit = cv::Mat::zeros(3, 1, CV_32F);
		cv::Mat xs(left_xs, CV_32FC1), ys(left_ys, CV_32FC1);
		polyfit(ys, xs, new_left_fit, 2);
	}
	if (!right_fit.empty()) {
		new_right_fit = cv::Mat::zeros(3, 1, CV_32F);
		cv::Mat xs(right_xs, CV_32FC1), ys(right_ys, CV_32FC1);
		polyfit(ys, xs, new_right_fit, 2);
	}

	return;
}

#endif
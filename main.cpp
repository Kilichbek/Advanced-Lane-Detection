#include <iostream>
#include <filesystem>
#include "calibrator.h"
#include "utils.h"

namespace fs = std::experimental::filesystem;
using namespace std;

class WindowBox{
	int x_left, x_center, x_right;
	int y_bottom, y_top;
	int width, height, mincount;
	bool lane_found;
	cv::Mat img_window;
	vector<cv::Point> nonzero;

	int count_nonzero(void) const { return nonzero.size(); }
	bool is_noise(void) const { return (count_nonzero() > img_window.rows * img_window.cols * .75); }

public:
	WindowBox() : x_left(0), x_center(0), x_right(0), 
		y_bottom(0), y_top(0), 
		width(0), height(0), 
		mincount(0), lane_found(false) {}
	WindowBox(cv::Mat& binary_img, int x_center, int y_top, 
		int width = 220, int height = 80, 
		int mincount = 50, bool lane_found = false);

	inline friend std::ostream& operator<< (std::ostream& out, WindowBox const& window);

	// getters
	void get_centers(int& x_center, int& y_center) const { x_center = this->x_center; y_center = (y_top - y_bottom) / 2; }
	const WindowBox get_next_windowbox(cv::Mat& binary_img) const;

	// hassers 
	bool has_line(void) const { return (count_nonzero() > mincount) ^ is_noise(); }
	bool has_lane(void);
};

inline void lane_histogram(cv::Mat const& img, cv::Mat& histogram)
{
	// Histogram 
	cv::Mat cropped = img(cv::Rect(0, img.rows / 2, img.cols, img.rows / 2));
	cv::reduce(cropped, histogram, 0, cv::REDUCE_SUM, CV_32S);

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
	vector<cv::Point2f>& src, vector<cv::Point2f>& dst,
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

inline void draw_lines(cv::Mat& img, const vector<cv::Point2f> vertices)
{
	vector<cv::Point> contour(vertices.begin(), vertices.end());

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

inline void perspective_transforms(vector<cv::Point2f> const& src, vector<cv::Point2f>  const& dst,
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

void read_imgs(const string& path_to_imgs, vector<string>& imgs)
{
	for (const auto& entry : fs::directory_iterator(path_to_imgs)) {
		fs::path path = entry.path();
		imgs.push_back(path.u8string());
	}

	return;
}

void binary_topdown(const cv::Mat& undistorted, cv::Mat& warped)
{

	// top down view warp of the undistorted binary image
	int y_bottom = 720;
	int y_top = 425;
	vector<cv::Point2f> src, dst;

	calc_warp_points(undistorted, src, dst, y_bottom, y_top);

	// calculate matrix for perspective warp
	cv::Mat M, Minv;
	perspective_transforms(src, dst, M, Minv);

	// TODO: handle daytime shadow images
	// convert to HLS color space
	cv::Mat combined;
	combined_threshold(undistorted, combined);

	// get a warped image
	perspective_warp(combined, warped, M);
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
void start_calibration(const vector<string>& imgs, CameraCalibrator& calibrator)
{

	cout << "Start Calibration ..." << endl;

	cv::Mat image = cv::imread(imgs[0], 0);
	cv::Size imgSize = image.size();
	cv::Size boardCells(9, 6);

	int successes = calibrator.addChessboardPoints(imgs, boardCells);
	double error = calibrator.calibrate(imgSize);
	cv::Mat cameraMatrix = calibrator.getCameraMatrix();
	cv::Mat  distCoeffs = calibrator.getDistCoeffs();

	cout << "------------------------ Calibration Log ------------------------" << endl;
	cout << "Image Size: " << imgSize << endl;
	cout << "Calibration Error: " << error << endl;
	cout << "Camera Matrix: " << cameraMatrix << endl;
	cout << "Dist Matrix: " << distCoeffs << endl;
	cout << " Success " << successes << endl;
	cout << "------------------------ end ------------------------" << endl;

	return;
}

int main()
{
	string path_to_files = "C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\camera_cal";
	vector<string> chessboard_imgs;
	read_imgs(path_to_files, chessboard_imgs);

	CameraCalibrator calibrator;
	start_calibration(chessboard_imgs, calibrator);

	cv::Mat sample_img = cv::imread("C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\test_images\\straight_lines1.jpg");

	// undistort the image
	cv::Mat undistorted = calibrator.remap(sample_img);
	cv::Mat warped;
	binary_topdown(undistorted, warped);

	// Histogram 
	cv::Mat histogram;
	lane_histogram(warped, histogram);

	// create output image
	cv::Mat out_img;
	auto channels = vector<cv::Mat>{ warped,warped,warped };
	cv::merge(channels, out_img);

	// Peaks
	cv::Point leftx_base, rightx_base;
	lane_peaks(histogram, leftx_base, rightx_base);

	// Window  height
	int nwindows = 9;
	int window_height = warped.rows / nwindows;

	vector<cv::Point> nonzero, nonzero_y, nonzero_x;
	cv::findNonZero(warped, nonzero);

	cv::Point leftx_current, rightx_current;
	leftx_current = leftx_base;
	rightx_current = rightx_base;

	int margin = 110, minpix = 50;
	int win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high;

	vector<cv::Point> good_left_inds, good_right_inds, left_lane_inds, right_lane_inds;

	for (int window = 0; window < nwindows; window++) {

		// Identify window boundaries in x and y (and right and left)
		win_y_low = warped.rows - (window + 1) * window_height;
		win_y_high = warped.rows - window * window_height;
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

		cv::Mat win_left = warped(rect_left);
		cv::Mat win_right = warped(rect_right);

		cv::findNonZero(win_left, good_left_inds);
		cv::findNonZero(win_right, good_right_inds);

		// Append these indices to the vector
		// reserve() is optional - just to improve performance

		left_lane_inds.reserve(left_lane_inds.size() + good_left_inds.size());
		vector<cv::Point>::iterator start1 = good_left_inds.begin(), stop1 = good_left_inds.end();
		while (start1 != stop1) {
			cv::Point point = *start1;
			left_lane_inds.push_back(point + cv::Point(win_xleft_low, win_y_low));
			start1++;
		}

		right_lane_inds.reserve(right_lane_inds.size() + good_right_inds.size());
		vector<cv::Point>::iterator start2 = good_right_inds.begin(), stop2 = good_right_inds.end();
		while (start2 != stop2) {
			cv::Point point = *start2;
			right_lane_inds.push_back(point + cv::Point(win_xright_low, win_y_low));
			start2++;
		}

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
		if ((good_right_inds.size() > minpix) || (good_right_inds.size() > (win_right.cols * win_right.rows) * 0.75)) {
			double sum = 0;

			for (int i = 0; i < good_right_inds.size(); i++) {
				sum += (good_right_inds[i].x + win_xright_low);

			}
			int mean = sum / good_right_inds.size();
			rightx_current.x = mean;

		}
	}
	// Extract left and right line pixels
	cv::Mat leftx = cv::Mat::zeros(left_lane_inds.size(), 1, CV_32F);
	cv::Mat lefty = cv::Mat::zeros(left_lane_inds.size(), 1, CV_32F);

	for (int i = 0; i < left_lane_inds.size(); i++) {
		leftx.at<float>(i, 0) = left_lane_inds[i].x;
		lefty.at<float>(i, 0) = left_lane_inds[i].y;
	}

	cv::Mat rightx = cv::Mat::zeros(right_lane_inds.size(), 1, CV_32F);
	cv::Mat righty = cv::Mat::zeros(right_lane_inds.size(), 1, CV_32F);

	for (int i = 0; i < right_lane_inds.size(); i++) {
		rightx.at<float>(i, 0) = right_lane_inds[i].x;
		righty.at<float>(i, 0) = right_lane_inds[i].y;
	}

	// Fit a second order polynomial to each
	cv::Mat left_fit = cv::Mat::zeros(3, 1, CV_32F);
	polyfit(lefty, leftx, left_fit, 2);
	
	cv::Mat right_fit = cv::Mat::zeros(3, 1, CV_32F);
	polyfit(righty, rightx, right_fit, 2);

	// Define conversions in xand y from pixels space to meters
	float ym_per_pix = 30.0 / 720; // meters per pixel in y dimension
	float xm_per_pix = 3.7 / 700; // meters per pixel in x dimension

	// Fit a second order polynomial to each
	cv::Mat left_fit_m = cv::Mat::zeros(3, 1, CV_32F);
	cv::Mat right_fit_m = cv::Mat::zeros(3, 1, CV_32F);
	polyfit(lefty * ym_per_pix, leftx * xm_per_pix, left_fit_m, 2);
	polyfit(righty * ym_per_pix, rightx * xm_per_pix, right_fit_m, 2);

	vector<cv::Point2f> left_fitx, right_fitx;
	vector<double> ploty = linspace<double>(0, out_img.rows - 1, out_img.rows);
	vector<double>::iterator iter = ploty.begin(), end = ploty.end();
	while (iter != end) {
		double y = *iter;
		double x = left_fit.at<float>(2, 0) * y * y + left_fit.at<float>(1, 0) * y + left_fit.at<float>(0, 0);
		left_fitx.push_back(cv::Point2f(x, y));

		x = right_fit.at<float>(2, 0) * y * y + right_fit.at<float>(1, 0) * y + right_fit.at<float>(0, 0);
		right_fitx.push_back(cv::Point2f(x, y));

		iter++;
	}

	cv::Mat left_curve(left_fitx, true), right_curve(right_fitx, true);
	left_curve.convertTo(left_curve, CV_32S); //adapt type for polylines
	right_curve.convertTo(right_curve, CV_32S);
	polylines(out_img, left_curve, false, cv::Scalar(0, 0, 255), 2);
	polylines(out_img, right_curve, false, cv::Scalar(255, 0, 0), 2);


	imshow("Output", out_img);
	cv::waitKey(0);

	return 0;
}

WindowBox::WindowBox(cv::Mat& binary_img, int x_center, int y_top, int width, int height, int mincount, bool lane_found)
{
	this->x_center = x_center;
	this->y_top = y_top;
	this->width = width;
	this->height = height;
	this->mincount = mincount;
	this->lane_found = lane_found;
	
	// derived 	
	// identify window boundaries in x and y
	int margin = this->width / 2;
	x_left = this->x_center - margin;
	x_right = this->x_center + margin;
	y_bottom = y_top - this->height;

	// Identify the nonzero pixels in x and y within the window
	cv::Rect rect(cv::Point(x_left, y_bottom), cv::Point(x_right, y_top));
	cv::Mat img_window = binary_img(rect);
	cv::findNonZero(img_window, nonzero);
}

const WindowBox WindowBox::get_next_windowbox(cv::Mat& binary_img) const
{
	int new_y_top = y_bottom; // next box top starts at lasts bottom
	int new_x_center = x_center; // use existing center

	if (has_line()) {
		// recenter based on mean
		double sum = 0;
		for (auto const& point : nonzero) {
			sum += (point.x + x_left);
		}
		new_x_center = sum / nonzero.size();
	}
	
	return WindowBox(binary_img, new_x_center, new_y_top,
		this->width, this->height, 
		this->mincount, this->lane_found);
}



bool WindowBox::has_lane(void)
{
	if (!lane_found && has_line()) lane_found = true;
	return lane_found;
}

std::ostream& operator<<(std::ostream& out, WindowBox const& window)
{
	out << "Window Box [" << window.x_left << ", " << window.y_bottom << ", ";
	out << window.x_right << ", " << window.y_top << "]" << endl;

	return out;
}

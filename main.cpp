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
	friend void find_lane_windows(cv::Mat& binary_img, WindowBox& window_box, vector<WindowBox>& wboxes);
	friend void calc_lane_windows(cv::Mat& binary_img, int nwindows, int width);

	// getters
	void get_centers(int& x_center, int& y_center) const { x_center = this->x_center; y_center = (y_top - y_bottom) / 2; }
	void get_indices(cv::Mat& x, cv::Mat& y) const;
	const WindowBox get_next_windowbox(cv::Mat& binary_img) const;

	// hassers 
	bool has_line(void) const { return ((count_nonzero() > mincount) || is_noise()); }
	bool has_lane(void);
};

void combined_threshold(cv::Mat const& img, cv::Mat& dst);
cv::Mat calc_fit_from_boxes(vector<WindowBox> const& boxes);
void poly_fitx(vector<double> const& fity, vector<double>& fitx, cv::Mat const& line_fit);
void draw_polyline(cv::Mat& out_img, vector<double> const& fitx, vector<double> const& fity);

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

void find_lane_windows(cv::Mat& binary_img, WindowBox& window_box, vector<WindowBox>& wboxes)
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

void calc_lane_windows(cv::Mat& binary_img, int nwindows, int width)
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
	vector<WindowBox> left_boxes, right_boxes;
	find_lane_windows(binary_img, wbl, left_boxes);
	find_lane_windows(binary_img, wbr, right_boxes);


	// create output image
	cv::Mat out_img;
	auto channels = vector<cv::Mat>{ binary_img,binary_img,binary_img };
	cv::merge(channels, out_img);

	// Draw the windows on the visualization image
	for (const auto& box : left_boxes) {
		cv::Point pnt1(box.x_left, box.y_bottom), pnt2(box.x_right, box.y_top);
		cv::rectangle(out_img, pnt1, pnt2, cv::Scalar(0, 255, 0), 2);
	}

	for (const auto& box : right_boxes) {
		cv::Point pnt1(box.x_left, box.y_bottom), pnt2(box.x_right, box.y_top);
		cv::rectangle(out_img, pnt1, pnt2, cv::Scalar(0, 255, 0), 2);
	}

	cv::Mat left_fit = calc_fit_from_boxes(left_boxes);
	cv::Mat right_fit = calc_fit_from_boxes(right_boxes);
	vector<double> fity, left_fitx, right_fitx;
	fity = linspace<double>(0, binary_img.rows - 1, binary_img.rows);
	poly_fitx(fity, left_fitx, left_fit);
	poly_fitx(fity, right_fitx, right_fit);

	draw_polyline(out_img, left_fitx, fity);
	draw_polyline(out_img, right_fitx, fity);


	return;
}


void draw_polyline(cv::Mat& out_img,vector<double> const& fitx, vector<double> const& fity)
{
	assert(fitx.size() == fity.size());

	vector<cv::Point2f> points;
	for (int i = 0; i < fity.size(); i++)
		points.push_back(cv::Point2f(fitx[i], fity[i]));
	cv::Mat curve(points, true);
	curve.convertTo(curve, CV_32S); //adapt type for polylines
	cv::polylines(out_img, curve, false, cv::Scalar(0, 0, 255), 2);

	imshow("Rectangle", out_img);
}

void poly_fitx(vector<double> const& fity, vector<double>& fitx, cv::Mat const& line_fit)
{
	for (auto const& y : fity) {
		double x = line_fit.at<float>(2, 0) * y * y + line_fit.at<float>(1, 0) * y + line_fit.at<float>(0, 0);
		fitx.push_back(x);
	}

	return;
}

cv::Mat calc_fit_from_boxes(vector<WindowBox> const& boxes)
{
	int n = boxes.size();
	vector<cv::Mat> xmatrices, ymatrices;
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

void window_polyfit(cv::Mat& binary_img)
{
	vector<cv::Point> nonzero;
	cv::findNonZero(binary_img, nonzero);
	for (auto const& point : nonzero) {

	}
}
int main()
{
	string path_to_files = "C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\camera_cal";
	vector<string> chessboard_imgs;
	read_imgs(path_to_files, chessboard_imgs);

	CameraCalibrator calibrator;
	start_calibration(chessboard_imgs, calibrator);

	cv::Mat sample_img = cv::imread("C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\test_images\\test4.jpg");

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
	calc_lane_windows(warped, nwindows,width);

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

void WindowBox::get_indices(cv::Mat& x, cv::Mat& y) const
{
	// clear matrices
	x.release();
	y.release();

	int npoints = count_nonzero();
	x = cv::Mat::zeros(npoints, 1, CV_32F);
	y = cv::Mat::zeros(npoints, 1, CV_32F);

	for (int i = 0; i < npoints; i++) {
		x.at<float>(i, 0) = nonzero[i].x + x_left;
		y.at<float>(i, 0) = nonzero[i].y + y_bottom;
	}
	
	return;
}

const WindowBox WindowBox::get_next_windowbox(cv::Mat& binary_img) const
{
	if (y_bottom <= 0) return WindowBox(); // return empty box

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
	if (new_x_center + this->width / 2 > binary_img.cols) return WindowBox();
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

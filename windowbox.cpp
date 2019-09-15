#include "windowbox.h"

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
		
		double sum = 0;
		for (auto const& point : nonzero) {
			sum += (point.x + x_left);
		}
		new_x_center = sum / nonzero.size(); // recenter based on mean
	}
	if (new_x_center + this->width / 2 > binary_img.cols) return WindowBox(); // if outside of ROI return empty box

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
	out << window.x_right << ", " << window.y_top << "]" << std::endl;

	return out;
}

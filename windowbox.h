#ifndef __WINDOWBOX_H__
#define __WINDOWBOX_H__

#include <vector>
#include <ostream>
#include <opencv2/core.hpp>

class WindowBox {
	int x_left, x_center, x_right;
	int y_bottom, y_top;
	int width, height, mincount;
	bool lane_found;
	cv::Mat img_window;
	std::vector<cv::Point> nonzero;

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
	friend void find_lane_windows(cv::Mat& binary_img, WindowBox& window_box, std::vector<WindowBox>& wboxes);


	// getters
	void get_centers(int& x_center, int& y_center) const { x_center = this->x_center; y_center = (y_top - y_bottom) / 2; }
	void get_indices(cv::Mat& x, cv::Mat& y) const;
	const cv::Point get_bottom_left_point(void) const { return cv::Point(x_left, y_bottom); }
	const cv::Point get_top_right_point(void) const { return cv::Point(x_right, y_top); }
	const WindowBox get_next_windowbox(cv::Mat& binary_img) const;

	// hassers 
	bool has_line(void) const { return ((count_nonzero() > mincount) || is_noise()); }
	bool has_lane(void);
};

#endif
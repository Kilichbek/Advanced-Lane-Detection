#ifndef __CALIBRATOR_H__
#define __CALIBRATOR_H__

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d/calib3d_c.h>

class CameraCalibrator {
	//input points;
	//the points in world coordinates
	std::vector<std::vector<cv::Point3f>> object_points;
	// the image point positions in pixels
	std::vector<std::vector<cv::Point2f>> image_points;
	//output matrices
	cv::Mat camera_matrix, dist_coeffs;
	// used in image undistortion 
	cv::Mat map1, map2;

	// flag to specify how calibration is done
	int flag;
	bool must_init_undistort;

	void addPoints(std::vector<cv::Point2f>& img_corners, std::vector<cv::Point3f>& obj_corners);

public:
	CameraCalibrator() : flag(0), must_init_undistort(true) {};

	int add_chessboard_points(const std::vector<std::string>& filelist,
		cv::Size& board_size, bool visualize = false);
	double calibrate(cv::Size& image_size);
	void set_calibration_flag(bool radial8_coeff_enabled = false, bool tangential_param_enabled= false);
	// Remove distortion in an image (after calibration)
	cv::Mat remap(const cv::Mat& image);
	void show_undist_images(const std::vector<std::string>& filelist);
	// Getters
	cv::Mat get_camera_matrix() { return camera_matrix; }
	cv::Mat get_dist_coeffs() { return dist_coeffs; }
	void save_as(std::string const& filename) const;
	void load_settings(std::string const& filename);
};

#endif
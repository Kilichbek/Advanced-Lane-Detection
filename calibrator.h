#ifndef __CALIBRATOR_H__
#define __CALIBRATOR_H__

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/imgproc/types_c.h>

class CameraCalibrator {
	//input points;
	//the points in world coordinates
	std::vector<std::vector<cv::Point3f>> objectPoints;
	// the image pint positions in pixels
	std::vector<std::vector<cv::Point2f>> imagePoints;
	//output matrices
	cv::Mat cameraMatrix, distCoeffs;
	// used in image undistortion 
	cv::Mat map1, map2;

	// flag to specify how calibration is done
	int flag;
	bool mustInitUndistort;

	void addPoints(std::vector<cv::Point2f>& imgCorners, std::vector<cv::Point3f>& objCorners);

public:
	CameraCalibrator() : flag(0), mustInitUndistort(true) {};
	int addChessboardPoints(const std::vector<std::string>& filelist,
		cv::Size& boardSize, bool visualize = false);
	double calibrate(cv::Size& imageSize);
	void setCalibrationFlag(bool radial8CoeffEnabled = false, bool tangentialParamEnabled = false);
	// Remove distortion in an image (after calibration)
	cv::Mat remap(const cv::Mat& image);
	void showUndistortedImages(const std::vector<std::string>& filelist);
	// Getters
	cv::Mat getCameraMatrix() { return cameraMatrix; }
	cv::Mat getDistCoeffs() { return distCoeffs; }
};

#endif
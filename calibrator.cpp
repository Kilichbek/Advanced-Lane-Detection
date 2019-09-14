#include "calibrator.h"

void CameraCalibrator::addPoints(std::vector<cv::Point2f>& img_corners, std::vector<cv::Point3f>& obj_corners)
{
	image_points.push_back(img_corners);
	object_points.push_back(obj_corners);

	return;
}

int CameraCalibrator::add_chessboard_points(const std::vector<std::string>& filelist,
	cv::Size& board_size, bool visualize) {
	// the points on the chessboard
	std::vector<cv::Point2f> image_corners;
	std::vector<cv::Point3f> object_corners;

	// 3D Scene Points:
	// Initialize the chessboard corners
	// in the chessboard reference frame
	// The corners are at 3D location (X,Y,Z)= (i,j,0)
	for (int i = 0; i < board_size.height; i++)
		for (int j = 0; j < board_size.width; j++)
			object_corners.push_back(cv::Point3f(i, j, 0.0f));

	// 2D Image Points:
	cv::Mat image; // to contain chessboard image
	cv::Mat gray;
	int successes = 0;
	
	// for all viewpoints
	for (int i = 0; i < filelist.size(); i++) {
		// open the image
		image = cv::imread(filelist[i]);
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

		// get the chessboard corners
		bool found = cv::findChessboardCorners(image, board_size, image_corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		// get subpixel accuracy on the corners
		if (found) {
			cv::cornerSubPix(gray, image_corners, cv::Size(5, 5),// half size of search window
				cv::Size(-1, -1),
				cv::TermCriteria(cv::TermCriteria::MAX_ITER +
					cv::TermCriteria::EPS, 30,// max number of iterations
					0.1)); //min accuracy
		}

		// if we have a good board, add it to our data
		if (image_corners.size() == board_size.area()) {
			// add image and scene points from one view
			addPoints(image_corners, object_corners);
			successes++;
		}

		//Draw the corners
		if (visualize) {
			cv::drawChessboardCorners(gray, board_size, image_corners, found);
			cv::imshow("Corners on Chessboard", image);
			cv::waitKey(1000);
		}
	}

	return successes;
}

double CameraCalibrator::calibrate(cv::Size& image_size)
{
	// Calibrate the camera
	// return the re-projection error
	// output rotations and translations
	std::vector<cv::Mat> rvecs, tvecs;

	// start calibration
	return cv::calibrateCamera(object_points,
		image_points,
		image_size,
		camera_matrix,
		dist_coeffs,
		rvecs, tvecs,
		flag);
}

// Set the calibration options
// 8radialCoeffEnabled should be true if 8 radial coefficients are required (5 is default)
// tangential_param_enabledshould be true if tangeantial distortion is present
void CameraCalibrator::set_calibration_flag(bool radial8_coeff_enabled, bool tangentialParamEnabled) {

	// Set the flag used in cv::calibrateCamera()
	flag = 0;
	if (!tangentialParamEnabled) flag += CV_CALIB_ZERO_TANGENT_DIST;
	if (radial8_coeff_enabled) flag += CV_CALIB_RATIONAL_MODEL;
}

cv::Mat CameraCalibrator::remap(const cv::Mat& image)
{
	cv::Mat undistorted;
	if (must_init_undistort) { // called once per calibration
		cv::initUndistortRectifyMap(
			camera_matrix, // computed camera matrix
			dist_coeffs, // computed distortion matrix
			cv::Mat(), // optional rectification (none)
			cv::Mat(), // camera matrix to generate undistorted
			image.size(), // size of undistorted
			CV_32FC1, // type of output map
			map1, map2); // the x and y mapping functions
		must_init_undistort = false;
	}
	// Apply mapping functions
	cv::remap(image, undistorted, map1, map2,
		cv::INTER_LINEAR); // interpolation type
	return undistorted;
}

void CameraCalibrator::show_undist_images(const std::vector<std::string>& filelist)
{
	cv::Mat image, undist_image;

	for (int i = 0; i < filelist.size(); i++) {
		// open the image
		image = cv::imread(filelist[i]);
		undist_image = remap(image);
		cv::imshow("Undistorted Image", undist_image);
		cv::waitKey(15000);
	}

	return;
}

void CameraCalibrator::save_as(std::string const& filename) const
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	
	fs << "camera_matrix" << camera_matrix << "dist_coeffs" << dist_coeffs;
	fs << "map1" << map1 << "map2" << map2;
	fs << "must_init_undistort" << must_init_undistort;

	fs.release();

	return;
}

void CameraCalibrator::load_settings(std::string const& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);

	fs["camera_matrix"] >> camera_matrix;
	fs["dist_coeffs"] >> dist_coeffs;
	fs["map1"] >> map1;
	fs["map2"] >> map2;
	fs["must_init_undistort"] >> must_init_undistort;

	fs.release();
	return;
}

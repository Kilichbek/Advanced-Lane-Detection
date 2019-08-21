#include <iostream>
#include <filesystem>
#include "calibrator.h"


namespace fs = std::experimental::filesystem;
using namespace std;

int main()
{
	string path_to_files = "C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\camera_cal";
	vector<string> files;

	CameraCalibrator camCalibrator;
	

	cout << "Start Calibration ..." << endl;
	for (const auto& entry : fs::directory_iterator(path_to_files)) {
		fs::path path = entry.path();
		files.push_back(path.u8string());
	}

	cv::Mat image = cv::imread(files[0], 0);
	cv::Size imgSize = image.size();
	cv::Size boardCells(9, 6);

	int successes = camCalibrator.addChessboardPoints(files, boardCells);
	double error = camCalibrator.calibrate(imgSize);
	cv::Mat cameraMatrix = camCalibrator.getCameraMatrix();
	cv::Mat  distCoeffs = camCalibrator.getDistCoeffs();

	cout << "------------------------ Calibration Log ------------------------" << endl;
	cout << "Image Size: " << imgSize << endl;
	cout << "Calibration Error: " << error << endl;
	cout << "Camera Matrix: " << cameraMatrix << endl;
	cout << "Dist Matrix: " << distCoeffs << endl;
	cout << " Success " << successes << endl;
	cout << "------------------------ end ------------------------" << endl;

	//camCalibrator.showUndistortedImages(files);

	string path_to_imgs = "C:\\Users\\PC\\Documents\\CarND-Advanced-Lane-Lines-P4-master\\test_images";
	vector<string> images;
	
	bool first = true;
	for (const auto& ent : fs::directory_iterator(path_to_imgs)) {
		fs::path pathe = ent.path();
		if (!first) images.push_back(pathe.u8string());
		first = false;
	}

	cv::Mat sample_img = cv::imread(images[0]);
	cv::Mat win_mat(cv::Size(sample_img.size().width*2, sample_img.size().height), CV_8UC3);
	cv::Mat undist_img = camCalibrator.remap(sample_img);
	cv::Mat dest,undistHSV;
	// hconcat(sample_img,undist_img, dest);
	// Display big mat
	//cv::cvtColor(undist_img, undistHSV, CV_BGR2RGB);
	cv::imshow("Images",undist_img);
	cv::waitKey(0);
	return 0;
}
#include <iostream>
#include <filesystem>
#include <string>
#include "calibrator.h"
#include "utils.h"
#include "windowbox.h"
#include <stdlib.h>

namespace fs = std::experimental::filesystem;

// Window parameters
#define N_WINDOWS 9
#define WINDOW_WIDTH 100

int main()
{
	CameraCalibrator calibrator;
	std::string filename = "camera.yml";
	
	bool exists = std::experimental::filesystem::exists(filename);

	if (exists) {
		calibrator.load_settings(filename);
	}
	else {
		std::string path_to_files = "camera_cal";
		std::vector<std::string> chessboard_imgs;
		read_imgs(path_to_files, chessboard_imgs);
		start_calibration(chessboard_imgs, calibrator);
		calibrator.save_as(filename);
	}

	cv::Mat sample_img = cv::imread("faulty.jpg");
	//imshow("faulty", sample_img);
	cv::Mat un = calibrator.remap(sample_img);

	cv::VideoCapture cap("project_video.mp4");
	cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, un.size());
	
	if (!cap.isOpened()) { // Check if camera opened successfully
		std::cout << "Error opening video stream or file" << std::endl;
		exit(EXIT_FAILURE);
	}

	cv::Mat warped, Minv, M;
	int nframe = 0;
	int height = un.rows;
	cv::Scalar red(0, 0, 255), blue(255, 0, 0), yellow(153, 255, 255);
	cv::Rect r1(cv::Point(0, 0), cv::Point(320, 180));
	cv::Rect r2(cv::Point(320, 0), cv::Point(640, 180));
	cv::Rect r3(cv::Point(640, 0), cv::Point(960, 180));
	cv::Rect r4(cv::Point(960, 0), cv::Point(1280, 180));

	while (1) {
		
		cv::Mat frame;
		cap >> frame; // Capture frame-by-frame
		if (frame.empty()) break; // If the frame is empty, break immediately

		nframe++;

		system("cls");
		std::cout <<"Processing frame #: "<< nframe << std::endl;
		
		if (nframe==763) imwrite("faulty.jpg", frame);

		// undistort the image
		cv::Mat undistorted = calibrator.remap(frame);
		binary_topdown(undistorted, warped, M, Minv);


		// Histogram 
		cv::Mat histogram;
		lane_histogram(warped, histogram);

		// Peaks
		cv::Point leftx_base, rightx_base;
		lane_peaks(histogram, leftx_base, rightx_base);

		std::vector<WindowBox> left_boxes, right_boxes;
		calc_lane_windows(warped, N_WINDOWS, WINDOW_WIDTH, left_boxes, right_boxes);

		cv::Mat left_fit = calc_fit_from_boxes(left_boxes);
		cv::Mat right_fit = calc_fit_from_boxes(right_boxes);

		// generate x and values for plotting
		std::vector<double> fitx, fity, left_fitx, right_fitx, hist_fity, new_left_fitx, new_right_fitx;
		fity = linspace<double>(0, warped.rows - 1, warped.rows);
		fitx = linspace<double>(0, warped.cols - 1, warped.cols);

		for (int i = 0; i < histogram.cols; i++)
			hist_fity.push_back(height - histogram.at<int>(0, i));

		poly_fitx(fity, left_fitx, left_fit);
		poly_fitx(fity, right_fitx, right_fit);

		cv::Mat warp_back = cv::Mat::zeros(undistorted.size(), CV_8UC3);

		draw_line(undistorted, left_fit, right_fit, Minv, warp_back);

		//-----------------------------------------------------------
		// create output image
		cv::Mat out_img;
		auto channels = std::vector<cv::Mat>{ warped,warped,warped };
		cv::merge(channels, out_img);

		draw_boxes(out_img, left_boxes);
		draw_boxes(out_img, right_boxes);

		// draw polynomial curve
		draw_polyline(out_img, left_fitx, fity, blue);
		draw_polyline(out_img, right_fitx, fity, blue);
		

		//hls[1]
		cv::Mat hls[3], dst;
		cv::cvtColor(undistorted, dst, cv::COLOR_BGR2HLS);
		cv::split(dst, hls);
		place_img(hls[2],warp_back, r1);

		//binary  combined
		combined_threshold(undistorted, dst);
		place_img(dst, warp_back, r2);

		//windows
		place_img(out_img, warp_back, r3);

		//hisotgram
		cv::Mat black_img(undistorted.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		draw_polyline(black_img, fitx, hist_fity, red);
		place_img(black_img, warp_back, r4);

		// Write the frame into the file 'outcpp.avi'
		video.write(warp_back);
	}


	// When everything done, release the video capture object
	cap.release();
	video.release();

	return 0;
}
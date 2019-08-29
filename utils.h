#ifndef __UTILS_H__
#define __UTILS_H__

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

void undistToHLS(const cv::Mat& src, cv::Mat& dest, CameraCalibrator& calibrator)
{
	cv::Mat undist_img = calibrator.remap(src);
	cv::cvtColor(undist_img, dest, cv::COLOR_BGR2HLS);

	return;
}

void absSobelThresh(cv::Mat& src, cv::Mat& dest, char orient = 'x', int kernel_size = 3, int thresh_min = 0, int thresh_max = 255)
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
#endif
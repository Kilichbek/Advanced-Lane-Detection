# Advanced Lane Detection 
### Udacity Self-Driving Cars ( Nanodegree )
This repository contains the code that can detect lane lines on the road. 
Advanced computer vision techniques are used in this project.

![sample lane detection result](https://raw.githubusercontent.com/hortovanyi/udacity-advanced-lane-finding-project/master/output_images/writeup_intro_road.gif)

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Written in C++ with OpenCV library
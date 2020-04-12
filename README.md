[//]: # (Image References)
[image1]: ./test_images_output/solidWhiteRight.jpg

# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![lane lines overlaid][image1]

Overview
---

This project deals with a practical problem: given monovision images and videos from a forward looking car camera, infer the bounderies of the car's lane. Lane bounderies are clearly an important reference feature for correct (and safe) driving.

In this first effort, we make heavy use of [OpenCV](https://opencv.org/) and apply classical computer vision methods (Canny edge detection, Hough transforms...). For a more detailed account, read the [write-up](writeup.md).

A full lane detection pipeline is demonstrated in the Jupyter notebook P1.ipynb. The Python script sequence.py may also be of interest - it creates intermediate images for each step in the pipeline, allowing easier visualization. The lr_lane_detect.py script applies the whole lane line detection pipeline to video input.

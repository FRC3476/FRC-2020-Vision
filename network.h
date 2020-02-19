#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

struct exp_data {
	cv::Point2d centroid; 
};

int sendUDP(std::vector<exp_data> d);

void setupUDP();

bool getExposure();

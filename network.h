#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

struct exp_data {
	cv::Point2d centroid; 
};

//typedef struct exp_data vis_data;

int sendUDP(exp_data d);

void setupUDP();

bool getExposure();

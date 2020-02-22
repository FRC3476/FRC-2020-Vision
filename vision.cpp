#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <math.h>
#include "log.h"
#include "network.h"
#include <unistd.h>
#include <time.h>
#include <chrono>
#include <numeric>


#define USE_GSTREAMER 0

#define HIGH_EXP 0.03
#define LOW_EXP 0.001

#define DEBUG

using namespace cv;
using namespace std;

cv::VideoCapture stream; 
cv::VideoWriter writer; 

bool curExpHigh = false;
double fpsA[5] = {0, 0, 0, 0, 0};

int c = 0;
int prevSwitchC = 0;
	
Mat frame;

//Return the min integer in vector
int minElem(vector<int> v) {
	if(v.size() < 1) return 0;
	int m = v[0];
	for(int i = 0; i < v.size(); i++) {
		m = min(v[i], m);
	}
	return m;

}

//Return the max integer in vector
int maxElem(vector<int> v) {
	if(v.size() < 1) return 0;
	int m = v[0];
	for(int i = 0; i < v.size(); i++) {
		m = max(v[i], m);
	}
	return m;

}

//Return the magnitude of a vector
float magnitude(Point2d p) {
	return sqrt(p.x * p.x + p.y * p.y);

}

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

//line intersection 
bool findIntersection(Vec4f line, Vec4f line2 , Point2f &returnPoint ) {
	return intersection(Point(line[2]+line[0]*1000, line[3]+line[1]*1000), Point(line[2], line[3]), Point(line2[2]+line2[0]*1000, line2[3]+line2[1]*1000), Point(line2[2], line2[3] ), returnPoint);
}


//Setup camera object, image properties, and gstreamer streams
void setupCam(VideoCapture stream) {
	#if USE_GSTREAMER
		if(!stream.open("v4l2src device=/dev/v4l/by-path/platform-tegra-xhci-usb-0:3.3:1.0-video-index0 ! image/jpeg, width=640, height=480 ! jpegparse ! jpegdec ! videoconvert ! appsink")) return 0;
	#else
		if(!stream.open("/dev/v4l/by-path/platform-tegra-xhci-usb-0:3.3:1.0-video-index0")) return 0;
	#endif
	stream.set(CAP_PROP_FRAME_WIDTH, 640);
    stream.set(CAP_PROP_FRAME_HEIGHT,480);
    stream.set(CAP_PROP_FPS, 60);
	stream.set(CAP_PROP_BRIGHTNESS, 80.0/256.0);
	stream.set(CAP_PROP_CONTRAST, 25.0/256.0);
	stream.set(CAP_PROP_SATURATION, 60.0/256.0);
	stream.set(CAP_PROP_EXPOSURE, 12/10000.0);
	
	writer.open("appsrc ! autovideoconvert ! video/x-raw, width=640, height=480 ! omxh264enc control-rate=2 bitrate=125000 ! video/x-h264, stream-format=byte-stream ! h264parse ! rtph264pay mtu=1400 ! udpsink host=127.0.0.1 clients=10.34.76.5:5800 port=5800 sync=false async=false ", 0, (double) 5, cv::Size(640, 480), true);

}

//Use to throw away initial frames
void wasteFramesAndDelay(int n) {
	usleep(1000000);
	for(int i = 0; i < n; i++) {
		Mat frame;
		stream >> frame;
	}
}

//Update camera exposure mode (driver vision v robot) 
void updateExposure() {
		bool expStateHigh = getExposure();
		if(expStateHigh != curExpHigh) {
			curExpHigh = expStateHigh;
			if(expStateHigh) stream.set(CAP_PROP_EXPOSURE, HIGH_EXP);
			else {
				stream.set(CAP_PROP_EXPOSURE, LOW_EXP);
				prevSwitchC = c;
			}
		}

}

//count FPS and display on image
void updateFPS(std::chrono::microseconds cur, std::chrono::microseconds prevTime, Mat &drawing) {
		c+=1;
		double deltaT = ((double)std::chrono::duration_cast<std::chrono::microseconds>(cur-prevTime).count()/1e6);
		fpsA[c%5] = 1.0/deltaT;
		double fps = 0;
		for(int i = 0; i < 5; i++) fps+=fpsA[i];
		fps /= 5;
		
		if(c%20>10) circle(drawing, Point(10, 10), 3, COLOR_RED, -1);
		char fpsStr[5];
		sprintf(fpsStr, "%.0f", fps);
		putText(drawing, fpsStr, Point(590, 10), FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2, LINE_AA);
}

void drawLine(Mat m, Vec4f line) {
	line(m, Point(leftLine[2]-leftLine[0] * 1000, leftLine[3]-leftLine[1] * 1000), 
		Point(leftLine[2]+leftLine[0] * 1000, leftLine[3]+leftLine[1] * 1000), Scalar(128));
}

int main(int argc, char** argv ) {
	
	setupCam(stream);
	setupUDP();
	wasteFramesAndDelay();
	
	Mat kernel;
	kernel = cv::getStructuringElement(MORPH_CROSS, Size(3,3));
	
	//main vision loop
	while(1) { 
		//keep track of times, request camera frame
		if( (cv::waitKey(1) & 0xFF) == ' ');
		auto cur = std::chrono::high_resolution_clock::now();
		auto prevTime = cur;
		if(!stream.isOpened()) return -1;
				
				
		bool valid = true;
		Mat frame, colorFilter, fbw, hsvFrame;
		stream >> frame;
		//Threshold image and remove stray pixels
		cv::inRange(hsvFrame, Scalar(35,40,50), Scalar(200, 255, 255), fbw);
		cv::morphologyEx(fbw, fbw, MORPH_OPEN, kernel); 
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		//Detect objects, choose the largest
		findContours(fbw, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0,0) );
		vector<vector<Point> > hulls( contours.size() ); 
		for(int i = 0; i < contours.size(); i++) {
			convexHull(contours[i], hulls[i]);
		}
		int maxIndex = -1;
		double oldMax = -1;
		for(int i = 0; i < hulls.size(); i++) {
			Moments m = moments(hulls[i], true);
			if(m.m00 > oldMax) {
				oldMax = m.m00;
				maxIndex = i;		
			}
		}
		//Request next frame if no viable targets
		if(maxIndex == -1 || contours.size() < 1 || contours[maxIndex].size() < 1) continue;
		
		Point2d centroid = Point2d(hulls[maxIndex].m10/hulls[maxIndex].m00, hulls[maxIndex].m01/hulls[maxIndex].m00);
		vector<vector<Point>> collection; 
		//Store points in x-->y lookup and y-->x lookup, get object bounding points 
		int maxY, minY, maxX, minX;
		maxY = minY = contours[maxIndex][0].y;
		maxX = minX = contours[maxIndex][0].x;
		std::map<int,vector<int>> ytoxMap;
		std::map<int,vector<int>> xtoyMap;
		for(int i = 0; i < contours[maxIndex].size(); i++) {
			ytoxMap[(contours[maxIndex][i].y) ].push_back(contours[maxIndex][i].x);
			xtoyMap[contours[maxIndex][i].x].push_back(contours[maxIndex][i].y);
			maxY = max(maxY, contours[maxIndex][i].y);
			minY = min(minY, contours[maxIndex][i].y);
			maxX = max(maxX, contours[maxIndex][i].x);
			minX = min(minX, contours[maxIndex][i].x);
		}
		
		int searchStartY = (int) (minY+(maxY-minY)/3);
		int searchEndY = (int) (maxY-(maxY-minY)/3);
		int searchStartX = (int) (minX+(maxX-minX)/3);
		int searchEndX = (int) (maxX-(maxX-minX)/3);
		
		#ifdef DEBUG 
		line(contDisplay, Point(0, SearchStartY), Point(640, SearchStartY), Scalar(190), 4);
		line(contDisplay, Point(0, SearchEndY), Point(640, SearchEndY), Scalar(190), 4);
		#endif
		
		
		vector<Point> leftPoints, rightPoints, bottomPoints, topPoints, innerLeftPoints, innerRightPoints, innerBottomPoints;
		// find points for outside left, outside right, inner left, and inner right lines 
		for(int i = searchStartY; i < searchEndY; i++) {	
			leftPoints.push_back(Point(minElem( ytoxMap[i] ), i) );
			rightPoints.push_back(Point(maxElem( ytoxMap[i] ), i) );
			if(ytoxMap[i].size() == 4) {
				std::sort(ytoxMap[i].begin(), ytoxMap[i].end());
				//circle( contDisplay, Point(ytoxMap[i][2], i), 2, Scalar(64));
				innerLeftPoints.push_back(Point(ytoxMap[i][1], i));
				innerRightPoints.push_back(Point(ytoxMap[i][2], i));
			}
		}
		// find points for bottom line and inner bottom line
		for(int i = searchStartX; i < searchEndX; i++) {
			bottomPoints.push_back(Point(i, maxElem( xtoyMap[i] )) );
			if(i > searchStartX + (maxX-minX)/8 && i < searchEndX - (maxX-minX)/8) {
				//circle( contDisplay, Point(i, minElem( xtoyMap[i] )), 2, Scalar(64));
				innerBottomPoints.push_back(Point(i, minElem( xtoyMap[i] )));
			}	
		}
		// find points for top line 
		topPoints.push_back(Point(minX+2, minElem( xtoyMap[minX+2] )) );
		topPoints.push_back(Point(maxX-2, minElem( xtoyMap[maxX-2] )) );
		
		// could not find enough points to make lines 
		if(leftPoints.size() < 2 || rightPoints.size() < 2 || innerLeftPoints.size() < 2 
			|| innerRightPoints.size() < 2 || innerBottomPoints.size() < 2 || topPoints.size() < 2
			|| bottomPoints.size() < 2) continue;
	
		// fit lines to points via least squares reg 
		Vec4f leftLine, rightLine, bottomLine, topLine, innerLeftLine, innerRightLine, innerBottomLine;
		fitLine(leftPoints, leftLine, CV_DIST_L2, 0, 0.01, 0.01);
		fitLine(rightPoints, rightLine, CV_DIST_L2, 0, 0.01, 0.01);
		fitLine(bottomPoints, bottomLine, CV_DIST_L2, 0, 0.01, 0.01);
		fitLine(topPoints, topLine, CV_DIST_L2, 0, 0.01, 0.01);
		fitLine(innerLeftPoints, innerLeftLine, CV_DIST_L2, 0, 0.01, 0.01);
		fitLine(innerRightPoints, innerRightLine, CV_DIST_L2, 0, 0.01, 0.01);	
		fitLine(innerBottomPoints, innerBottomLine, CV_DIST_L2, 0, 0.01, 0.01);
		
		// find points of intersection for hexagon corners 
		Point2f bottomLeft, bottomRight, topLeft, topRight, innerBottomLeft, innerBottomRight, innerTopLeft, innerTopRight;
		bool a = findIntersection(leftLine, bottomLine , bottomLeft);
		a = a&& findIntersection(rightLine, bottomLine , bottomRight);
		a = a&& findIntersection(topLine, leftLine , topLeft);
		a = a&& findIntersection(topLine, rightLine , topRight);
		a = a&& findIntersection(innerBottomLine, innerLeftLine , innerBottomLeft);
		a = a&& findIntersection(innerBottomLine, innerRightLine , innerBottomRight);
		a = a&& findIntersection(topLine, innerLeftLine , innerTopLeft);
		a = a&& findIntersection(topLine, innerRightLine , innerTopRight);
		
		drawLine(contDisplay, bottomLine);
		drawLine(contDisplay, leftLine);
		drawLine(contDisplay, rightLine);
		drawLine(contDisplay, topLine);

		if(!a) continue;
		
		// append points hexagon corners to array 
		std::vector<cv::Point2f> corners; 
		corners.push_back(bottomLeft);
		corners.push_back(bottomRight);
		corners.push_back(topLeft);
		corners.push_back(topRight); 
		corners.push_back(innerTopLeft);
		corners.push_back(innerTopRight);	
		corners.push_back(innerBottomLeft);	
		corners.push_back(innerBottomRight);				
		for(int i = 0; i < corners.size(); i++ ){
			circle( frame, corners[i], 0, Scalar(255, 0, 0));			
		}
		
	
		updateFPS();
		sendUDP(data);
		
	}
	return 0;
}

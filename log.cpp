#include <stdio.h>
#include "log.h"
#include <string>

using namespace std;

FILE *file;

void initLog() {
//	file = fopen("vision.log", "a"); 

}

void log(std::string line) {
	FILE *file = fopen("/home/ubuntu/Documents/2019/FRC-2019-Vision/vision.log", "a+");
	fputs(line.c_str(), file);
	//fflush(file);
	//printf("logging \n");	
	fclose(file);
}

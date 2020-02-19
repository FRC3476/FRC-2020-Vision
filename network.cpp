#include <stdio.h> 
#include "network.h"
#include "arpa/inet.h"
#include "sys/socket.h"
#include <cstring>
#include <stdlib.h>
#include <thread>
#include <mutex>

#define PORT 5805
#define IP "10.34.76.45"
#define MAXLINE 100

#define IS_IP '0'
#define IS_EXP '1'

bool expHigh = false;
bool recieveIP = false;
bool setup = false;
int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP); 
sockaddr_in server;
//int testV = 0;
void recvUDP();
std::mutex mutex;

void setupUDP() {
	
        std::thread recvThread(recvUDP);
        recvThread.detach();

}

void setupTransmit(const char* ip) {
	printf("IP: %s \n \n", ip);	
	const char *buffer = "test UDP packet";
	int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP); 
	//bzero(&server, sizeof(server));	
	memset(&server, 0, sizeof(server));
	server.sin_family = AF_INET;
	server.sin_addr.s_addr = inet_addr(ip);
	server.sin_port = htons(PORT);
	mutex.lock();
	setup = true;
	mutex.unlock();
	//std::thread recvThread(recvUDP);
	//recvThread.detach();
}

int sendUDP(std::vector<exp_data> d) {
	mutex.lock();
	if(!setup) {
		mutex.unlock();
		return -1;
	}
	mutex.unlock();
	float *data = (float *)malloc(sizeof(float)*2*d.size());
	int c = 0;
	for(int i = 0; i < d.size(); i++) {
		data[c] = (float)d[i].centroid.x;
		data[c+1] = (float)d[i].centroid.y;
		c+=2;
	}

	//printf("%f", data[0]);
	

	sendto(s, data, sizeof(float)*4*d.size(), 0, (sockaddr*)&server, sizeof(server));
	free(data);	
	return 0; 
}

void recvUDP() {
	//return;
	int r = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	sockaddr_in raddr;
	raddr.sin_family = AF_INET;
	raddr.sin_addr.s_addr = INADDR_ANY;
	raddr.sin_port = htons(PORT);

	bind(r, (sockaddr*) &raddr, (socklen_t)sizeof(raddr));
	ssize_t len;
	char *data = new char[MAXLINE];
	while(1) {
		
		//testV++;
		printf("trying to recieve \n");
		len = recv(r, data, MAXLINE, MSG_WAITALL);
		data[len] = '\0';
		if(data[0] == IS_IP) {
			if(recieveIP != true){
				recieveIP = true;
				setupTransmit(data+1);
			}
		} else if(data[0] == IS_EXP) {
			mutex.lock();
			expHigh = data[1] == 'h';  
			mutex.unlock();
		}
		printf("finished\n");
		printf("%s, %lu\n", data, strlen(data));
	}
	delete[] data;
}

bool getExposure() {

	mutex.lock();
	bool temp = expHigh; 
	mutex.unlock();
	return temp;
}

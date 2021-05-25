#pragma once 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <queue>
#include <mutex>
#include <unistd.h>
#include "base64.h"
#include "ArduinoJson.h"
#include "benchmark.h"
#include "net.h"
#include "cpu.h"

#define PROB fprintf(stderr, ">>> prob: %d\n", __LINE__);

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

void startFramework(const char* function_name, int w, int h, bool disable_camera = false);

void enableStream();
void disableStream();

int MatToJpgByteArray(const cv::Mat mat, std::vector<unsigned char> &buff);
void CaptureThread(void);
void SendingThread(void);
void PIPEReadThread(void);
void PIPESendThread(void);
void getMat(cv::Mat &image);
void sendMat(cv::Mat &image);
int readPIPE(std::string &payload);
int sendPIPE(std::string &payload);
void sendPIPEMessage(std::string msg);
void sendPIPEJsonDoc(DynamicJsonDocument &doc);
bool isStreamOpend(void);

int checkDevice(const char* id);
std::string getFunctionName(void);

int loadModel(const std::string path, ncnn::Net &net);
int loadModel(const std::string pathparam, const std::string pathbin, ncnn::Net &net);
void resetAllocator(void);

void resetRenderDoc(void);
void RD_addRectangle(float x, float y, float w, float h, std::string color);
void RD_addCircle(float x, float y, float r, std::string color);
void RD_addString(float x, float y, std::string payload, std::string color);
void RD_addPoint(float x, float y, std::string color);
void RD_addLine(float x1, float y1, float x2, float y2, int thickness, std::string color);

void RD_addRectangle(double x, double y, double w, double h, std::string color);
void RD_addCircle(double x, double y, double r, std::string color);
void RD_addString(double x, double y, std::string payload, std::string color);
void RD_addPoint(double x, double y, std::string color);
void RD_addLine(double x1, double y1, double x2, double y2, int thickness, std::string color);

void RD_addRectangle(int x, int y, int w, int h, std::string color);
void RD_addCircle(int x, int y, int r, std::string color);
void RD_addString(int x, int y, std::string payload, std::string color);
void RD_addPoint(int x, int y, std::string color);
void RD_addLine(int x1, int y1, int x2, int y2, int thickness, std::string color);

void RD_addRectangle(float x, float y, float w, float h, std::string color, int multiple);
void RD_addCircle(float x, float y, float r, std::string color, int multiple);
void RD_addString(float x, float y, std::string payload, std::string color, int multiple);
void RD_addPoint(float x, float y, std::string color, int multiple);
void RD_addLine(float x1, float y1, float x2, float y2, int thickness, std::string color, int multiple);
void RD_addRectangle(double x, double y, double w, double h, std::string color, int multiple);
void RD_addCircle(double x, double y, double r, std::string color, int multiple);
void RD_addString(double x, double y, std::string payload, std::string color, int multiple);
void RD_addPoint(double x, double y, std::string color, int multiple);
void RD_addLine(double x1, double y1, double x2, double y2, int thickness, std::string color, int multiple);
void RD_addRectangle(int x, int y, int w, int h, std::string color, int multiple);
void RD_addCircle(int x, int y, int r, std::string color, int multiple);
void RD_addString(int x, int y, std::string payload, std::string color, int multiple);
void RD_addPoint(int x, int y, std::string color, int multiple);
void RD_addLine(int x1, int y1, int x2, int y2, int thickness, std::string color, int multiple);

void RD_addPolygon(std::vector<cv::Point> &points, std::string color);
void RD_addPolygon(std::vector<cv::Point> &points, std::string color, int multiple);
void RD_clear(void);

void sendRenderDoc(void);
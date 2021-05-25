#ifndef BASE_H
#define BASE_H
#include <cmath>
#include <cstring>
#include "net.h"
#include <opencv2/core/core.hpp>

typedef struct FaceInfo {
    cv::Rect_<float> rect;
    float prob;
    int x[2];
    int y[2];
    int landmark[10];
    std::vector<float> feature;
    int match_id;
    float match_score;
} FaceInfo;

ncnn::Mat resize(ncnn::Mat src, int w, int h);

ncnn::Mat bgr2rgb(ncnn::Mat src);

ncnn::Mat rgb2bgr(ncnn::Mat src);

void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M);

void warpAffineMatrix(ncnn::Mat src, ncnn::Mat &dst, float *M, int dst_w, int dst_h);

#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "net.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <thread>
#include <queue>
#include <mutex>
#include <unistd.h>
#include "base64.h"
#include "benchmark.h"
#include "ArduinoJson.h"
#include "framework.h"

#define IMAGE_DIV 2

enum
{
    IMAGE_FEED_MODE_RGB,
    IMAGE_FEED_MODE_MASK
};

int image_feed_mode = IMAGE_FEED_MODE_RGB;

uint8_t l_min = 0, l_max = 27;
uint8_t a_min = 108, a_max = 148;
uint8_t b_min = 108, b_max = 148;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

#define SEGMENT_NUM         (8)
#define SEGMENT_HEIGHT      ((480 / IMAGE_DIV) / 8)
#define WINDOW_WIDTH        (100 / IMAGE_DIV)
#define WINDOW_WIDTH_DIV2   (WINDOW_WIDTH / 2)

int segment_pos[SEGMENT_NUM][2] = {0};
std::vector<cv::Point> segment_point;
int segment_pos_avg = 0;
int segment_pos_sum = 0;
int computeSegment(cv::Mat &src, cv::Mat &image, int seg, int left_x, int right_x, int &next_left_x, int &next_right_x)
{
    int lrange, rrange, idx;
    int max_hist;
    cv::Point marker;
    int urange = seg * SEGMENT_HEIGHT;
    int brange = seg * SEGMENT_HEIGHT + SEGMENT_HEIGHT;

    lrange = left_x - WINDOW_WIDTH_DIV2;
    rrange = left_x + WINDOW_WIDTH_DIV2;
    if(lrange < 0)
    {
        lrange = 0;
    }
    if(rrange > (640 / IMAGE_DIV))
    {
        rrange = 640 / IMAGE_DIV;
    }
    // fprintf(stderr, "compute seg %d, lx = %d, rx = %d, lrange = %d, rrange = %d, urange = %d, brange = %d\n"
    // , seg, left_x, right_x, lrange, rrange, urange, brange);

    max_hist = 0;
    idx = 0;
    int left_suma = 0;
    int left_sumb = 0;
    int left_vhist[WINDOW_WIDTH];

    marker.x = left_x;
    marker.y = (seg * SEGMENT_HEIGHT) + (SEGMENT_HEIGHT / 2);
    // cv::rectangle(src, cv::Rect(rx, ry, WINDOW_WIDTH, SEGMENT_HEIGHT), cv::Scalar(0, 255, 255), 2, cv::LINE_8, 0);
    #ifdef LOCAL_RENDER
        cv::drawMarker(src, marker, cv::Scalar(0, 255, 255), cv::MARKER_CROSS , 10, 2, 8);
    #else
        RD_addPoint(marker.x, marker.y, "#CDCD00", IMAGE_DIV);
    #endif


    for(int i = lrange; i < rrange; i++)
    {
        left_vhist[idx] = 0;
        for(int j = urange; j < brange; j++)
        {
            if(image.at<uint8_t>(j, i))
            {
                left_vhist[idx]++;
            }
        }
        left_suma += left_vhist[idx] * idx;
        left_sumb += left_vhist[idx];
        // cv::line(src, cv::Point(i, ry + SEGMENT_HEIGHT), cv::Point(i, ry + SEGMENT_HEIGHT - left_vhist[idx]), cv::Scalar(0, 255, 255), 1);
        idx++;
    }

    if(left_sumb == 0)
    {
        next_left_x = left_x;
    }
    else
    {
        next_left_x = left_suma / left_sumb + lrange;
    }
    segment_pos[seg][0] = next_left_x;
    
    lrange = right_x - WINDOW_WIDTH_DIV2;
    rrange = right_x + WINDOW_WIDTH_DIV2;
    if(lrange < 0)
    {
        lrange = 0;
    }
    if(rrange > (640 / IMAGE_DIV))
    {
        rrange = 640 / IMAGE_DIV;
    }

    max_hist = 0;
    idx = 0;
    int right_suma = 0;
    int right_sumb = 0;
    int right_vhist[WINDOW_WIDTH];

    marker.x = right_x;
    #ifdef LOCAL_RENDER
        cv::drawMarker(src, marker, cv::Scalar(0, 255, 255), cv::MARKER_CROSS , 10, 2, 8);
    #else
        RD_addPoint(marker.x, marker.y, "#CDCD00", IMAGE_DIV);
    #endif
    
    // cv::rectangle(src, cv::Rect(rx, ry, WINDOW_WIDTH, SEGMENT_HEIGHT), cv::Scalar(0, 255, 255), 2, cv::LINE_8, 0);

    for(int i = lrange; i < rrange; i++)
    {
        right_vhist[idx] = 0;
        for(int j = urange; j < brange; j++)
        {
            if(image.at<uint8_t>(j, i))
            {
                right_vhist[idx]++;
            }
        }
        right_suma += right_vhist[idx] * idx;
        right_sumb += right_vhist[idx];
        // cv::line(src, cv::Point(i, ry + SEGMENT_HEIGHT), cv::Point(i, ry + SEGMENT_HEIGHT - right_vhist[idx]), cv::Scalar(0, 255, 255), 1);
        idx++;
    }

    if(right_sumb == 0)
    {
        next_right_x = right_x;
    }
    else
    {
        next_right_x = right_suma / right_sumb + lrange;
    }
    segment_pos[seg][1] = next_right_x;
    cv::Point p;
    p.x = next_left_x + ((next_right_x - next_left_x) / 2);
    p.y = (seg * SEGMENT_HEIGHT) + (SEGMENT_HEIGHT / 2);
    segment_point.push_back(p);
    segment_pos_sum += p.x;

    return right_suma < right_sumb ? right_suma : right_sumb;

    // fprintf(stderr, "next_left_x = %d, next_right_x = %d\n"
    // , next_left_x, next_right_x);
}

void findLine(cv::Mat &image_mask, int &left_x, int &right_x)
{
    int vhist[image_mask.cols];

    for(int i = 0; i < image_mask.cols; i++)
    {
        vhist[i] = 0;
        for(int j = 0; j < image_mask.rows; j++)
        {
            if(image_mask.at<uint8_t>(j, i))
            {
                vhist[i]++;
            }
        }
    }

    int left_cnt = 0;
    for(int i = 0; i < image_mask.cols / 2; i++)
    {
        if(vhist[i] > left_cnt)
        {
            left_cnt = vhist[i];
            left_x = i;
        }
    }

    int right_cnt = 0;
    for(int i = image_mask.cols / 2; i < image_mask.cols; i++)
    {
        if(vhist[i] > right_cnt)
        {
            right_cnt = vhist[i];
            right_x = i;
        }
    }
}

int left_x = 0, right_x = 0;
void process(cv::Mat &image)
{
    int next_left_x;
    int next_right_x;
    int input_left_x = left_x;
    int input_right_x = right_x;

    cv::Mat image_lab;
    cv::Mat image_mask;
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
    cv::inRange(image_lab, cv::Scalar(l_min, a_min, b_min), cv::Scalar(l_max, a_max, b_max), image_mask);
    cv::morphologyEx(image_mask, image_mask, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    
    bool miss_flag = false;
    if(left_x == 0 || right_x == 0)
    {
        miss_flag = true;
    }
    else if(left_x >= right_x)
    {
        miss_flag = true;
    }
    else if((right_x - left_x) < WINDOW_WIDTH / 3)
    {
        miss_flag = true;
    }
    else
    {
        int sum = computeSegment(image, image_mask, SEGMENT_NUM - 1, input_left_x, input_right_x, next_left_x, next_right_x);
        if(sum < 30)
        {
            miss_flag = true;
        }
    }

    if(miss_flag == true)
    {
        findLine(image_mask, left_x, right_x);
        input_left_x = left_x;
        input_right_x = right_x;

        for(int seg = SEGMENT_NUM - 1; seg >= 0; seg--)
        {
            computeSegment(image, image_mask, seg, input_left_x, input_right_x, next_left_x, next_right_x);
            input_left_x = next_left_x;
            input_right_x = next_right_x;
            if(seg == SEGMENT_NUM - 1)
            {
                left_x = next_left_x;
                right_x = next_right_x;
            }
        }
    }
    else
    {
        input_left_x = next_left_x;
        input_right_x = next_right_x;
        left_x = next_left_x;
        right_x = next_right_x;

        for(int seg = SEGMENT_NUM - 2; seg >= 0; seg--)
        {
            computeSegment(image, image_mask, seg, input_left_x, input_right_x, next_left_x, next_right_x);
            input_left_x = next_left_x;
            input_right_x = next_right_x;
        }
    }

    // for(int i = 0; i < image_mask.cols; i++)
    // {
    //     cv::Rect rect(i * IMAGE_DIV, 480 - vhist[i], IMAGE_DIV, vhist[i]);
    //     cv::rectangle(image, rect, cv::Scalar(0, 255, 0), -1, cv::LINE_8, 0);
    // }

    // cv::Rect rect1(left_x, 0, IMAGE_DIV, 480 / IMAGE_DIV);
    // cv::rectangle(image, rect1, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);

    // cv::Rect rect2(right_x, 0, IMAGE_DIV, 480 / IMAGE_DIV);
    // cv::rectangle(image, rect2, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);

    cv::Vec4f fitline;
    for (int i = 0; i < segment_point.size(); i++)
    {
    #ifdef LOCAL_RENDER
        cv::circle(image, segment_point[i], 5, cv::Scalar(0, 0, 255), 2, 8, 0);
    #else
        RD_addPoint(segment_point[i].x, segment_point[i].y, "#CC0000", IMAGE_DIV);
    #endif
    }
    cv::fitLine(segment_point, fitline, cv::DIST_L2, 0, 1e-2, 1e-2);
    segment_point.clear();
    segment_pos_avg = segment_pos_sum / SEGMENT_NUM;
    segment_pos_sum = 0;

    if(fitline[0] != 0)
    {
        cv::Point point0;
        point0.x = fitline[2];
        point0.y = fitline[3];
        double k = fitline[1] / fitline[0];

        DynamicJsonDocument doc(4096);
        doc["x"] = point0.x;
        doc["y"] = point0.y;
        doc["k"] = k;
        sendPIPEJsonDoc(doc);
    
        // y = k(x - x0) + y0)
        cv::Point point1, point2;
        fprintf(stderr, "k = %.1lf \n", k);

        if(isStreamOpend())
        {
            if(k > -25 && k < 25)
            {
                point1.x = 0;
                point1.y = k * (point1.x - point0.x) + point0.y;
                point2.x = 640 / IMAGE_DIV;
                point2.y = k * (point2.x - point0.x) + point0.y;
            }
            else
            {
                point1.x = segment_pos_avg;
                point1.y = 0;
                point2.x = segment_pos_avg;
                point2.y = 480 / IMAGE_DIV;
            }

        #ifdef LOCAL_RENDER
            cv::line(image, point1, point2, cv::Scalar(0, 255, 0), 2, 8, 0);
        #else
            RD_addLine(point1.x, point1.y, point2.x, point2.y, 2, "#00CD00", IMAGE_DIV);
        #endif
        }
    }

    if(image_feed_mode == IMAGE_FEED_MODE_RGB)
    {
        sendMat(image);
    }
    else
    {
        sendMat(image_mask);
    }
}


void KmeansColorQuantization(cv::Mat image, int clusternum, cv::Point2f &max_center, float &variance_a, float &variance_b)
{
    cv::Mat image_lab;
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);

    int total_points = image_lab.rows * image_lab.cols;
    cv::Mat points(total_points, 2, CV_32F), labels;
    std::vector<cv::Point2f> centers;

    int idx = 0;
    for(int i = 0; i < image_lab.rows; i++)
    {
        for(int j = 0; j < image_lab.cols; j++)
        {
            cv::Vec3b lab = image_lab.at<cv::Vec3b>(i, j);

            points.at<float>(idx, 0) = lab.val[1];
            points.at<float>(idx, 1) = lab.val[2];
            idx++;
        }
    }

    double compactness = kmeans(points, clusternum, labels,
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    fprintf(stderr, "compactness = %lf\n", compactness);

    int cluster_idx_count[clusternum];
    memset(cluster_idx_count, 0, sizeof(int) * clusternum);

    for(int i = 0; i < total_points; i++ )
    {
        cluster_idx_count[labels.at<int>(i)]++;
    }

    int max_count = 0;
    int max_idx = 0;
    for(int i = 0; i < clusternum; i++)
    {
        fprintf(stderr, "%.1f, %.1f, %d, %d\n", centers[i].x, centers[i].y, i, cluster_idx_count[i]);
        if(cluster_idx_count[i] > max_count)
        {
            max_count = cluster_idx_count[i];
            max_idx = i;
        }
    }

    int dbuf[max_count][2];
    idx = 0;
    for(int i = 0; i < total_points; i++)
    {
        if(labels.at<int>(i) == max_idx)
        {
            int a = (int)(points.at<float>(i, 0));
            int b = (int)(points.at<float>(i, 1));
            // int da = abs((int)(centers[max_idx].x) - a);
            // int db = abs((int)(centers[max_idx].y) - b);
            // int d = da > db ? da : db;
            dbuf[idx][0] = abs((int)(centers[max_idx].x) - a);
            dbuf[idx][1] = abs((int)(centers[max_idx].y) - b);
            idx++;
        }
    }

    uint64_t suma = 0, sumb = 0;
    for(int i = 0; i < max_count; i++)
    {
        suma += dbuf[i][0];
        sumb += dbuf[i][1];
    }
    int avga = (int)((float)suma / (float)(max_count));
    int avgb = (int)((float)sumb / (float)(max_count));

    uint64_t square_suma = 0, square_sumb = 0;
    for(int i = 0; i < max_count; i++)
    {
        int temp = avga - dbuf[i][0];
        square_suma += temp * temp;
        temp = avgb - dbuf[i][1];
        square_sumb += temp * temp;
    }
    float square_avga = (float)square_suma / (float)max_count;
    float square_avgb = (float)square_sumb / (float)max_count;
    variance_a = sqrt(square_avga);
    variance_b = sqrt(square_avgb);
    max_center = centers[max_idx];
}

int main(int argc, char **argv)
{
    startFramework("Lane Line Tracker", 640, 480);

    char strbuf[30];
    while (1)
    {
        double t_start = ncnn::get_current_time();
        cv::Mat mat_src;
        getMat(mat_src);
        cv::Mat mat_div;
        cv::resize(mat_src, mat_div, cv::Size(), 1.0f / IMAGE_DIV, 1.0f / IMAGE_DIV, cv::INTER_AREA);
        
        std::string payload;
        if(readPIPE(payload))
        {
            try
            {
                DynamicJsonDocument doc(1024);
                deserializeJson(doc, payload);
                if(doc.containsKey("config"))
                {
                    if(doc["config"] == getFunctionName() || doc["config"] == "web_update_data")
                    {
                        if(doc.containsKey("l_min"))
                        {
                            uint8_t _l_min = doc["l_min"];
                            uint8_t _l_max = doc["l_max"];
                            uint8_t _a_min = doc["a_min"];
                            uint8_t _a_max = doc["a_max"];
                            uint8_t _b_min = doc["b_min"];
                            uint8_t _b_max = doc["b_max"];
                            l_min = _l_min;
                            l_max = _l_max;
                            a_min = _a_min;
                            a_max = _a_max;
                            b_min = _b_min;
                            b_max = _b_max;
                            sendPIPEMessage("Data updated.");
                        }
                        else if(doc.containsKey("mode"))
                        {
                            if(doc["mode"] == "mask")
                            {
                                image_feed_mode = IMAGE_FEED_MODE_MASK;
                            }
                            else
                            {
                                image_feed_mode = IMAGE_FEED_MODE_RGB;
                            }
                        }
                        else
                        {
                            cv::Rect roi;
                            roi.x = doc["x"].as<int>();
                            roi.y = doc["y"].as<int>();
                            roi.width = doc["w"].as<int>();
                            roi.height = doc["h"].as<int>();
                            roi.x /= IMAGE_DIV;
                            roi.y /= IMAGE_DIV;
                            roi.width /= IMAGE_DIV;
                            roi.height /= IMAGE_DIV;
                            cv::Mat cut = mat_div(roi).clone();
                            // imwrite("temp.jpg", cut);
                            float variance_a, variance_b;
                            cv::Point2f result;
                            KmeansColorQuantization(cut, 2, result, variance_a, variance_b);
                            int delta_a = 10 + (int)(variance_a / 2.0f);
                            int delta_b = 10 + (int)(variance_b / 2.0f);
                            int a = (int)(result.x);
                            int b = (int)(result.y);
                            l_min = 0;
                            l_max = 255;
                            a_min = a - delta_a < 0 ? 0 : a - delta_a;
                            a_max = a + delta_a > 255 ? 255: a + delta_a;
                            b_min = b - delta_b < 0 ? 0 : b - delta_b;
                            b_max = b + delta_b > 255 ? 255 : b + delta_b;
                            char buf[128];
                            // sprintf(buf, "Data updated. A = %.0f, B = %.0f, vA = %.2f, vB = %.2f", result.x, result.y, variance_a, variance_b);

                            DynamicJsonDocument doc2(2048);
                            // sprintf(buf, "Data updated. A = %.0f, B = %.0f, vA = %.2f, vB = %.2f", result.x, result.y, variance_a, variance_b);

                            doc2["a_cal"] = result.x;
                            doc2["b_cal"] = result.y;
                            doc2["va"] = variance_a;
                            doc2["vb"] = variance_b;
                            doc2["l_min"] = l_min;
                            doc2["l_max"] = l_max;
                            doc2["a_min"] = a_min;
                            doc2["a_max"] = a_max;
                            doc2["b_min"] = b_min;
                            doc2["b_max"] = b_max;
                            sendPIPEJsonDoc(doc2);

                            doc2["web"] = 1;
                            sendPIPEJsonDoc(doc2);
                        }
                    }
                }
            }
            catch(...)
            {
                fprintf(stderr, "[ERROR] Can not parse json.");
            }
        }
        process(mat_div);
        

        double t_end = ncnn::get_current_time();
        double dt_total = t_end - t_start;
        fprintf(stderr, "total %3.1f\n", dt_total);
    }
    return 0;
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstring>
#include "framework.h"

#define IMAGE_DIV 2

enum
{
    IMAGE_FEED_MODE_RGB,
    IMAGE_FEED_MODE_MASK
};

int image_feed_mode = IMAGE_FEED_MODE_RGB;

uint8_t l_min = 0, l_max = 255;
uint8_t a_min = 102, a_max = 240;
uint8_t b_min = 160, b_max = 205;

void process(cv::Mat &image, cv::Mat &mat_mask, cv::Point2f &center, float &radius, cv::Point &mcenter)
{
    cv::Mat image_lab;
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
    cv::inRange(image_lab, cv::Scalar(l_min, a_min, b_min), cv::Scalar(l_max, a_max, b_max), mat_mask);
    cv::morphologyEx(mat_mask, mat_mask, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> max_contour;
    std::vector<cv::Vec4i> hireachy;
    cv::findContours(mat_mask, contours, hireachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));

    if (contours.size() > 0)
    {
        double maxArea=0;
        for (int i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea)
            {
                maxArea = area;
                max_contour = contours[i];
            }
        }

        cv::minEnclosingCircle(max_contour, center, radius);

        cv::Moments m = cv::moments(max_contour);
        mcenter.x = int(m.m10 / m.m00);
        mcenter.y = int(m.m01 / m.m00);
    }
    else
    {
        RD_clear();
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
    // cv::Mat image;

    // image = cv::imread("./images/test.jpg");

    // cv::Point2f result;
    // float variance_a, variance_b;
    // KmeansColorQuantization(image, 2, result, variance_a, variance_b);

    // fprintf(stderr, "%f, %f, %f, %f\n", result.x, result.y, variance_a, variance_b);

    // return 0;

    startFramework("Color Tracker", 640, 480);

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
        
        
        cv::Point2f center;
	    float radius=0;
        cv::Point mcenter;
        cv::Mat mat_mask;
        process(mat_div, mat_mask, center, radius, mcenter);

        if(isStreamOpend() && radius != 0)
        {
        #ifdef LOCAL_RENDER
            cv::circle(mat_div, cv::Point(center.x, center.y), (int)radius, cv::Scalar(0,255,0), 2);
            cv::circle(mat_div, mcenter, 2, cv::Scalar(0, 255, 0), -1);
        #else
            RD_addCircle(center.x, center.y, radius, "#00CD00", IMAGE_DIV);
            RD_addPoint(mcenter.x, mcenter.y, "#00CD00", IMAGE_DIV);
        #endif
        }


        DynamicJsonDocument doc(1024);

        doc["cx"] = center.x * IMAGE_DIV;
        doc["cy"] = center.y * IMAGE_DIV;
        doc["r"] = (int)radius * IMAGE_DIV;
        doc["mx"] = mcenter.x * IMAGE_DIV;
        doc["my"] = mcenter.y * IMAGE_DIV;

        sendPIPEJsonDoc(doc);

        if(image_feed_mode == IMAGE_FEED_MODE_RGB)
        {
            sendMat(mat_div);
        }
        else
        {
            sendMat(mat_mask);
        }

        double t_end = ncnn::get_current_time();
        double dt_total = t_end - t_start;
        fprintf(stderr, "total %3.1f\n", dt_total);
    }

    return 0;
}

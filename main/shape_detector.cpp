#include "opencv2/features2d/features2d.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "framework.h"

#define IMAGE_DIV 2

enum
{
    SHAPE_UNIDENTIFIED = 0,
    SHAPE_TRIANGLE,
    SHAPE_SQUARE,
    SHAPE_RECTANGLE,
    SHAPE_PENTAGON,
    SHAPE_CIRCLE
};

const char* kShapeNames[] = {
    "unidentified",
    "triangle",
    "square",
    "rectangle",
    "pentagon",
    "circle"
};

using namespace cv;
using namespace std;

int ShapeDetect(const Mat &curve)
{
    int shape = SHAPE_UNIDENTIFIED;
    double peri = arcLength(curve, true);
    Mat approx;
    approxPolyDP(curve, approx, 0.04 * peri, true); // 0.01~0.05
    const int num_of_vertices = approx.rows;

    // if the shape is a triangle, it will have 3 vertices
    if (num_of_vertices == 3)
    {
        shape = SHAPE_TRIANGLE;
    }
    else if (num_of_vertices == 4)
    {// if the shape has 4 vertices, it is either a square or a rectangle
        // Compute the bounding box of the contour and
        // use the bounding box to compute the aspect ratio
        Rect rec = boundingRect(approx);
        double ar = 1.0 * rec.width / rec.height;

        // A square will have an aspect ratio that is approximately
        // equal to one, otherwise, the shape is a rectangle
        if (ar >= 0.95 && ar <= 1.05)
        {
            shape = SHAPE_SQUARE;
        }
        else
        {
            shape = SHAPE_RECTANGLE;
        }
    }
    else if (num_of_vertices == 5)
    {// if the shape is a pentagon, it will have 5 vertices
        shape = SHAPE_PENTAGON;
    }
    else
    {// otherwise, we assume the shape is a circle
        shape = SHAPE_CIRCLE;
    }
    return shape;
}

int main( int argc, char** argv )
{
    startFramework("Shape Detector", 640, 480);

    cv::Mat mat_ref, mat_div;
    bool is_ref_need_update = true;
    int stable_frame_count = 0;
    int active_frame_count = 0;

    // std::string testpath(argv[1]);
    // Mat image = imread(testpath);
    // Mat gray;
    // cvtColor(image, gray, COLOR_BGR2GRAY);
    
    // Mat blurred, thresh;
    // GaussianBlur(gray, blurred, Size(5, 5), 0.0);
    // threshold(blurred, thresh, 60, 255, THRESH_BINARY);

    // vector< vector<Point> > contours;
    // vector<Vec4i> hierarchy;
    // findContours(thresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    // vector<Point> c;
    // for (size_t i = 0; i < contours.size(); i++)
    // {
    //     c = contours[i];
    //     // Rect crect = boundingRect(c);
    //     // compute the center of the contour, then detect the name of the
    //     // shape using only the contour
    //     Moments M = moments(c);
    //     int cX = static_cast<int>(M.m10 / M.m00);
    //     int cY = static_cast<int>(M.m01 / M.m00);
    //     int shape = ShapeDetect(Mat(c));
    //     char str[100];
    //     if(shape == SHAPE_RECTANGLE || shape == SHAPE_SQUARE)
    //     {
    //         RotatedRect rotate_area = minAreaRect(c);
    //         sprintf(str, "%s %.1f deg", kShapeNames[shape], rotate_area.angle);
    //     }
    //     else
    //     {
    //         sprintf(str, "%s", kShapeNames[shape]);
    //     }
    //     drawContours(image, contours, i, Scalar(0, 255, 0), 2);
    //     Point pt(cX, cY);
    //     putText(image, str, pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
    //     imwrite("result.png", image);
    // }

    // return 0;

    if(1)
    {
        Mat mat_src;
        getMat(mat_src);
        resize(mat_src, mat_div, Size(), 1.0f / IMAGE_DIV, 1.0f / IMAGE_DIV, INTER_AREA);
        Mat mat_div_gray;
        cvtColor(mat_div, mat_div_gray, COLOR_BGR2GRAY);
        GaussianBlur(mat_div_gray, mat_div_gray, Size(5, 5), 0);
        mat_ref = mat_div_gray;
    }

    while(1)
    {
        double t_start = ncnn::get_current_time();
        Mat mat_src;
        getMat(mat_src);
        resize(mat_src, mat_div, Size(), 1.0f / IMAGE_DIV, 1.0f / IMAGE_DIV, INTER_AREA);
        Mat mat_div_gray;
        cvtColor(mat_div, mat_div_gray, COLOR_BGR2GRAY);
        GaussianBlur(mat_div_gray, mat_div_gray, Size(5, 5), 0);

        std::string payload;
        if(readPIPE(payload))
        {
            sendPIPEMessage("Background updated.");
            is_ref_need_update = true;
        }

        Mat mat_delta;
        absdiff(mat_ref, mat_div_gray, mat_delta);
        
        double delta = cv::sum(mat_delta)[0];
        fprintf(stderr, "delta = %lf\n", delta);

        if(delta < 150000)
        {
            is_ref_need_update = true;
            // stable_frame_count++;
            // if(stable_frame_count > 50)
            // {
            //     is_ref_need_update = true;
            //     stable_frame_count = 0;
            // }
        }
        else
        {
            stable_frame_count = 0;
            active_frame_count++;
        }

        Mat mat_mask;
        threshold(mat_delta, mat_mask, 50, 255, THRESH_BINARY);
        morphologyEx(mat_mask, mat_mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
        // dilate(mat_mask, mat_mask, Mat(), Point(-1,-1), 2);

        cv::Point2f center;
        float radius;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Point> max_contour;
        std::vector<cv::Vec4i> hireachy;
        cv::findContours(mat_mask, contours, hireachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));
        int area;
        int valid_cnt = 0;
        if (contours.size() > 0)
        {
            std::vector<cv::Point> c;
            DynamicJsonDocument doc(1024 * 16);
            for (size_t i = 0; i < contours.size(); i++)
            {
                area = cv::contourArea(contours[i]);
                if (area < 50)
                {
                    continue;
                }
                Rect bbox = boundingRect(contours[i]);
                c = contours[i];
                doc["shape"][valid_cnt]["x"] = bbox.x * IMAGE_DIV;
                doc["shape"][valid_cnt]["y"] = bbox.y * IMAGE_DIV;
                doc["shape"][valid_cnt]["w"] = bbox.width * IMAGE_DIV;
                doc["shape"][valid_cnt]["h"] = bbox.height * IMAGE_DIV;
                // Rect crect = boundingRect(c);
                // compute the center of the contour, then detect the name of the
                // shape using only the contour
                Moments M = moments(c);
                int cX = static_cast<int>(M.m10 / M.m00);
                int cY = static_cast<int>(M.m01 / M.m00);
                int shape = ShapeDetect(Mat(c));
                doc["shape"][valid_cnt]["name"] = kShapeNames[shape];

                char str[100];
                if(shape == SHAPE_RECTANGLE || shape == SHAPE_SQUARE)
                {
                    RotatedRect rotate_area = minAreaRect(c);
                    doc["shape"][valid_cnt]["angle"] = rotate_area.angle;
                    sprintf(str, "%s %.1f deg, area = %d", kShapeNames[shape], rotate_area.angle, area);
                }
                else
                {
                    sprintf(str, "%s, area = %d", kShapeNames[shape], area);
                }
                
            #ifdef LOCAL_RENDER
                drawContours(mat_div, contours, i, Scalar(0, 255, 0), 2);
                putText(mat_div, str, Point(cX, cY), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            #else
                // RD_addPolygon(c, "#00CD00", IMAGE_DIV);
                drawContours(mat_div, contours, i, Scalar(0, 255, 0), 2);
                RD_addString(cX, cY, str, "#CDCD00", IMAGE_DIV);
            #endif

                doc["shape"][valid_cnt]["area"] = area;
                valid_cnt++;
            }

            doc["num"] = valid_cnt;
            sendPIPEJsonDoc(doc);
        }
        else
        {
            RD_clear();
        }

        sendMat(mat_div);

        if(is_ref_need_update)
        {
            is_ref_need_update = false;
            mat_ref = mat_div_gray.clone();
        }

        double t_end = ncnn::get_current_time();
        double dt_total = t_end - t_start;
        fprintf(stderr, "total %3.1f\n", dt_total);
    }
    
    return 0;
}

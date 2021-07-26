#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstring>
#include "framework.h"
#include "benchmark.h"

#define IMAGE_DIV 2

typedef struct{
    cv::Rect roi;
    int contour_idx = -1;
}mask_object;

using namespace cv;

int compute_obj_distance(Rect r1, Rect r2)
{
	int dmin;
    int r1_cx = r1.x + (r1.width / 2);
    int r1_cy = r1.y + (r1.height / 2);
    int r2_cx = r2.x + (r2.width / 2);
    int r2_cy = r2.y + (r2.height / 2);
	int dx, dy;
	dx = abs(r2_cx - r1_cx);
	dy = abs(r2_cy - r1_cy);
	
	if((dx < ((r1.width + r2.width)/ 2)) && (dy >= ((r1.height + r2.height) / 2)))
	{
		dmin = dy - ((r1.height + r2.height) / 2);
	}
	else if((dx >= ((r1.width + r2.width)/ 2)) && (dy < ((r1.height + r2.height) / 2)))
	{
		dmin = dx - ((r1.width + r2.width)/ 2);
	}
	else if((dx >= ((r1.width + r2.width)/ 2)) && (dy >= ((r1.height + r2.height) / 2)))
	{
		int delta_x = dx - ((r1.width + r2.width)/ 2);
		int delta_y = dy - ((r1.height + r2.height)/ 2);
        dmin = delta_x > delta_y ? delta_x : delta_y;
	}
	else
	{
		dmin = -1;
	}
	
	return dmin;
}

int merge_roi(std::vector<mask_object> &objects, std::vector<std::vector<cv::Point>> &contours, int dthr)
{
    int i, j;
    int merge_count = 0;

    for(i = 0; i < objects.size(); i++)
    {
        if(objects[i].roi.width == 0)
        {
            continue;
        }
        for(j = i + 1; j < objects.size(); j++)
        {
            if(objects[j].roi.width == 0)
            {
                continue;
            }
            if(compute_obj_distance(objects[i].roi, objects[j].roi) <= dthr)
            {
                merge_count++;

                Point br1 = objects[i].roi.br();
                Point br2 = objects[j].roi.br();
                Point tl1 = objects[i].roi.tl();
                Point tl2 = objects[j].roi.tl();

                tl1.x = tl1.x < tl2.x ? tl1.x : tl2.x;
                tl1.y = tl1.y < tl2.y ? tl1.y : tl2.y;
                br1.x = br1.x > br2.x ? br1.x : br2.x;
                br1.y = br1.y > br2.y ? br1.y : br2.y;
                
                objects[i].roi = Rect(tl1, br1);
                objects[j].roi.width = 0;

                std::vector<cv::Point> merge_points = contours[objects[i].contour_idx];
                merge_points.insert(merge_points.end(), contours[objects[j].contour_idx].begin(), contours[objects[j].contour_idx].end());
                contours[objects[i].contour_idx] = merge_points;

                i = 0;
                j = 0;
            }
        }
    }
    return merge_count;
}

int main(int argc, char **argv)
{
    startFramework("Motion Tracker", 640, 480);

    cv::Mat mat_ref, mat_div;
    bool is_ref_need_update = false;
    int stable_frame_count = 0;
    int active_frame_count = 0;

    if(1)
    {
        Mat mat_src;
        getMat(mat_src);
        resize(mat_src, mat_div, Size(), 1.0f / IMAGE_DIV, 1.0f / IMAGE_DIV, INTER_AREA);
        Mat mat_div_gray;
        cvtColor(mat_div, mat_div_gray, COLOR_BGR2GRAY);
        GaussianBlur(mat_div_gray, mat_div_gray, Size(21, 21), 0);
        mat_ref = mat_div_gray;
    }

    while (1)
    {
        double t_start = ncnn::get_current_time();
        Mat mat_src;
        getMat(mat_src);
        resize(mat_src, mat_div, Size(), 1.0f / IMAGE_DIV, 1.0f / IMAGE_DIV, INTER_AREA);
        Mat mat_div_gray;
        cvtColor(mat_div, mat_div_gray, COLOR_BGR2GRAY);
        GaussianBlur(mat_div_gray, mat_div_gray, Size(21, 21), 0);

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

        // if(active_frame_count > 200)
        // {
        //     is_ref_need_update = true;
        //     stable_frame_count = 0;
        //     active_frame_count = 0;
        // }

        Mat mat_mask;
        threshold(mat_delta, mat_mask, 50, 255, THRESH_BINARY);
        morphologyEx(mat_mask, mat_mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
        dilate(mat_mask, mat_mask, Mat(), Point(-1,-1), 2);

        cv::Point2f center;
        float radius;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Point> max_contour;
        std::vector<cv::Vec4i> hireachy;
        cv::findContours(mat_mask, contours, hireachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));

        if (contours.size() > 0)
        {
            int area = 0;
            DynamicJsonDocument doc(1024 * 16);
            int valid_cnt = 0;
            for (int i = 0; i < contours.size(); i++)
            {
                area = cv::contourArea(contours[i]);
                if (area > 50)
                {
                    valid_cnt++;
                    // std::vector<cv::Point> hull(contours[i].size());
                    // convexHull(contours[i], hull);
                    RotatedRect rotate_area = minAreaRect(contours[i]);
                    Rect bbox = boundingRect(contours[i]);
                    // Rect bbox = boundingRect(contours[i]);

                    // double score = matchShapes(contours[i], template_points, CONTOURS_MATCH_I3, 1);
                    // score = 100 - (score * 100);
                    // if(score < 0)
                    // {
                    //     score = 0;
                    // }
                    // if(score > 100)
                    // {
                    //     score = 100;
                    // }

                #ifdef LOCAL_RENDER
                    rectangle(mat_div, bbox, Scalar( 255, 255, 255 ), 2, 1 );
                    Point2f vertices[4];
                    rotate_area.points(vertices);
                    for (int i = 0; i < 4; i++)
                        line(mat_div, vertices[i], vertices[(i+1)%4], Scalar(0,255,255), 2);
                    // drawContours(mat_div, vertices, i, Scalar(0, 255, 0), 2);
                    char str[20];
                    sprintf(str, "%.1f deg", rotate_area.angle);
                    putText(mat_div, str, rotate_area.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
                #else
                    RD_addRectangle(bbox.x, bbox.y, bbox.width, bbox.height, "#00CD00", IMAGE_DIV);
                    // Point2f vertices[4];
                    // rotate_area.points(vertices);
                    // for (int k = 0; k < 4; k++)
                    //     line(mat_div, vertices[i], vertices[(i+1)%4], Scalar(0,255,255), 2);
                    //     RD_addLine(vertices[k].x, vertices[(k+1)%4].x, vertices[k].y, vertices[(k+1)%4].y, 2, "#CDCD00", IMAGE_DIV);
                    char str[20];
                    sprintf(str, "%.1f deg", rotate_area.angle);
                    RD_addString(rotate_area.center.x, rotate_area.center.y, str, "#CC0000", IMAGE_DIV);
                #endif
                    doc["roi"][i]["x"] = bbox.x * IMAGE_DIV;
                    doc["roi"][i]["y"] = bbox.y * IMAGE_DIV;
                    doc["roi"][i]["w"] = bbox.width * IMAGE_DIV;
                    doc["roi"][i]["h"] = bbox.height * IMAGE_DIV;
                    doc["roi"][i]["angle"] = rotate_area.angle;
                    doc["roi"][i]["area"] = area * IMAGE_DIV * IMAGE_DIV;
                }
            }

            doc["num"] = valid_cnt;
            sendPIPEJsonDoc(doc);
        }
        else
        {
            RD_clear();
        }

        // if (contours.size() > 0)
        // {
        //     double maxArea = 0;
        //     double area = 0;
        //     std::vector<mask_object> objects;
        //     for (int i = 0; i < contours.size(); i++)
        //     {
        //         area = cv::contourArea(contours[i]);

        //         if (area > 100)
        //         {
        //             mask_object obj;
        //             obj.roi = boundingRect(contours[i]);
        //             obj.contour_idx = i;
        //             objects.push_back(obj);
        //         }
        //     }

        //     merge_roi(objects, contours, 40);

        //     if(objects.size() > 0)
        //     {
        //         DynamicJsonDocument doc(1024 * 16);
        //         doc["num"] = objects.size();
        //         for (int i = 0; i < objects.size(); i++)
        //         {
        //             if(objects[i].roi.width == 0)
        //             {
        //                 continue;
        //             }

        //             std::vector<cv::Point> hull(contours[objects[i].contour_idx].size());
        //             convexHull(contours[objects[i].contour_idx], hull);
        //             RotatedRect rotate_area = minAreaRect(hull);
                    
        //         #define LOCAL_RENDER
        //         #ifdef LOCAL_RENDER
        //             rectangle(mat_div, objects[i].roi, Scalar( 0, 255, 0 ), 2, 1 );
        //             Point2f vertices[4];
        //             rotate_area.points(vertices);
        //             for (int i = 0; i < 4; i++)
        //                 line(mat_div, vertices[i], vertices[(i+1)%4], Scalar(0,255,255), 2);
        //             // drawContours(mat_div, vertices, i, Scalar(0, 255, 0), 2);
        //             char str[20];
        //             sprintf(str, "%.1f deg", rotate_area.angle);
        //             putText(mat_div, str, rotate_area.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
        //         #else
        //             RD_addRectangle(objects[i].roi.x, objects[i].roi.y, objects[i].roi.width, objects[i].roi.height, "#00CD00", IMAGE_DIV);
        //         #endif
        //             doc["roi"][i]["x"] = objects[i].roi.x;
        //             doc["roi"][i]["y"] = objects[i].roi.y;
        //             doc["roi"][i]["width"] = objects[i].roi.width;
        //             doc["roi"][i]["height"] = objects[i].roi.height;
        //         }

        //         sendPIPEJsonDoc(doc);
        //     }
        // }

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

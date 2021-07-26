#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstring>
#include "framework.h"
#include "benchmark.h"

#define IMAGE_DIV 2

typedef struct{
    std::vector<cv::Point> contour;
    std::string name;
}shape_template_t;

std::vector<shape_template_t> shape_templates;

using namespace cv;

int save_templates(std::vector<shape_template_t> &templates, std::string path)
{
    FILE *fp_feature;
    FILE *fp_info;
	
    fp_feature = fopen(std::string(path + "/shape_matching_templates.dat").c_str(), "wb");
    if(fp_feature == NULL)
	{
		perror("open file error:");
		return 0;
	}

    fp_info = fopen(std::string(path + "/shape_matching_info.json").c_str(), "w");
    if(fp_info == NULL)
	{
		perror("open file error:");
		return 0;
	}

    DynamicJsonDocument doc(4096);
    std::string payload;

    doc["shape_num"] = shape_templates.size();
    for(int i = 0; i < shape_templates.size(); i++)
    {
        doc["shapes"][i]["name"] = shape_templates[i].name;
        doc["shapes"][i]["point_num"] = shape_templates[i].contour.size();

        for(int j = 0; j < shape_templates[i].contour.size(); j++)
        {
            fwrite(&(shape_templates[i].contour[j].x), sizeof(int), 1, fp_feature);
            fwrite(&(shape_templates[i].contour[j].y), sizeof(int), 1, fp_feature);
        }
    }

    serializeJson(doc, payload);
    fwrite(payload.c_str(), payload.length(), 1, fp_info);

    fclose(fp_feature);
    fclose(fp_info);

    return 1;
}

int load_templates(std::vector<shape_template_t> &templates, std::string path)
{
    FILE *fp_feature;
    FILE *fp_info;
    DynamicJsonDocument doc(1024 * 8);
	
    fp_feature = fopen(std::string(path + "/shape_matching_templates.dat").c_str(), "rb");
    if(fp_feature == NULL)
	{
        doc["web"] = 1;
        doc["error"] = "can not load features";
        sendPIPEJsonDoc(doc);
		perror("open file error:");
		return 0;
	}

    fp_info = fopen(std::string(path + "/shape_matching_info.json").c_str(), "r");
    if(fp_info == NULL)
	{
        doc["web"] = 1;
        doc["error"] = "can not load features";
        sendPIPEJsonDoc(doc);
		perror("open file error:");
		return 0;
	}

    fseek(fp_info, SEEK_SET, SEEK_END);  
    int size = ftell(fp_info);
    char buffer[size];

    rewind(fp_info);
    fread(buffer, size, 1, fp_info);

    std::string payload(buffer);
    deserializeJson(doc, payload);
    doc["web"] = 1;

    int shape_num = doc["shape_num"].as<int>();

    templates.clear();
    templates.resize(shape_num);
    // fprintf(stderr, "shape_num = %d, templates size = %d\n", shape_num, templates.size());
    for(int i = 0; i < shape_num; i++)
    {
        templates[i].name = doc["shapes"][i]["name"].as<std::string>();
        int point_num = doc["shapes"][i]["point_num"].as<int>();
        templates[i].contour.resize(point_num);
        for(int j = 0; j < point_num; j++)
        {
            int x, y;
            fread(&x, sizeof(int), 1, fp_feature);
            fread(&y, sizeof(int), 1, fp_feature);
            // fprintf(stderr, "%d, %d\n", x, y);
            templates[i].contour[j] = cv::Point(x, y);
        }
    }

    sendPIPEJsonDoc(doc);

    fclose(fp_feature);
    fclose(fp_info);

    fprintf(stderr, "loaded %d shape(s).\n", templates.size());

    return 1;
}

std::vector<cv::Point> extractShape(std::string path)
{
    cv::Mat mat_in = imread(path);
    std::vector<cv::Point> approx;
    Mat mat_mask, mat_gray;
    cvtColor(mat_in, mat_gray, COLOR_BGR2GRAY);
    threshold(mat_gray, mat_mask, 128, 255, THRESH_BINARY);
    bitwise_not(mat_mask, mat_mask);

    cv::Point2f center;
    float radius;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> max_contour;
    std::vector<cv::Vec4i> hireachy;
    cv::findContours(mat_mask, contours, hireachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));

    if(contours.size() == 0)
    {
        return approx;
    }

    double max_area = 0;
    int idx = 0;
    for(int i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if(area > max_area)
        {
            max_area = area;
            idx = i;
        }
    }

    // std::vector<cv::Point> hull(contours[idx].size());
    // convexHull(contours[idx], hull);

    double peri = arcLength(contours[idx], true);
    approxPolyDP(contours[idx], approx, 0.01 * peri, true); // 0.01~0.05

    // fprintf(stderr, "num_of_vertices  = %d\n", approx.size());
    // for(int i = 0; i < approx.size(); i++)
    // {
    //     cv::circle(mat_in, cv::Point(approx[i].x, approx[i].y), 2, cv::Scalar(0, 0, 255), -1);
    // }

    // imwrite("result.png", mat_in);

    return approx;
}

std::vector<float> matchShape(std::vector<shape_template_t> &templates, std::vector<cv::Point> &contour)
{
    std::vector<float> score(templates.size());
    for(int i = 0; i < templates.size(); i++)
    {
        double temp = cv::matchShapes(contour, templates[i].contour, CONTOURS_MATCH_I3, 1);
        temp = 100 - (temp * 100);
        if(temp < 0)
        {
            temp = 0;
        }
        if(temp > 100)
        {
            temp = 100;
        }
        score[i] = temp;
    }
    return score;
}

int main(int argc, char **argv)
{
    // shape_template_t shape;
    // shape.contour = extractShape(argv[1]);
    // shape.name = "arrow";
    // shape_templates.push_back(shape);
    // shape.contour = extractShape(argv[2]);
    // shape.name = "hourglass";
    // shape_templates.push_back(shape);
    // save_templates(shape_templates, "./data");
    // return 0;

    startFramework("Shape Matching", 640, 480);

    load_templates(shape_templates, "./data");

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
        GaussianBlur(mat_div_gray, mat_div_gray, Size(5, 5), 0);
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
        GaussianBlur(mat_div_gray, mat_div_gray, Size(5, 5), 0);

        
        std::string payload;
        if(readPIPE(payload))
        {
            try
            {
                DynamicJsonDocument doc(4096);
                deserializeJson(doc, payload);
                if((doc["config"] == "web_update_data") || (doc["config"] == getFunctionName()))
                {
                    if(doc["operation"] == "update")
                    {
                        is_ref_need_update = true;
                    }
                    else if(doc["operation"] == "addshape")
                    {
                        shape_template_t shape;
                        shape.contour = extractShape("./uploads/temp/shape.png");
                        shape.name = doc["name"].as<std::string>();
                        shape_templates.push_back(shape);
                        sendPIPEMessage("Shape added.");
                        save_templates(shape_templates, "./data");
                    }
                    else if(doc["operation"] == "reset")
                    {
                        shape_templates.clear();
                    }
                }
            }
            catch(...)
            {
                fprintf(stderr, "[ERROR] Can not parse json.");
            }
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
        // dilate(mat_mask, mat_mask, Mat(), Point(-1,-1), 2);

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
                float max_score = 0;
                int max_idx = -1;

                if (area > 50)
                {
                    std::vector<cv::Point> hull, approx;
                    // convexHull(contours[i], hull);

                    double peri = arcLength(contours[i], true);
                    approxPolyDP(contours[i], approx, 0.01 * peri, true); // 0.01~0.05
                    // approx= contours[i];

                    RotatedRect rotate_area = minAreaRect(approx);
                    Rect bbox = boundingRect(approx);

                    if(shape_templates.size() > 0)
                    {
                        std::vector<float> scores = matchShape(shape_templates, approx);
                        for(int j = 0; j < scores.size(); j++)
                        {
                            if(scores[j] > max_score)
                            {
                                max_score = scores[j];
                                max_idx = j;
                            }
                        }
                    }    

                    Point2f vertices[4];
                    rotate_area.points(vertices);                

                // #define LOCAL_RENDER
                #ifdef LOCAL_RENDER
                    // rectangle(mat_div, bbox, Scalar( 255, 255, 255 ), 2, 1 );
                    for (int k = 0; k < 4; k++)
                        line(mat_div, vertices[k], vertices[(k+1)%4], Scalar(0,255,255), 2);
                    for (int k = 1; k < approx.size(); k++)
                        line(mat_div, approx[k-1], approx[k], Scalar(0,255,0), 2);
                    line(mat_div, approx[approx.size() - 1], approx[0], Scalar(0,255,0), 2);
                    char str[20];
                    // drawContours(mat_div, contours, i, Scalar(0, 255, 0), FILLED, 8, hireachy);
                    if(max_idx != -1)
                    {
                        if(max_score > 30)
                        {
                            doc["shape"][valid_cnt]["name"] = shape_templates[max_idx].name;
                            doc["shape"][valid_cnt]["max_score"] = max_score;
                            sprintf(str, "%s, %.1f%%", shape_templates[max_idx].name.c_str(), max_score);
                            
                            putText(mat_div, str, rotate_area.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
                        }
                        else
                        {
                            doc["shape"][valid_cnt]["name"] = "unidentified";
                            putText(mat_div, "unidentified", rotate_area.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
                        }
                    }
                #else
                    for (int k = 0; k < 4; k++)
                        line(mat_div, vertices[k], vertices[(k+1)%4], Scalar(0,255,255), 2);
                    for (int k = 1; k < approx.size(); k++)
                        line(mat_div, approx[k-1], approx[k], Scalar(0,255,0), 2);
                    line(mat_div, approx[approx.size() - 1], approx[0], Scalar(0,255,0), 2);
                    // RD_addPolygon(approx, "#00CD00", IMAGE_DIV);
                    // for (int k = 0; k < 4; k++)
                    //     RD_addLine(vertices[k].x, vertices[(k+1)%4].x, vertices[k].y, vertices[(k+1)%4].y, 2, "#CDCD00", IMAGE_DIV);
                    if(max_idx != -1)
                    {
                        char str[20];
                        if(max_score > 60)
                        {
                            sprintf(str, "%s, %.1f%%", shape_templates[max_idx].name.c_str(), max_score);
                            RD_addString(rotate_area.center.x, rotate_area.center.y, str, "#CC0000", IMAGE_DIV);
                            doc["shape"][valid_cnt]["name"] = shape_templates[max_idx].name;
                            doc["shape"][valid_cnt]["max_score"] = max_score;
                        }
                        else
                        {
                            RD_addString(rotate_area.center.x, rotate_area.center.y, "unidentified", "#CC0000", IMAGE_DIV);
                            doc["shape"][valid_cnt]["name"] = "unidentified";
                        }
                    }
                    
                    // RD_addRectangle(bbox.x, bbox.y, bbox.width, bbox.height, "#00CD00", IMAGE_DIV);
                #endif
                    for(int k = 0; k < contours[i].size(); k++)
                    {
                        doc["shape"][valid_cnt]["edge"][k]["x"] = contours[i][k].x;
                        doc["shape"][valid_cnt]["edge"][k]["y"] = contours[i][k].y;
                    }
                    doc["shape"][valid_cnt]["x"] = bbox.x * IMAGE_DIV;
                    doc["shape"][valid_cnt]["y"] = bbox.y * IMAGE_DIV;
                    doc["shape"][valid_cnt]["w"] = bbox.width * IMAGE_DIV;
                    doc["shape"][valid_cnt]["h"] = bbox.height * IMAGE_DIV;
                    doc["shape"][valid_cnt]["area"] = area;
                    valid_cnt++;
                }
            }

            doc["num"] = valid_cnt;
            sendPIPEJsonDoc(doc);
        }
        else
        {
            RD_clear();
        }
        // sendMat(mat_mask);
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

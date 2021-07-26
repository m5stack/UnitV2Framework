#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "framework.h"

#define IMAGE_DIV 2

using namespace cv;

int main( int argc, char** argv )
{
    startFramework("Target Tracker", 640, 480);

    bool is_tracker_init = false;
    Rect2d track_roi;

    // Ptr<Tracker> tracker = TrackerKCF::create();
    Ptr<Tracker> tracker = cv::TrackerMOSSE::create();
    

    while(1)
    {
        double t_start = ncnn::get_current_time();
        Mat mat_src;
        getMat(mat_src);

        cv::Mat mat_div;
        cv::resize(mat_src, mat_div, cv::Size(), 1.0f / IMAGE_DIV, 1.0f / IMAGE_DIV, cv::INTER_AREA);

        std::string payload;
        if(readPIPE(payload))
        {
            try
            {
                DynamicJsonDocument doc(4096);
                deserializeJson(doc, payload);
                if(doc["config"] == "web_update_data" || doc["config"] == getFunctionName())
                {
                    track_roi.x = (double)(doc["x"].as<int>() / IMAGE_DIV);
                    track_roi.y = (double)(doc["y"].as<int>() / IMAGE_DIV);
                    track_roi.width = (double)(doc["w"].as<int>() / IMAGE_DIV);
                    track_roi.height = (double)(doc["h"].as<int>() / IMAGE_DIV);
                    fprintf(stderr, "update roi  x: %.0f, y: %.0f, w: %.0f, h:%.0f\n", track_roi.x, track_roi.y, track_roi.width, track_roi.height);

                    if(track_roi.width == 0 || track_roi.height == 0)
                    {
                        continue;
                    }
                    else
                    {
                        if(is_tracker_init == true)
                        {
                            tracker->clear();
                            tracker = cv::TrackerMOSSE::create();
                        }
                        tracker->init(mat_div, track_roi);
                        is_tracker_init = true;
                        sendPIPEMessage("ROI updated.");
                        continue;
                    }
                }
            }
            catch(...)
            {
                fprintf(stderr, "[ERROR] Can not parse json.");
            }
        }

        if(is_tracker_init)
        {
            bool result = tracker->update(mat_div, track_roi);
            
            if(result == true)
            {
                if(isStreamOpend())
                {
                #ifdef LOCAL_RENDER
                    rectangle(mat_div, track_roi, Scalar( 0, 255, 0 ), 2, 1 );
                #else
                    RD_addRectangle(track_roi.x, track_roi.y, track_roi.width, track_roi.height, "#00CD00", IMAGE_DIV);
                #endif
                }

                DynamicJsonDocument doc(1024);

                doc["x"] = track_roi.x * IMAGE_DIV;
                doc["y"] = track_roi.y * IMAGE_DIV;
                doc["w"] = track_roi.width * IMAGE_DIV;
                doc["h"] = track_roi.height * IMAGE_DIV;

                sendPIPEJsonDoc(doc);
            }
            // else
            // {
            //     RD_clear();
            // }
        }
        
        sendMat(mat_div);

        double t_end = ncnn::get_current_time();
        double dt_total = t_end - t_start;
        fprintf(stderr, "total %3.1f\n", dt_total);
    }
    return 0;
}
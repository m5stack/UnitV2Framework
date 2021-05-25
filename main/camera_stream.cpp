#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "framework.h"

using namespace cv;

#define IMAGE_DIV 2

int main( int argc, char** argv )
{
    startFramework("Camera Stream", 640, 480);
    int low_quality_mode = false;

    while(1)
    {
        Mat mat_src;
        getMat(mat_src);

        if(low_quality_mode == true)
        {
            resize(mat_src, mat_src, Size(), 1.0f / IMAGE_DIV, 1.0f / IMAGE_DIV, INTER_AREA);
        }

        std::string payload;
        if(readPIPE(payload))
        {
            try
            {
                DynamicJsonDocument doc(1024);
                deserializeJson(doc, payload);
                if((doc["config"] == "web_update_data") || (doc["config"] == getFunctionName()))
                {
                    if(doc["operation"] == "enable")
                    {
                        low_quality_mode = true;
                        sendPIPEMessage("Low Quality Mode Enabled.");
                    }
                    else
                    {
                        low_quality_mode = false;
                        sendPIPEMessage("Low Quality Mode Disabled.");
                    }
                }
            }
            catch(...)
            {
                fprintf(stderr, "[ERROR] Can not parse json.");
            }
        }

        sendMat(mat_src);
    }
    
    return 0;
}
#include "framework.h"

// #define CAPTURE_USE_THREAD

std::queue<cv::Mat> queue_input_mat;
std::mutex queue_input_mat_lock;

std::queue<cv::Mat> queue_output_mat;
std::mutex queue_output_mat_lock;

std::queue<std::string> queue_recive_pipe;
std::mutex queue_recive_pipe_lock;

std::queue<std::string> queue_sending_pipe;
std::mutex queue_sending_pipe_lock;

std::mutex global_lock;
bool global_stream_enable = false;

int check_failed = -1;

cv::VideoCapture capture(0);
std::string running_function_name;

StaticJsonDocument<4096> render_doc;
int render_doc_item_cnt = 0;
bool is_render_need_clear = false;

std::string getFunctionName(void)
{
    return running_function_name;
}

void sendPIPEMessage(std::string msg)
{
    DynamicJsonDocument doc(4096);

    doc["msg"] = msg;

    sendPIPEJsonDoc(doc);
}

void sendPIPEJsonDoc(DynamicJsonDocument &doc)
{
    doc["running"] = running_function_name;
    std::string payload;
    serializeJson(doc, payload);
    // if(doc.containsKey("debug"))
    // {
    //     fprintf(stderr, "send %s\n", payload.c_str());
    // }
    sendPIPE(payload);
    payload.clear();
}

void startFramework(const char* function_name, int w, int h, bool disable_camera)
{
    running_function_name = function_name;
    fprintf(stderr, "function: %s start.\n", function_name);

    setvbuf(stdout, NULL, _IONBF, 0);

    // check_failed = checkDevice("01fc:220b");
    check_failed = false;
    fprintf(stderr, "Copyright 2021 M5Stack Technology Co., Ltd. All rights reserved.\n");
    // check_failed |= checkDevice("01fd:220a");
    // fprintf(stderr, "------------------------------------------------\n ");
    // fprintf(stderr, "------------------------------------------------\n ");
    // fprintf(stderr, "------------------------------------------------\n ");
    // fprintf(stderr, "------------------------------------------------\n ");
    // fprintf(stderr, "\n!!! Hardware authentication has been closed !!!\n ");
    // fprintf(stderr, "------------------------------------------------\n ");
    // fprintf(stderr, "------------------------------------------------\n ");
    // fprintf(stderr, "------------------------------------------------\n ");
    // fprintf(stderr, "------------------------------------------------\n ");
    // check_failed = false;

    if(disable_camera == false)
    {
        if (!capture.isOpened())
        {
            std::cout << "Cannot open cam" << std::endl;
            return;
        }
        // capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
        capture.set(cv::CAP_PROP_FPS, 30);
        capture.set(cv::CAP_PROP_FOURCC, cv::CAP_OPENCV_MJPEG);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, w);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, h);

        #ifdef CAPTURE_USE_THREAD
            std::thread thread_capture(CaptureThread);
            thread_capture.detach();
        #endif
    }

    std::thread thread_sending(SendingThread);
    std::thread thread_piperead(PIPEReadThread);
    std::thread thread_pipesend(PIPESendThread);
    
    thread_sending.detach();
    thread_piperead.detach();
    thread_pipesend.detach();

    global_stream_enable = false;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    char buf[256];
    sprintf(buf, "Running %s, Copyright 2021 M5Stack Technology Co., Ltd. All rights reserved.", function_name);
    sendPIPEMessage(buf);
}

void enableStream()
{
    global_stream_enable = true;
}

void disableStream()
{
    global_stream_enable = false;
}


int MatToJpgByteArray(const cv::Mat mat, std::vector<unsigned char> &buff)
{
    if (mat.empty())
    {
        return 0;
    }
    std::vector<int> param = std::vector<int>(2);
    param[0] = cv::IMWRITE_JPEG_QUALITY;
    param[1] = 90;
    cv::imencode(".jpg", mat, buff, param);
    return 0;
}

void CaptureThread(void)
{
    perror("Capture thread start\n");

    while (1)
    {
        cv::Mat img;
        capture >> img;
        queue_input_mat_lock.lock();
        queue_input_mat.push(img);
        queue_input_mat_lock.unlock();
        // sleep(0.001);
    }
}

void SendingThread(void)
{
    perror("Sending thread start\n");

    while (1)
    {
        // global_lock.lock();
        if(global_stream_enable == false)
        {
            // global_lock.unlock();
            sleep(1);
            continue;
        }
        // global_lock.unlock();

        cv::Mat image;
        while (1)
        {
            if (queue_output_mat.empty())
            {
                sleep(0.001);
                continue;
            }
            else
            {
                // queue_output_mat_lock.lock();
                image = queue_output_mat.front();
                queue_output_mat.pop();
                // queue_output_mat_lock.unlock();
                break;
            }
        }
        std::vector<unsigned char> buf_jpg;
        MatToJpgByteArray(image, buf_jpg);
        std::string payload = "{\"img\":\"";
        payload += base64_encode(buf_jpg.data(), buf_jpg.size());
        payload += "\"}";
        sendPIPE(payload);
    }
}

void PIPEReadThread(void)
{
    perror("PIPERecv thread start\n");

    while(1)
    {
        char buffer[512];
        fgets(buffer, 512, stdin);
        std::string payload(buffer);

        
        if(payload[0] != '_')
        {
            queue_recive_pipe_lock.lock();
            queue_recive_pipe.push(payload);
            queue_recive_pipe_lock.unlock();
            continue;
        }

        DynamicJsonDocument doc(1024);
        deserializeJson(doc, payload.c_str() + 1);

        if(doc.containsKey("stream"))
        {
            if(doc["stream"] == 1)
            {
                // fprintf(stderr, "sys: stream enabled.\n");
                global_lock.lock();
                global_stream_enable = true;
                global_lock.unlock();
            }
            else
            {
                // fprintf(stderr, "sys: stream disabled.\n");
                global_lock.lock();
                global_stream_enable = false;
                global_lock.unlock();

                queue_output_mat_lock.lock();
                while(1)
                {
                    if(queue_output_mat.empty())
                    {
                        break;
                    }
                    queue_output_mat.pop();
                }
                queue_output_mat_lock.unlock();
            }
        }
    }
}

void PIPESendThread(void)
{
    perror("PIPESend thread start\n");

    while(1)
    {
        if(!queue_sending_pipe.empty())
        {
            std::string payload = queue_sending_pipe.front();
            // fprintf(stderr, "payload size = %d\n", payload.length());
            // if(payload.length() < 5000)
            // {
            //     fprintf(stderr, "PIPE SEND: %s", payload.c_str());
            // }
            fputs(payload.c_str(), stdout);
            queue_sending_pipe.pop();
        }
        else
        {
            sleep(0.001);
        }
        
        // queue_sending_pipe_lock.lock();
        // while(1)
        // {
        //     if(queue_sending_pipe.empty())
        //     {
        //         break;
        //     }
        //     queue_sending_pipe.pop();
        // }
        // queue_sending_pipe_lock.unlock();
    }
}


void getMat(cv::Mat &image)
{
    #ifdef CAPTURE_USE_THREAD
        while (1)
        {
            if (queue_input_mat.empty())
            {
                sleep(0.001);
                continue;
            }

            queue_input_mat_lock.lock();
            image = queue_input_mat.back();
            while (!queue_input_mat.empty())
            {
                queue_input_mat.pop();
            }
            queue_input_mat_lock.unlock();
            return;
        }
    #else
        capture >> image;
    #endif

    if(check_failed)
    {
        sleep(3);
    }
}

void sendMat(cv::Mat &image)
{
    // global_lock.lock();
    if(global_stream_enable == false)
    {
        // global_lock.unlock();
        return;
    }
    // global_lock.unlock();

    if(queue_output_mat.size() > 3)
    {
        return;
    }
    // queue_output_mat_lock.lock();
    sendRenderDoc();
    queue_output_mat.push(image);
    // fprintf(stderr, "queue_output_mat size = %d\n", queue_output_mat.size());
    // queue_output_mat_lock.unlock();
}

int readPIPE(std::string &payload)
{
    if(queue_recive_pipe.empty())
    {
        return 0;
    }

    queue_recive_pipe_lock.lock();
    payload = queue_recive_pipe.front();
    queue_recive_pipe.pop();
    queue_recive_pipe_lock.unlock();

    return 1;
}

int sendPIPE(std::string &payload)
{
    payload += "\r\n";
    if(queue_sending_pipe.size() > 10)
    {
        fprintf(stderr, "pipe busy! size = %d\n", queue_sending_pipe.size());
        return 1;
    }
    
    queue_sending_pipe.push(payload);
    return 0;
}

bool isStreamOpend(void)
{
    return global_stream_enable;
}

int checkDevice(const char* id)
{
    char buf[1024];

    FILE *fp;

    if ((fp = popen("dmesg | grep Camera", "r")) == NULL) 
    {
        return -1;
    }

    while (fgets(buf, 1024, fp) != NULL) {
        // fprintf(stderr, "OUTPUT: %s", buf);
        if(strstr(buf, id) != NULL)
        {   
            if(pclose(fp))  
            {
                return -1;
            }
            // fprintf(stderr, "Copyright 2021 M5Stack, Inc. All rights reserved.\n", buf);
            return 0;
        }
    }

    if(pclose(fp))  
    {
        return -1;
    }

    return 1;
}


int loadModel(const std::string path, ncnn::Net &net)
{
    std::string param = path + ".param";
    std::string bin = path + ".bin";
    net.load_param(param.c_str());
    net.load_model(bin.c_str());
    net.opt.openmp_blocktime = 0;
    net.opt.lightmode = true;
    net.opt.num_threads = ncnn::get_cpu_count();
    // net.opt.blob_allocator = &g_blob_pool_allocator;
    // net.opt.workspace_allocator = &g_workspace_pool_allocator;
    // net.opt.use_winograd_convolution = true;
    // net.opt.use_sgemm_convolution = true;
    // net.opt.use_int8_inference = true;
    // net.opt.use_vulkan_compute = false;
    // net.opt.use_fp16_packed = true;
    // net.opt.use_fp16_storage = true;
    // net.opt.use_fp16_arithmetic = true;
    // net.opt.use_int8_storage = true;
    // net.opt.use_int8_arithmetic = true;
    // net.opt.use_packing_layout = true;
    // net.opt.use_shader_pack8 = false;
    // net.opt.use_image_storage = false;
    // net.opt.use_bf16_storage = false;

    return 0;
}

int loadModel(const std::string pathparam, const std::string pathbin, ncnn::Net &net)
{
    net.load_param(pathparam.c_str());
    net.load_model(pathbin.c_str());
    net.opt.openmp_blocktime = 0;
    net.opt.lightmode = true;
    net.opt.num_threads = ncnn::get_cpu_count();
}

void resetAllocator(void)
{
    // g_blob_pool_allocator.clear();
    // g_workspace_pool_allocator.clear();
}


void resetRenderDoc(void)
{
    render_doc.clear();
    render_doc["render"] = 1;
    render_doc_item_cnt = 0;
}

void RD_addRectangle(float x, float y, float w, float h, std::string color, int multiple)
{
    RD_addRectangle((int)x * multiple, (int)y * multiple, (int)w * multiple, (int)h * multiple, color);
}

void RD_addCircle(float x, float y, float r, std::string color, int multiple)
{
    RD_addCircle((int)x * multiple, (int)y * multiple, (int)r * multiple, color);
}

void RD_addString(float x, float y, std::string payload, std::string color, int multiple)
{
    RD_addString((int)x * multiple, (int)y * multiple, payload, color);
}

void RD_addPoint(float x, float y, std::string color, int multiple)
{
    RD_addPoint((int)x * multiple, (int)y * multiple, color);
}

void RD_addLine(float x1, float y1, float x2, float y2, int thickness, std::string color, int multiple)
{
    RD_addLine((int)x1 * multiple, (int)y1 * multiple, (int)x2 * multiple, (int)y2 * multiple, thickness, color);
}

void RD_addRectangle(double x, double y, double w, double h, std::string color, int multiple)
{
    RD_addRectangle((int)x * multiple, (int)y * multiple, (int)w * multiple, (int)h * multiple, color);
}

void RD_addCircle(double x, double y, double r, std::string color, int multiple)
{
    RD_addCircle((int)x * multiple, (int)y * multiple, (int)r, color);
}

void RD_addString(double x, double y, std::string payload, std::string color, int multiple)
{
    RD_addString((int)x * multiple, (int)y * multiple, payload, color);
}

void RD_addPoint(double x, double y, std::string color, int multiple)
{
    RD_addPoint((int)x * multiple, (int)y * multiple, color);
}

void RD_addLine(double x1, double y1, double x2, double y2, int thickness, std::string color, int multiple)
{
    RD_addLine((int)x1 * multiple, (int)y1 * multiple, (int)x2 * multiple, (int)y2 * multiple, thickness, color);
}

void RD_addRectangle(int x, int y, int w, int h, std::string color, int multiple)
{
    RD_addRectangle((int)x * multiple, (int)y * multiple, (int)w * multiple, (int)h * multiple, color);
}

void RD_addCircle(int x, int y, int r, std::string color, int multiple)
{
    RD_addCircle((int)x * multiple, (int)y * multiple, (int)r * multiple, color);
}

void RD_addString(int x, int y, std::string payload, std::string color, int multiple)
{
    RD_addString((int)x * multiple, (int)y * multiple, payload, color);
}

void RD_addPoint(int x, int y, std::string color, int multiple)
{
    RD_addPoint((int)x * multiple, (int)y * multiple, color);
}

void RD_addLine(int x1, int y1, int x2, int y2, int thickness, std::string color, int multiple)
{
    RD_addLine((int)x1 * multiple, (int)y1 * multiple, (int)x2 * multiple, (int)y2 * multiple, thickness, color);
}


void RD_addRectangle(float x, float y, float w, float h, std::string color)
{
    RD_addRectangle((int)x, (int)y, (int)w, (int)h, color);
}

void RD_addCircle(float x, float y, float r, std::string color)
{
    RD_addCircle((int)x, (int)y, (int)r, color);
}

void RD_addString(float x, float y, std::string payload, std::string color)
{
    RD_addString((int)x, (int)y, payload, color);
}

void RD_addPoint(float x, float y, std::string color)
{
    RD_addPoint((int)x, (int)y, color);
}

void RD_addLine(float x1, float y1, float x2, float y2, int thickness, std::string color)
{
    RD_addLine((int)x1, (int)y1, (int)x2, (int)y2, thickness, color);
}

void RD_addRectangle(double x, double y, double w, double h, std::string color)
{
    RD_addRectangle((int)x, (int)y, (int)w, (int)h, color);
}

void RD_addCircle(double x, double y, double r, std::string color)
{
    RD_addCircle((int)x, (int)y, (int)r, color);
}

void RD_addString(double x, double y, std::string payload, std::string color)
{
    RD_addString((int)x, (int)y, payload, color);
}

void RD_addPoint(double x, double y, std::string color)
{
    RD_addPoint((int)x, (int)y, color);
}

void RD_addLine(double x1, double y1, double x2, double y2, int thickness, std::string color)
{
    RD_addLine((int)x1, (int)y1, (int)x2, (int)y2, thickness, color);
}

void RD_addRectangle(int x, int y, int w, int h, std::string color)
{
    render_doc["items"][render_doc_item_cnt]["type"] = "rectangle";
    render_doc["items"][render_doc_item_cnt]["x1"] = x;
    render_doc["items"][render_doc_item_cnt]["y1"] = y;
    render_doc["items"][render_doc_item_cnt]["w1"] = w;
    render_doc["items"][render_doc_item_cnt]["h1"] = h;
    render_doc["items"][render_doc_item_cnt]["color"] = color;
    render_doc_item_cnt++;
}

void RD_addCircle(int x, int y, int r, std::string color)
{
    render_doc["items"][render_doc_item_cnt]["type"] = "circle";
    render_doc["items"][render_doc_item_cnt]["x1"] = x;
    render_doc["items"][render_doc_item_cnt]["y1"] = y;
    render_doc["items"][render_doc_item_cnt]["r1"] = r;
    render_doc["items"][render_doc_item_cnt]["color"] = color;
    render_doc_item_cnt++;
}

void RD_addString(int x, int y, std::string payload, std::string color)
{
    render_doc["items"][render_doc_item_cnt]["type"] = "string";
    render_doc["items"][render_doc_item_cnt]["x1"] = x;
    render_doc["items"][render_doc_item_cnt]["y1"] = y;
    render_doc["items"][render_doc_item_cnt]["payload"] = payload;
    render_doc["items"][render_doc_item_cnt]["color"] = color;
    render_doc_item_cnt++;
}

void RD_addPoint(int x, int y, std::string color)
{
    render_doc["items"][render_doc_item_cnt]["type"] = "point";
    render_doc["items"][render_doc_item_cnt]["x1"] = x;
    render_doc["items"][render_doc_item_cnt]["y1"] = y;
    render_doc["items"][render_doc_item_cnt]["color"] = color;
    render_doc_item_cnt++;
}

void RD_addLine(int x1, int y1, int x2, int y2, int thickness, std::string color)
{
    render_doc["items"][render_doc_item_cnt]["type"] = "line";
    render_doc["items"][render_doc_item_cnt]["x1"] = x1;
    render_doc["items"][render_doc_item_cnt]["y1"] = y1;
    render_doc["items"][render_doc_item_cnt]["x2"] = x2;
    render_doc["items"][render_doc_item_cnt]["y2"] = y2;
    render_doc["items"][render_doc_item_cnt]["thickness"] = thickness;
    render_doc["items"][render_doc_item_cnt]["color"] = color;
    render_doc_item_cnt++;
}

void RD_addPolygon(std::vector<cv::Point> &points, std::string color)
{
    render_doc["items"][render_doc_item_cnt]["type"] = "polygon";
    render_doc["items"][render_doc_item_cnt]["color"] = color;
    for(size_t i = 0; i < points.size(); i++)
    {
        render_doc["items"][render_doc_item_cnt]["x"][i] = points.at(i).x;
        render_doc["items"][render_doc_item_cnt]["y"][i] = points.at(i).y;
    }
    render_doc_item_cnt++;
}

void RD_addPolygon(std::vector<cv::Point> &points, std::string color, int multiple)
{
    render_doc["items"][render_doc_item_cnt]["type"] = "polygon";
    render_doc["items"][render_doc_item_cnt]["color"] = color;
    for(size_t i = 0; i < points.size(); i++)
    {
        render_doc["items"][render_doc_item_cnt]["x"][i] = points.at(i).x * multiple;
        render_doc["items"][render_doc_item_cnt]["y"][i] = points.at(i).y * multiple;
    }
    render_doc_item_cnt++;
}

void RD_clear(void)
{
    if(is_render_need_clear)
    {
        is_render_need_clear = false;
        render_doc.clear();
        render_doc["render"] = 0;
        render_doc_item_cnt = 0;
        std::string payload;
        serializeJson(render_doc, payload);
        sendPIPE(payload);
        payload.clear();
        resetRenderDoc();
    }
}

void sendRenderDoc(void)
{
    if(render_doc_item_cnt != 0)
    {
        is_render_need_clear = true;
        std::string payload;
        serializeJson(render_doc, payload);
        sendPIPE(payload);
        payload.clear();
        resetRenderDoc();
    }
}


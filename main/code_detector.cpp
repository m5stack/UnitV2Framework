#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "zbar.h"
#include "net.h"
#include "framework.h"
#include "benchmark.h"

#define IMAGE_DIV 2

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

cv::Mat srnet(const cv::Mat &bgr)
{
    ncnn::Net ncnn_net;

    cv::imwrite("srnet_src.jpg", bgr);

    ncnn_net.load_param("models/srnet.param");
    ncnn_net.load_model("models/srnet.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    ncnn::Extractor ex = ncnn_net.create_extractor();

    ex.input("data", in);
    ncnn::Mat picture_out;
    ex.extract("fc", picture_out);

    cv::Mat a(picture_out.h, picture_out.w, CV_8UC3);
    picture_out.to_pixels(a.data, ncnn::Mat::PIXEL_BGR2RGB);

    cv::imwrite("srnet_result.jpg", a);

    return a;
}

int idx = 0;
static std::string zbar_read(cv::Mat src)
{
    std::string ret;
    cv::Mat gray;

    if (src.channels() == 1)
        gray = src;
    else
        cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);

    // cv::threshold(gray, gray, 0, 255, cv::THRESH_OTSU);
    int width = gray.cols;
    int height = gray.rows;
    // create a reader
    zbar::ImageScanner scanner;
    // configure the reader
    scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 0);
    scanner.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
    // scanner.set_config(zbar::ZBAR_CODE128, zbar::ZBAR_CFG_ENABLE, 1);
    unsigned char *pdata = (unsigned char *)gray.data;
    zbar::Image imageZbar(width, height, "Y800", pdata, width * height);
    int n = scanner.scan(imageZbar);
    // fprintf(stderr, "n = %d\n", n);
    if (n > 0)
    {

        // extract results
        for (zbar::Image::SymbolIterator symbol = imageZbar.symbol_begin();
             symbol != imageZbar.symbol_end();
             ++symbol)
        {
            // do something useful with results
            std::string decodedFmt = symbol->get_type_name();
            std::string symbolData = symbol->get_data();

            ret = std::string(symbolData);

            // std::cout << decodedFmt << std::endl;
            // std::cout << symbolData << std::endl;
        }
    }
    else
    {
        // printf("Not got a barcode!\n");
        // char error_str[64];
        // sprintf(error_str, "Not got a barcode!");
        // std::cout << "Not got a barcode!" << std::endl;
    }
    // clean up
    imageZbar.set_data(NULL, 0);
    char buf[100];
    return ret;
}

static int nn_detect(const cv::Mat &bgr, ncnn::Net &net, std::vector<Object> &objects)
{
    resetAllocator();
    const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);

    //     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float *values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static void read_objects(const cv::Mat &src, const std::vector<Object> &objects, cv::Mat &image)
{
    static const char *kClassNames[] = {"Background", "QR/DM/Maxi", "SmallProgramCode", "PDF-417", "EAN", "Unknown"};

    if(objects.size() == 0)
    {
        RD_clear();
        return;
    }

    DynamicJsonDocument doc(1024 * 16);

    int code_num = 0;
    for (size_t i = 0; i < objects.size(); i++)
    {
        Object obj = objects[i];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        std::string content;
        if (obj.prob > 0.8)
        {
            int rw = (int)obj.rect.width * 1.5;
            int rh = (int)obj.rect.height * 1.5;
            int offsetw = (rw - obj.rect.width) / 2;
            int offseth = (rh - obj.rect.height) / 2;
            int rx = (int)obj.rect.x - offsetw;
            int ry = (int)obj.rect.y - offseth;
            if (rx < 0)
            {
                rx = 0;
            }
            if (ry < 0)
            {
                ry = 0;
            }
            if (rx + rw > src.cols)
            {
                rw = src.cols - rx;
            }
            if (ry + rh > src.rows)
            {
                rh = src.rows - ry;
            }
            cv::Rect roi(rx, ry, rw, rh);
            cv::Mat cut = src(roi).clone();
            // cut = srnet(cut);
            content = zbar_read(cut);
            fprintf(stderr, "content = %s\n", content.c_str());

            doc["code"][code_num]["prob"] = obj.prob;
            doc["code"][code_num]["x"] = (int)obj.rect.x;
            doc["code"][code_num]["y"] = (int)obj.rect.y;
            doc["code"][code_num]["w"] = (int)obj.rect.width;
            doc["code"][code_num]["h"] = (int)obj.rect.height;
            doc["code"][code_num]["type"] = kClassNames[obj.label];
            doc["code"][code_num]["content"] = content;

            code_num++;
        }
        doc["num"] = code_num;

        if(isStreamOpend())
        {
        #ifdef LOCAL_RENDER
            obj.rect.x /= IMAGE_DIV;
            obj.rect.y /= IMAGE_DIV;
            obj.rect.width /= IMAGE_DIV;
            obj.rect.height /= IMAGE_DIV;

            cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0), 4);

            char src[256];
            sprintf(src, "%s %.1f%%, %s", kClassNames[obj.label], obj.prob * 100, content.c_str());

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(src, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            label_size.height += 10;

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height)),
                        cv::Scalar(0, 255, 0), -1);

            cv::putText(image, src, cv::Point(x, y + label_size.height - 7),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        #else
            char src[256];
            sprintf(src, "%s %.1f%%, %s", kClassNames[obj.label], obj.prob * 100, content.c_str());

            RD_addRectangle(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, "#00CD00");
            RD_addString(obj.rect.x, obj.rect.y, src, "#00CD00");
        #endif
        }
    }

    if(code_num != 0)
    {
        sendPIPEJsonDoc(doc);
    }
}

int main(int argc, char **argv)
{
    startFramework("Code Detector", 640, 480);

    ncnn::Net net;
    loadModel("./models/mobilenet_ssd_code_detector_ncnn", net);

    std::vector<Object> objects;
    while (1)
    {
        double t_start = ncnn::get_current_time();
        cv::Mat mat_src;
        getMat(mat_src);

        nn_detect(mat_src, net, objects);


        cv::Mat mat_div;
        if(isStreamOpend())
        {
            cv::resize(mat_src, mat_div, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        }

        read_objects(mat_src, objects, mat_div);

        sendMat(mat_div);

        double t_end = ncnn::get_current_time();
        double dt_total = t_end - t_start;
        fprintf(stderr, "total %3.1f\n", dt_total);
    }
    return 0;
}

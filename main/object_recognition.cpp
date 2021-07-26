#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "framework.h"
#include <sys/stat.h>
#include <dirent.h>

#define IMAGE_DIV 2

enum
{
    MODEL_TYPE_NONE = 0,
    MODEL_TYPE_YOLOFASTEST,
    MODEL_TYPE_NANODET
};

std::string config_model_path = "./uploads/models/yolo_20class";
int config_model_type = MODEL_TYPE_NONE;
std::string config_model_param_path;
std::string config_model_bin_path;
std::string config_model_input_node;
std::vector<std::string> config_model_output_nodes;
int config_num_of_class;
std::vector<std::string> config_class_names;
bool config_load_success = false;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

#include <sys/stat.h>
#include <dirent.h>

int loadModelConfig(std::string path)
{
    #ifdef DEBUG
    fprintf(stderr, "%s List Dir: %s\n", __func__, path.c_str());
    struct stat dir_stat;
	lstat(path.c_str(), &dir_stat);
    struct dirent *filename;
 	DIR * dir;
	dir = opendir(path.c_str());
    while(( filename = readdir(dir)) != NULL )
	{
        fprintf(stderr, "%s\n", filename->d_name);
	}
    #endif

    if(path[path.length() - 1] != '/')
    {
        path += "/";
    }

    FILE *fp_info;
    DynamicJsonDocument doc(1024 * 8);

    fp_info = fopen(std::string(path + "model.json").c_str(), "r");
    if(fp_info == NULL)
	{
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

    std::string model_type = doc["modelType"].as<std::string>();
    if(model_type == "yolov3")
    {
        config_model_type = MODEL_TYPE_YOLOFASTEST;
        fprintf(stderr, "Model Type: YOLO v3\n");
    }
    else if(model_type == "nanodet")
    {
        config_model_type = MODEL_TYPE_NANODET;
        fprintf(stderr, "Model Type: NanoDet\n");
    }
    else
    {
        config_model_type = MODEL_TYPE_NONE;
        fprintf(stderr, "ERROR: Unknown model\n");
        return 0;
    }
    
    config_model_param_path = path + doc["modelParamName"].as<std::string>();
    config_model_bin_path = path + doc["modelBinName"].as<std::string>();
    if(config_model_type == MODEL_TYPE_YOLOFASTEST)
    {
        config_model_input_node = "data";
        config_model_output_nodes.push_back(doc["model"]["modelOutputNodes"][0]["Yolov3DetectionOutput"].as<std::string>());
    }
    else if(config_model_type == MODEL_TYPE_NANODET)
    {
        config_model_input_node = "input.1";
        for(int i = 0; i < 3; i++)
        {
            config_model_output_nodes.push_back(doc["model"]["modelOutputNodes"][i]["cls_pred"].as<std::string>());
            config_model_output_nodes.push_back(doc["model"]["modelOutputNodes"][i]["dis_pred"].as<std::string>());
        }
    }
    
    config_num_of_class = doc["numOfClass"].as<int>();

    // config_class_names.push_back("null");
    for(int i = 0; i < config_num_of_class; i++)
    {
        config_class_names.push_back(doc["className"][i].as<std::string>());
    }

    fprintf(stderr, "Load %s Successfull, Class: %d\n", path.c_str(), config_num_of_class);

    config_load_success = true;

    fclose(fp_info);
    return 1;
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_proposals(const ncnn::Mat& cls_pred, const ncnn::Mat& dis_pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = cls_pred.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = cls_pred.w;
    const int reg_max_1 = dis_pred.w / 4;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            const int idx = i * num_grid_x + j;

            const float* scores = cls_pred.row(idx);

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                if (scores[k] > score)
                {
                    label = k;
                    score = scores[k];
                }
            }

            if (score >= prob_threshold)
            {
                ncnn::Mat bbox_pred(reg_max_1, 4, (void*)dis_pred.row(idx));
                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(bbox_pred, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (j + 0.5f) * stride;
                float pb_cy = (i + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

static int detect_nanodet(const cv::Mat& bgr, ncnn::Net &nanodet, std::vector<Object>& objects)
{
    int width = bgr.cols;
    int height = bgr.rows;

    const int target_size = 320;
    const float prob_threshold = 0.4f;
    const float nms_threshold = 0.5f;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = nanodet.create_extractor();

    ex.input(config_model_input_node.c_str(), in_pad);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract(config_model_output_nodes[0].c_str(), cls_pred);
        ex.extract(config_model_output_nodes[1].c_str(), dis_pred);

        std::vector<Object> objects8;
        generate_proposals(cls_pred, dis_pred, 8, in_pad, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract(config_model_output_nodes[2].c_str(), cls_pred);
        ex.extract(config_model_output_nodes[3].c_str(), dis_pred);

        std::vector<Object> objects16;
        generate_proposals(cls_pred, dis_pred, 16, in_pad, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract(config_model_output_nodes[4].c_str(), cls_pred);
        ex.extract(config_model_output_nodes[5].c_str(), dis_pred);

        std::vector<Object> objects32;
        generate_proposals(cls_pred, dis_pred, 32, in_pad, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

int detect_yolofastest(cv::Mat &image, ncnn::Net &detector, std::vector<Object> &objects, int detector_size_width, int detector_size_height)
{
    resetAllocator();

    cv::Mat bgr = image.clone();
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                 bgr.cols, bgr.rows, detector_size_width, detector_size_height);

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = detector.create_extractor();
    // ex.set_num_threads(8);
    ex.input(config_model_input_node.c_str(), in);
    ncnn::Mat out;
    ex.extract(config_model_output_nodes[0].c_str(), out);

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

static void draw_objects(const std::vector<Object> &objects, cv::Mat &image)
{
    if(objects.size() == 0)
    {
        RD_clear();
        return;
    }
    DynamicJsonDocument doc(1024 * 16);

    doc["num"] = objects.size();
    for (size_t i = 0; i < objects.size(); i++)
    {
        Object obj = objects[i];

        doc["obj"][i]["prob"] = obj.prob;
        doc["obj"][i]["x"] = (int)obj.rect.x;
        doc["obj"][i]["y"] = (int)obj.rect.y;
        doc["obj"][i]["w"] = (int)obj.rect.width;
        doc["obj"][i]["h"] = (int)obj.rect.height;
        doc["obj"][i]["type"] = config_class_names.at(obj.label);

        if(isStreamOpend())
        {
            obj.rect.x /= IMAGE_DIV;
            obj.rect.y /= IMAGE_DIV;
            obj.rect.width /= IMAGE_DIV;
            obj.rect.height /= IMAGE_DIV;

            std::string content;

        #ifdef LOCAL_RENDER
            cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0), 4);
        #else
            RD_addRectangle(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, "#00CD00", IMAGE_DIV);
        #endif

            char src[256];
            sprintf(src, "%s %.1f%%", config_class_names.at(obj.label).c_str(), obj.prob * 100);

        #ifdef LOCAL_RENDER
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
            RD_addString(obj.rect.x, obj.rect.y, src, "#00CD00", IMAGE_DIV);
        #endif
        }
    }

    sendPIPEJsonDoc(doc);
}

std::vector<std::string> checkModels(std::string path)
{
    std::vector<std::string> files_name;

    struct stat dir_stat;
	lstat(path.c_str(), &dir_stat);
    struct dirent *filename;
 	DIR * dir;
	dir = opendir(path.c_str());
    while(( filename = readdir(dir)) != NULL )
	{
        if(strstr(filename->d_name, ".bin") == NULL)
        {
            continue;
        }
        std::string modelname(filename->d_name);
        modelname = modelname.substr(0, modelname.find("."));
        files_name.push_back(modelname);
	}
    return files_name;
}

int main(int argc, char **argv)
{
    startFramework("Object Recognition", 640, 480);

    ncnn::Net net;

    if(argc != 1)
    {
        config_model_path = argv[1];
    }

    loadModelConfig(config_model_path);
    loadModel(config_model_param_path, config_model_bin_path, net);

    std::vector<Object> objects;

    while (1)
    {
        double t_start = ncnn::get_current_time();
        cv::Mat mat_src;
        getMat(mat_src);

        if(config_load_success)
        {
            objects.clear();
            if(config_model_type == MODEL_TYPE_YOLOFASTEST)
            {
                detect_yolofastest(mat_src, net, objects, 320, 320);
            }
            else if(config_model_type == MODEL_TYPE_NANODET)
            {
                detect_nanodet(mat_src, net, objects);
            }
        }

        cv::Mat mat_div;
        if(isStreamOpend())
        {
            cv::resize(mat_src, mat_div, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        }

        if(config_load_success)
        {
            draw_objects(objects, mat_div);
        }
        else
        {
            char str[128];
            sprintf(str, "Failed to load %s", config_model_path.c_str());
            sendPIPEMessage(str);
        #ifdef LOCAL_RENDER
            cv::putText(mat_div, str, cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        #else
            RD_addString(10, 10, str, "#FF0000", IMAGE_DIV);
        #endif
        }

        sendMat(mat_div);

        double t_end = ncnn::get_current_time();
        double dt_total = t_end - t_start;
        fprintf(stderr, "total %3.1f\n", dt_total);
    }

    return 0;
}

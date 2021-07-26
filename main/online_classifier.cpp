#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "net.h"
#include "framework.h"
#include "benchmark.h"
#include <opencv2/ml.hpp>

#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <dirent.h>

#define PROB printf("prob %d\n", __LINE__);
#define IMAGE_DIV 2
#define TARGET_ROI_W    300
#define TARGET_ROI_H    300

enum
{
    OPERATION_MODE_TRAIN = 0,
    OPERATION_MODE_RUN = 1
};

using namespace cv;

typedef struct
{
    std::string name;
    int set_size;
    float feature[1024];
}class_t;

std::map<int, class_t> classes;
int operation_mode = OPERATION_MODE_RUN;

void pretty_print(const ncnn::Mat& m)
{
    fprintf(stderr, "w = %d, h = %d, c = %d\n", m.w, m.h, m.c);
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}


float cosine_similarity(const ncnn::Mat& A, int len_B, float* B)
{
    float w12 = 0.0, w1 = 0.0, w2 = 0.0 ;
    const float* A_ptr = A.channel(0);
    
    if(__builtin_expect(len_B != A.w, 0)){
        printf("Dim A: %u, Dim B: %u\n", A.w, len_B);
        printf("ERROR: Mat A doesnot have same dim with array B\n");
        return 0.0;
    }

    #ifdef PRINT_DEBUG_INFO
    printf("Consine Sim:\n");
    #endif
    #pragma omp parallel for default(shared) \
                      reduction(+:w12, w1, w2)
    for(int i = 0; i < A.w; ++i) {
        #ifdef PRINT_DEBUG_INFO
        printf("%d, %f, %f\n", i, A_ptr[i], B[i]);
        #endif
        w12 += A_ptr[i] * B[i] ;
        w1 += A_ptr[i] * A_ptr[i] ;
        w2 += B[i] * B[i] ;
    }

    float n12 = w1 * w2;
    if(n12 < 1e-6){
        n12 = 1e-6;
    }
    
    if(w12 < 0){
        w12 = -w12;
    }

    return w12 / sqrt(n12);
}

ncnn::Mat shufflenet(ncnn::Net& net, const cv::Mat &mat_in)
{
    resetAllocator();
    double t_start = ncnn::get_current_time();

    ncnn::Mat in = ncnn::Mat::from_pixels(mat_in.data, ncnn::Mat::PIXEL_BGR2RGB, 224, 224);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("data", in);
    ncnn::Mat fout;
    ex.extract("pool", fout);

    // cv::Mat feature_out(fout.c, 1, CV_32FC1);
    // memcpy((unsigned char*)feature_out.data, fout.data, fout.w * fout.h * sizeof(float));

    // pretty_print(fout);
    double t_end = ncnn::get_current_time();
    double dt_total = t_end - t_start;
    fprintf(stderr, "time = %3.1f\n", dt_total);

    return fout;
}

void update_one_feature(ncnn::Net& net, cv::Mat image, class_t &c)
{
    ncnn::Mat feature = shufflenet(net, image);

    if(c.set_size == 0)
    {
        memcpy(c.feature, feature.data, 1024 * sizeof(float));
        c.set_size++;
        return;
    }

    const float* ptr = feature.channel(0);
    for(int i = 0; i < feature.w; i++)
    {
        c.feature[i] = (c.feature[i] + ptr[i]) / 2.0f;
        // fprintf(stderr, "%f, ", c.feature[i]);
    }
    // fprintf(stderr, "\n");
    c.set_size++;
}

void extract_features(ncnn::Net& net, std::string image_dir, float *feature_avg)
{
    std::vector<std::string> files;

    struct stat dir_stat;
	lstat(image_dir.c_str(), &dir_stat);
    struct dirent *filename;
 	DIR * dir;
	dir = opendir(image_dir.c_str());
    while(( filename = readdir(dir)) != NULL )
	{
        if(strstr(filename->d_name, ".png") == NULL)
        {
            continue;
        }
        std::string path(image_dir);
        path += filename->d_name;
        files.push_back(path);
	}

    std::vector<ncnn::Mat> features;

    for(int i = 0; i < files.size(); i++)
    {
        fprintf(stderr, "processing %s\n", files[i].c_str());
        cv::Mat mat_in = cv::imread(files[i], 1);
        ncnn::Mat feature = shufflenet(net, mat_in);
        printf("w = %d, h = %d, c = %d\n", feature.w, feature.h, feature.c);
        features.push_back(feature);
    }


    double sum[1024];
    memset(sum, 0, 1024 * sizeof(double));
    for(int i = 0; i < features.size(); i++)
    {
        const float* ptr = features[i].channel(0);
        for(int j = 0; j < features[i].w; j++)
        {
            sum[j] += ptr[j];
        }
    }
    
    for(int j = 0; j < 1024; j++)
    {
        feature_avg[j] = sum[j] / features.size();
    }
}

bool load_featrues(std::string path)
{
    FILE *fp_feature;
    FILE *fp_info;
    DynamicJsonDocument doc(1024 * 8);
	
    fp_feature = fopen(std::string(path + "/classifier_online_features.dat").c_str(), "rb");
    if(fp_feature == NULL)
	{
        doc["web"] = 1;
        doc["error"] = "can not load features";
        sendPIPEJsonDoc(doc);
		perror("open file error:");
		return 0;
	}

    fp_info = fopen(std::string(path + "/classifier_online_info.json").c_str(), "r");
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
    doc["list_data"] = 1;

    fseek(fp_feature, SEEK_SET, SEEK_END);  
    int block = ftell(fp_feature) / (1024 * sizeof(float));
    rewind(fp_feature);

    classes.clear();
    for(int i = 0; i < block; i++)
    {
        class_t new_class;
        memset(new_class.feature, 0, 1024 * sizeof(float));
        int total = fread((unsigned char*)new_class.feature, sizeof(float), 1024, fp_feature);
        new_class.name = doc["classes"][i]["name"].as<std::string>();
        new_class.set_size = doc["classes"][i]["set_size"].as<int>();
        int idx = doc["classes"][i]["id"].as<int>();
        classes[idx] = new_class;

        fprintf(stderr, "class = %s, size = %d, id = %d, feature = %d bytes\n", classes[idx].name.c_str(), classes[idx].set_size, idx, total);
    }

    sendPIPEJsonDoc(doc);

    fclose(fp_feature);
    fclose(fp_info);

    fprintf(stderr, "loaded %d class(es).\n", block);

    return 1;
}

bool save_features(std::string path)
{
    FILE *fp_feature;
    FILE *fp_info;
	
    fp_feature = fopen(std::string(path + "/classifier_online_features.dat").c_str(), "wb");
    if(fp_feature == NULL)
	{
		perror("open file error:");
		return 0;
	}

    fp_info = fopen(std::string(path + "/classifier_online_info.json").c_str(), "w");
    if(fp_info == NULL)
	{
		perror("open file error:");
		return 0;
	}

    DynamicJsonDocument doc(4096);
    std::string payload;

    int idx = 0;
    for (auto iter = classes.begin(); iter != classes.end(); iter++)
    {
        fwrite((unsigned char*)iter->second.feature, sizeof(float), 1024, fp_feature);
        
        doc["classes"][idx]["name"] = iter->second.name;
        doc["classes"][idx]["set_size"] = iter->second.set_size;
        doc["classes"][idx]["id"] = iter->first;

        fprintf(stderr, "class = %s, size = %d, id = %d\n", iter->second.name.c_str(), iter->second.set_size, iter->first);

        idx++;
    }

    serializeJson(doc, payload);
    fwrite(payload.c_str(), payload.length(), 1, fp_info);

    fclose(fp_feature);
    fclose(fp_info);

    return 1;
}

int compare_class(ncnn::Net& net, cv::Mat mat_in)
{
    ncnn::Mat feature = shufflenet(net, mat_in);
    float max_delta = 0;
    int id = -1;

    DynamicJsonDocument doc(1024 * 16);

    doc["class_num"] = classes.size();
    if(classes.size() == 0)
    {
        return -1;
    }
    
    int i = 0;
    for (auto iter = classes.begin(); iter != classes.end(); iter++)
    {
        float delta = cosine_similarity(feature, 1024, iter->second.feature);
        doc["class"][i]["name"] = iter->second.name;
        doc["class"][i]["score"] = delta;
        
        if(delta > max_delta)
        {
            id = iter->first;
            max_delta = delta;
        }
        i++;
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "max_delta = %f, id = %d\n", max_delta, id);
    doc["best_match"] = classes[id].name;
    doc["best_score"] = max_delta;

    sendPIPEJsonDoc(doc);

    return id;
}

int main(int argc, char **argv)
{
    startFramework("Online Classifier", 640, 480);

    ncnn::Net net;
    loadModel("./models/shufflenet_v2_x0.5-opt", net);

    Rect target_roi((640 - (TARGET_ROI_W)) / 2, (480 - (TARGET_ROI_H)) / 2, TARGET_ROI_W, TARGET_ROI_H);
    Rect target_roi_div(((640 / IMAGE_DIV) - (TARGET_ROI_W / IMAGE_DIV)) / 2, 
                          ((480 / IMAGE_DIV) - (TARGET_ROI_H / IMAGE_DIV)) / 2, 
                          TARGET_ROI_W / IMAGE_DIV, TARGET_ROI_H / IMAGE_DIV);
    int last_update_id = -1;

    if(load_featrues("./data"))
    {
        operation_mode = OPERATION_MODE_RUN;
    }
    else
    {
        operation_mode = OPERATION_MODE_TRAIN;
    }

    while (1)
    {
        double t_start = ncnn::get_current_time();
        cv::Mat mat_src;
        getMat(mat_src);

        cv::Mat target = mat_src(target_roi).clone();

        std::string payload;
        if(readPIPE(payload))
        {
            try
            {
                DynamicJsonDocument doc(1024);
                deserializeJson(doc, payload);
                if((doc["config"] == "web_update_data") || (doc["config"] == getFunctionName()))
                {
                    operation_mode = OPERATION_MODE_TRAIN;
                    if(doc["operation"] == "train")
                    {
                        int class_id = doc["class_id"];
                        last_update_id = class_id;
                        if(classes.count(class_id) == 0)
                        {
                            class_t new_class;
                            memset(new_class.feature, 0, 1024 * sizeof(float));
                            new_class.set_size = 0;
                            new_class.name = doc["class"].as<std::string>();
                            classes[class_id] = new_class;
                        }
                        update_one_feature(net, target, classes[class_id]);
                        classes[class_id].name = doc["class"].as<std::string>();
                        char str[1024];
                        sprintf(str, "Training %s %d times.", classes[last_update_id].name.c_str(),
                                        classes[last_update_id].set_size);
                        sendPIPEMessage(str);
                    }
                    else if(doc["operation"] == "saverun")
                    {
                        sendPIPEMessage("Save and run.");
                        operation_mode = OPERATION_MODE_RUN;
                        save_features("./data");
                    }
                    else if(doc["operation"] == "reset")
                    {
                        sendPIPEMessage("Please take a picture.");
                        last_update_id = -1;
                        classes.clear();
                    }
                }
            }
            catch(...)
            {
                fprintf(stderr, "[ERROR] Can not parse json.");
            }
        }

        cv::Mat mat_div;
        cv::resize(mat_src, mat_div, cv::Size(), 1.0f / IMAGE_DIV, 1.0f / IMAGE_DIV, cv::INTER_AREA);

        char str[1024];
        if(operation_mode == OPERATION_MODE_TRAIN)
        {
            if(last_update_id == -1)
            {
                sprintf(str, "Please take a picture.");
            }
            else
            {
                sprintf(str, "Training %s %d times.", classes[last_update_id].name.c_str(),
                                        classes[last_update_id].set_size);
            }
        }
        else
        {
            int id = compare_class(net, target);
            if(id == -1)
            {
                sprintf(str, "Result: NULL");
            }
            else
            {
                sprintf(str, "Result: %s", classes[id].name.c_str());
            }
        }
        
        if(isStreamOpend())
        {
        #ifdef LOCAL_RENDER
            cv::putText(mat_div, str, cv::Point(10, 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            rectangle(mat_div, target_roi_div, Scalar( 0, 255, 0 ), 2, 1 );
        #else
            RD_addRectangle(target_roi_div.x, target_roi_div.y, target_roi_div.width, target_roi_div.height, "#00CD00", IMAGE_DIV);
            RD_addString(10, 20, str, "#00CD00");
        #endif
            
        }
        
        sendMat(mat_div);

        double t_end = ncnn::get_current_time();
        double dt_total = t_end - t_start;
        fprintf(stderr, "total %3.1f\n", dt_total);
    }

    return 0;
}

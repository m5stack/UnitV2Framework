#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "framework.h"
#include "arcface.h"

#define IMAGE_DIV (2)
#define MATCH_THRESHOLD (0.5f)

using namespace cv;
using namespace std;

enum
{
    OPERATION_MODE_TRAIN = 0,
    OPERATION_MODE_RUN = 1
};

int operation_mode = OPERATION_MODE_RUN;
int training_face_id = -1;

typedef struct
{
    std::string name;
    std::vector<float> feature;
}face_feature_t;

std::vector<face_feature_t> saved_faces;

static inline float intersection_area(const FaceInfo& a, const FaceInfo& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceInfo>& FaceInfos, int left, int right)
{
    int i = left;
    int j = right;
    float p = FaceInfos[(left + right) / 2].prob;

    while (i <= j)
    {
        while (FaceInfos[i].prob > p)
            i++;

        while (FaceInfos[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(FaceInfos[i], FaceInfos[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(FaceInfos, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(FaceInfos, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<FaceInfo>& FaceInfos)
{
    if (FaceInfos.empty())
        return;

    qsort_descent_inplace(FaceInfos, 0, FaceInfos.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceInfo>& FaceInfos, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = FaceInfos.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = FaceInfos[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const FaceInfo& a = FaceInfos[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const FaceInfo& b = FaceInfos[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// copy from src/layer/proposal.cpp
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++)
    {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++)
        {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

static void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, float prob_threshold, std::vector<FaceInfo>& FaceInfos)
{
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q = 0; q < num_anchors; q++)
    {
        const float* anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q + num_anchors);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++)
        {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++)
            {
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold)
                {
                    // apply center size
                    float dx = bbox.channel(0)[index];
                    float dy = bbox.channel(1)[index];
                    float dw = bbox.channel(2)[index];
                    float dh = bbox.channel(3)[index];

                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float pb_cx = cx + anchor_w * dx;
                    float pb_cy = cy + anchor_h * dy;

                    float pb_w = anchor_w * exp(dw);
                    float pb_h = anchor_h * exp(dh);

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    FaceInfo obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.landmark[0] = cx + (anchor_w + 1) * landmark.channel(0)[index];
                    obj.landmark[1] = cy + (anchor_h + 1) * landmark.channel(1)[index];
                    obj.landmark[2] = cx + (anchor_w + 1) * landmark.channel(2)[index];
                    obj.landmark[3] = cy + (anchor_h + 1) * landmark.channel(3)[index];
                    obj.landmark[4] = cx + (anchor_w + 1) * landmark.channel(4)[index];
                    obj.landmark[5] = cy + (anchor_h + 1) * landmark.channel(5)[index];
                    obj.landmark[6] = cx + (anchor_w + 1) * landmark.channel(6)[index];
                    obj.landmark[7] = cy + (anchor_h + 1) * landmark.channel(7)[index];
                    obj.landmark[8] = cx + (anchor_w + 1) * landmark.channel(8)[index];
                    obj.landmark[9] = cy + (anchor_h + 1) * landmark.channel(9)[index];
                    obj.prob = prob;

                    FaceInfos.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
}

static int detect_retinaface(ncnn::Net &retinaface, const cv::Mat& bgr, std::vector<FaceInfo>& FaceInfos)
{
    resetAllocator();

    const float prob_threshold = 0.8f;
    const float nms_threshold = 0.4f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);

    ncnn::Extractor ex = retinaface.create_extractor();

    ex.input("data", in);

    std::vector<FaceInfo> faceproposals;

    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
        ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride32", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 32.f;
        scales[1] = 16.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceInfo> FaceInfos32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, FaceInfos32);

        faceproposals.insert(faceproposals.end(), FaceInfos32.begin(), FaceInfos32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
        ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride16", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceInfo> FaceInfos16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, FaceInfos16);

        faceproposals.insert(faceproposals.end(), FaceInfos16.begin(), FaceInfos16.end());
    }

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
        ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride8", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 2.f;
        scales[1] = 1.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceInfo> FaceInfos8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, FaceInfos8);

        faceproposals.insert(faceproposals.end(), FaceInfos8.begin(), FaceInfos8.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();

    FaceInfos.resize(face_count);
    for (int i = 0; i < face_count; i++)
    {
        FaceInfos[i] = faceproposals[picked[i]];

        // clip to image size
        float x0 = FaceInfos[i].rect.x;
        float y0 = FaceInfos[i].rect.y;
        float x1 = x0 + FaceInfos[i].rect.width;
        float y1 = y0 + FaceInfos[i].rect.height;

        x0 = std::max(std::min(x0, (float)img_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float)img_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float)img_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float)img_h - 1), 0.f);

        FaceInfos[i].rect.x = x0;
        FaceInfos[i].rect.y = y0;
        FaceInfos[i].rect.width = x1 - x0;
        FaceInfos[i].rect.height = y1 - y0;
        FaceInfos[i].x[0] = x0;
        FaceInfos[i].y[0] = y0;
        FaceInfos[i].x[1] = x1;
        FaceInfos[i].y[1] = y1;
    }

    return 0;
}

static void draw_faces(const cv::Mat& image, const std::vector<FaceInfo>& FaceInfos)
{
    if(FaceInfos.size() == 0)
    {
        RD_clear();
        return;
    }

    DynamicJsonDocument doc(1024 * 16);

    doc["num"] = FaceInfos.size();

    for (size_t i = 0; i < FaceInfos.size(); i++)
    {
        const FaceInfo& obj = FaceInfos[i];

        doc["face"][i]["x"] = obj.rect.x * IMAGE_DIV;
        doc["face"][i]["y"] = obj.rect.y * IMAGE_DIV;
        doc["face"][i]["w"] = obj.rect.width * IMAGE_DIV;
        doc["face"][i]["h"] = obj.rect.height * IMAGE_DIV;
        doc["face"][i]["prob"] = obj.prob;
        if((obj.match_id >= 0))
        {
            doc["face"][i]["match_prob"] = obj.match_score;
            if(saved_faces.size() > 0)
            {
                doc["face"][i]["name"] = saved_faces[obj.match_id].name;
            }
        }
        else
        {
            doc["face"][i]["name"] = "unidentified";
        }
        
        for(int j = 0; j < 10; j += 2)
        {
            doc["face"][i]["mark"][j / 2]["x"] = (int)obj.landmark[j] * IMAGE_DIV;
            doc["face"][i]["mark"][j / 2]["y"] = (int)obj.landmark[j + 1] * IMAGE_DIV;
        }

        // fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", obj.prob,
        //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        if(isStreamOpend())
        {
            char text[256];
            
            if((obj.match_id >= 0) && (saved_faces.size() > 0) && (obj.match_score > MATCH_THRESHOLD))
            {
                // sprintf(text, "%.1f%%, %d, %s, %.1f%%,", obj.prob * 100, obj.match_id, saved_faces[obj.match_id].name.c_str(), obj.match_score * 100);
                sprintf(text, "%d, %s, %.1f%%,", obj.match_id, saved_faces[obj.match_id].name.c_str(), obj.match_score * 100);
            #ifdef LOCAL_RENDER
                cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0));
                // sprintf(text, "%.1f%%, %d, %s", obj.prob * 100, obj.match_id, saved_faces[obj.match_id].name.c_str());
                sprintf(text, "%d, %s", obj.match_id, saved_faces[obj.match_id].name.c_str());
            #else
                RD_addRectangle(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, "#00CD00", IMAGE_DIV);
                RD_addString(obj.rect.x, obj.rect.y, text, "#00CD00", IMAGE_DIV);
            #endif
            }
            else
            {
                // sprintf(text, "%.1f%%", obj.prob * 100);
                sprintf(text, "unidentified");
            #ifdef LOCAL_RENDER
                cv::rectangle(image, obj.rect, cv::Scalar(0, 0, 255));
            #else
                RD_addRectangle(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, "#CC0000", IMAGE_DIV);
                RD_addString(obj.rect.x, obj.rect.y, text, "#CC0000", IMAGE_DIV);
            #endif
            }

        #ifdef LOCAL_RENDER
            cv::circle(image, cv::Point(obj.landmark[0], obj.landmark[1]), 2, cv::Scalar(0, 255, 255), -1);
            cv::circle(image, cv::Point(obj.landmark[2], obj.landmark[3]), 2, cv::Scalar(0, 255, 255), -1);
            cv::circle(image, cv::Point(obj.landmark[4], obj.landmark[5]), 2, cv::Scalar(0, 255, 255), -1);
            cv::circle(image, cv::Point(obj.landmark[6], obj.landmark[7]), 2, cv::Scalar(0, 255, 255), -1);
            cv::circle(image, cv::Point(obj.landmark[8], obj.landmark[9]), 2, cv::Scalar(0, 255, 255), -1);
        #else
            RD_addPoint(obj.landmark[0], obj.landmark[1], "#FFCC00", IMAGE_DIV);
            RD_addPoint(obj.landmark[2], obj.landmark[3], "#FFCC00", IMAGE_DIV);
            RD_addPoint(obj.landmark[4], obj.landmark[5], "#FFCC00", IMAGE_DIV);
            RD_addPoint(obj.landmark[6], obj.landmark[7], "#FFCC00", IMAGE_DIV);
            RD_addPoint(obj.landmark[8], obj.landmark[9], "#FFCC00", IMAGE_DIV);
        #endif

        #ifdef LOCAL_RENDER
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            label_size.height += 10;

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;
            
            if(obj.match_score > MATCH_THRESHOLD)
            {
                cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height)),
                        cv::Scalar(0, 255, 0), -1);
            }
            else
            {
                cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height)),
                        cv::Scalar(0, 0, 255), -1);
            }
            cv::putText(image, text, cv::Point(x, y + label_size.height - 7),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        #endif
        }
    }

    sendPIPEJsonDoc(doc);
}

void merge_feature(std::vector<float> feature_src, std::vector<float> new_feature)
{
    if(feature_src.size() == 0)
    {
        feature_src = new_feature;
    }
    else
    {
        for(int i = 0; i < feature_src.size(); i++)
        {
            feature_src[i] = (feature_src[i] + new_feature[i]) / 2.0f;
        }
    }
}

void train_faces(Arcface &arc, const cv::Mat& image, std::vector<FaceInfo>& FaceInfos)
{
    if(FaceInfos.size() == 0)
    {
        return;
    }

    DynamicJsonDocument doc(2048);
    
    int max_area = 0;
    int max_face = -1;
    for (size_t i = 0; i < FaceInfos.size(); i++)
    {
        if(FaceInfos[i].rect.area() > max_area)
        {
            max_area = FaceInfos[i].rect.area();
            max_face = i;
        }
    }

    const FaceInfo& obj = FaceInfos[max_face];

    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);
    ncnn::Mat ncnn_imface = preprocess(ncnn_img, obj);
    std::vector<float> face_feature = arc.getFeature(ncnn_imface);

    char text[256];
    bool is_missing = false;
    if(saved_faces[training_face_id].feature.size() == 0)
    {
        saved_faces[training_face_id].feature = face_feature;

    #ifdef LOCAL_RENDER
        cv::rectangle(image, obj.rect, cv::Scalar(14, 201, 255));
    #else
        RD_addRectangle(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, "#FFC90E", IMAGE_DIV);
    #endif
        // sprintf(text, "%.1f%%, %s", obj.prob * 100, saved_faces[training_face_id].name.c_str());
        sprintf(text, "%s", saved_faces[training_face_id].name.c_str());
        doc["status"] = "training";
        doc["x"] = obj.rect.x * IMAGE_DIV;
        doc["y"] = obj.rect.y * IMAGE_DIV;
        doc["w"] = obj.rect.width * IMAGE_DIV;
        doc["h"] = obj.rect.height * IMAGE_DIV;
        doc["prob"] = obj.prob;
        doc["name"] = saved_faces[training_face_id].name;
    }
    else
    {
        float score = calcSimilar(saved_faces[training_face_id].feature, face_feature);
        if(score < 0.7f)
        {
        #ifdef LOCAL_RENDER
            cv::rectangle(image, obj.rect, cv::Scalar(0, 0, 255));
        #else
            RD_addRectangle(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, "#CC0000", IMAGE_DIV);
        #endif
            // sprintf(text, "%.1f%%, Missing, %.1f%%,", obj.prob * 100, score * 100);
            sprintf(text, "Missing, %.1f%%,", score * 100);
            is_missing = true;
            doc["status"] = "missing";
            doc["x"] = obj.rect.x * IMAGE_DIV;
            doc["y"] = obj.rect.y * IMAGE_DIV;
            doc["w"] = obj.rect.width * IMAGE_DIV;
            doc["h"] = obj.rect.height * IMAGE_DIV;
            doc["prob"] = score;
            doc["name"] = saved_faces[training_face_id].name;
        }
        else
        {
        #ifdef LOCAL_RENDER
            cv::rectangle(image, obj.rect, cv::Scalar(14, 201, 255));
        #else
            RD_addRectangle(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, "#FFC90E", IMAGE_DIV);
        #endif
            // sprintf(text, "%.1f%%, %s, %.1f%%", obj.prob * 100, saved_faces[training_face_id].name.c_str(), score * 100);
            sprintf(text, "%s, %.1f%%", saved_faces[training_face_id].name.c_str(), score * 100);
            merge_feature(saved_faces[training_face_id].feature, face_feature);
            doc["status"] = "training";
            doc["x"] = obj.rect.x * IMAGE_DIV;
            doc["y"] = obj.rect.y * IMAGE_DIV;
            doc["w"] = obj.rect.width * IMAGE_DIV;
            doc["h"] = obj.rect.height * IMAGE_DIV;
            doc["prob"] = score;
            doc["name"] = saved_faces[training_face_id].name;
        }
    }

    sendPIPEJsonDoc(doc);
    
#ifdef LOCAL_RENDER
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    label_size.height += 10;

    int x = obj.rect.x;
    int y = obj.rect.y - label_size.height - baseLine;
    if (y < 0)
        y = 0;
    if (x + label_size.width > image.cols)
        x = image.cols - label_size.width;
#endif
    
    if(is_missing)
    {
    #ifdef LOCAL_RENDER
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height)), cv::Scalar(0, 0, 255), -1);
    #else
        RD_addString(obj.rect.x, obj.rect.y, text, "#CC0000", IMAGE_DIV);
    #endif
    }
    else
    {
    #ifdef LOCAL_RENDER
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height)), cv::Scalar(14, 201, 255), -1);
    #else
        RD_addString(obj.rect.x, obj.rect.y, text, "#FFC90E", IMAGE_DIV);
    #endif
    }
    #ifdef LOCAL_RENDER
    cv::putText(image, text, cv::Point(x, y + label_size.height - 7),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    #endif
    
    return;
}

void extract_features(Arcface &arc, const cv::Mat& image, std::vector<FaceInfo>& FaceInfos)
{
    if(FaceInfos.size() == 0)
    {
        return;
    }

    vector<ncnn::Mat> faces_mat; 
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);
    for(int i = 0; i < FaceInfos.size(); i++)
    {
        faces_mat.push_back(preprocess(ncnn_img, FaceInfos[i]));
    }
    
    for(int i = 0; i < faces_mat.size(); i++)
    {
        FaceInfos[i].feature = arc.getFeature(faces_mat[i]); // feature size = 128
    }
}

void faces_compare(vector<face_feature_t> &features_cmp, vector<FaceInfo>& FaceInfos)
{
    if((FaceInfos.size() == 0) || (features_cmp.size() == 0))
    {
        return;
    }
    for(int i = 0; i < FaceInfos.size(); i++)
    {
        FaceInfos[i].match_score = 0;
        FaceInfos[i].match_id = -1;
        for(int j = 0; j < features_cmp.size(); j++)
        {
            float score = calcSimilar(FaceInfos[i].feature, features_cmp[j].feature);
            if((score > MATCH_THRESHOLD) && (score > FaceInfos[i].match_score))
            {
                FaceInfos[i].match_score = score;
                FaceInfos[i].match_id = j;
            }
        }
    }
}

int update_one_feature(face_feature_t &face, vector<FaceInfo>& FaceInfos, cv::Point face_pos)
{
    int face_id = -1;
    for(int i = 0; i < FaceInfos.size(); i++)
    {
        if(FaceInfos[i].rect.contains(face_pos))
        {
            face_id = i;
            break;
        }
    }

    if(face_id == -1)
    {
        return -1;
    }

    merge_feature(face.feature, FaceInfos[face_id].feature);

    return 0;
}


bool load_featrues(std::string path)
{
    FILE *fp_feature;
    FILE *fp_info;
    DynamicJsonDocument doc(1024 * 8);
	
    fp_feature = fopen(std::string(path + "/face_recognition_features.dat").c_str(), "rb");
    if(fp_feature == NULL)
	{
        doc["web"] = 1;
        doc["error"] = "can not load features";
        sendPIPEJsonDoc(doc);
		perror("open file error:");
		return 0;
	}

    fp_info = fopen(std::string(path + "/face_recognition_info.json").c_str(), "r");
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
    int block = ftell(fp_feature) / (128 * sizeof(float));
    rewind(fp_feature);

    saved_faces.clear();
    float feature[128];
    for(int i = 0; i < block; i++)
    {
        face_feature_t new_face;
        int total = fread((unsigned char*)feature, sizeof(float), 128, fp_feature);
        new_face.name = doc["faces"][i]["name"].as<std::string>();
        new_face.feature = std::vector<float>(feature, feature + 128);
        fprintf(stderr, "name = %s, id = %d\n", new_face.name.c_str(), i);
        saved_faces.push_back(new_face);
    }

    sendPIPEJsonDoc(doc);

    fclose(fp_feature);
    fclose(fp_info);

    fprintf(stderr, "loaded %d face(s).\n", block);

    return 1;
}

bool save_features(std::string path)
{
    FILE *fp_feature;
    FILE *fp_info;
	
    fp_feature = fopen(std::string(path + "/face_recognition_features.dat").c_str(), "wb");
    if(fp_feature == NULL)
	{
		perror("open file error:");
		return 0;
	}

    fp_info = fopen(std::string(path + "/face_recognition_info.json").c_str(), "w");
    if(fp_info == NULL)
	{
		perror("open file error:");
		return 0;
	}

    DynamicJsonDocument doc(1024 * 8);
    std::string payload;

    for(int i = 0; i < saved_faces.size(); i ++)
    {
        fwrite((unsigned char*)(saved_faces[i].feature.data()), sizeof(float), 128, fp_feature);
        
        doc["faces"][i]["name"] = saved_faces[i].name;

        fprintf(stderr, "name = %s\n", saved_faces[i].name.c_str());
    }

    serializeJson(doc, payload);
    fwrite(payload.c_str(), payload.length(), 1, fp_info);

    fclose(fp_feature);
    fclose(fp_info);

    return 1;
}

int main(int argc, char* argv[])
{
    startFramework("Face Recognition", 640, 480);

    load_featrues("./data");

    std::vector<FaceInfo> FaceInfos;

    ncnn::Net net_retinaface;
    loadModel("./models/mnet.25-opt", net_retinaface);
    Arcface arc("./models");

    while(1)
    {
        double t_start = ncnn::get_current_time();
        cv::Mat mat_src;
        getMat(mat_src);
        cv::Mat mat_div;
        cv::resize(mat_src, mat_div, cv::Size(), 1.0f / IMAGE_DIV, 1.0f / IMAGE_DIV, cv::INTER_AREA);

        detect_retinaface(net_retinaface, mat_div, FaceInfos);


        if(operation_mode == OPERATION_MODE_TRAIN)
        {
            train_faces(arc, mat_div, FaceInfos);
        }
        else
        {
            extract_features(arc, mat_div, FaceInfos);
            faces_compare(saved_faces, FaceInfos);
            draw_faces(mat_div, FaceInfos);
        }
        
        sendMat(mat_div);

        std::string payload;
        if(readPIPE(payload))
        {
            try
            {
                DynamicJsonDocument doc(1024);
                deserializeJson(doc, payload);
                if((doc["config"] == "web_update_data") || (doc["config"] == getFunctionName()))
                {
                    if(doc["operation"] == "train")
                    {
                        int face_id = doc["face_id"];
                        if(face_id > saved_faces.size())
                        {
                            fprintf(stderr, "Invalid face id.\n");
                            sendPIPEMessage("Invalid face id.");
                            goto pipe_handle_done;
                        }
                        else if(face_id == saved_faces.size())
                        {
                            face_feature_t new_face;
                            saved_faces.push_back(new_face);
                        }
                        
                        saved_faces[face_id].name = doc["name"].as<std::string>();
                        operation_mode = OPERATION_MODE_TRAIN;
                        training_face_id = face_id;

                        saved_faces[training_face_id].feature.clear();

                        char str[256];
                        sprintf(str, "Training %s", saved_faces[face_id].name.c_str());
                        sendPIPEMessage(str);
                        
                        // if(update_one_feature(saved_faces[face_id], FaceInfos, cv::Point(x, y)))
                        // {
                        //     char str[1024];
                        //     sprintf(str, "Name %s update failed, No face in the selected location.", saved_faces[face_id].name.c_str());
                        //     sendPIPEMessage(str);
                        // }
                        // else
                        // {
                        //     char str[1024];
                        //     sprintf(str, "Name %s updated.", saved_faces[face_id].name.c_str());
                        //     sendPIPEMessage(str);
                        // }
                    }
                    else if(doc["operation"] == "stoptrain")
                    {
                        operation_mode = OPERATION_MODE_RUN;
                        sendPIPEMessage("Exit training mode.");
                    }
                    else if(doc["operation"] == "saverun")
                    {
                        operation_mode = OPERATION_MODE_RUN;
                        sendPIPEMessage("Faces saved.");
                        save_features("./data");
                    }
                    else if(doc["operation"] == "reset")
                    {
                        operation_mode = OPERATION_MODE_RUN;
                        sendPIPEMessage("Reset success");
                        saved_faces.clear();
                    }
                }
            }
            catch(...)
            {
                fprintf(stderr, "[ERROR] Can not parse json.");
            }
        }
        pipe_handle_done:;

        FaceInfos.clear();

        double t_end = ncnn::get_current_time();
        double dt_total = t_end - t_start;
        fprintf(stderr, "total %3.1f\n", dt_total);
    }

    // Mat imgdst;
    // if (argc == 2)
    // {
    //     imgdst = imread(argv[1]);
    // }
    // ncnn::Mat ncnn_img1 = ncnn::Mat::from_pixels(imgdst.data, ncnn::Mat::PIXEL_BGR, imgdst.cols, imgdst.rows);

    // MtcnnDetector detector("./models/arcface");

    // double start = (double)getTickCount();
    // vector<FaceInfo> results1 = detector.Detect(ncnn_img1);
    // cout << "Detection Time: " << (getTickCount() - start) / getTickFrequency() << "s" << std::endl;

    // start = (double)getTickCount();
    // vector<FaceInfo> results2 = detector.Detect(ncnn_img2);
    // cout << "Detection Time: " << (getTickCount() - start) / getTickFrequency() << "s" << std::endl;

    // ncnn::Mat det1 = preprocess(ncnn_img1, results1[0]);
    // ncnn::Mat det2 = preprocess(ncnn_img2, results2[0]);
    

    // Arcface arc("./models/arcface");

    // start = (double)getTickCount();
    // vector<float> feature1 = arc.getFeature(det1);
    // cout << "Extraction Time: " << (getTickCount() - start) / getTickFrequency() << "s" << std::endl;

    // start = (double)getTickCount();
    // vector<float> feature2 = arc.getFeature(det2);
    // cout << "Extraction Time: " << (getTickCount() - start) / getTickFrequency() << "s" << std::endl;

    // std::cout << "Similarity: " << calcSimilar(feature1, feature2) << std::endl;;

    // imshow("det1", ncnn2cv(det1));
    // imshow("det2", ncnn2cv(det2));

    // waitKey(0);
    return 0;
}

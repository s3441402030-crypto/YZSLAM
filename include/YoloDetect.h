#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream> // 确保包含此头文件以支持文件导出

using namespace std;

class YoloDetection
{
public:
    YoloDetection();
    ~YoloDetection();
    
    void GetImage(cv::Mat& RGB);
    void ClearImage();
    void ClearArea();
    bool Detect();
    void SaveCleanDepth(const cv::Mat& depth, double timestamp);

    // ========== 新增：统计相关公有接口 ==========
    void ResetStat();
    double GetDynamicPointRatio();
    void SaveRatioToFile(const std::string& path);

    // 图像处理及NMS接口
    float generate_scale(cv::Mat& image, const std::vector<int>& target_size);
    float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size);
    torch::Tensor xyxy2xywh(const torch::Tensor& x);
    torch::Tensor xywh2xyxy(const torch::Tensor& x);
    torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold);
    torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.4, float iou_thres = 0.5, int max_det = 300);
    torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape);
    torch::Tensor process_mask(const torch::Tensor& protos, const torch::Tensor& masks_in, const torch::Tensor& bboxes, const cv::Size& shape);

public:
    cv::Mat mRGB;
    cv::Mat mOutMask; 

    torch::jit::script::Module mModule;
    std::vector<std::string> mClassnames;
    torch::Device mDevice{torch::kCPU};

    vector<string> mvDynamicNames;
    vector<cv::Rect2i> mvDynamicArea;
    map<string, vector<cv::Rect2i>> mmDetectMap;
    vector<cv::Rect2i> mvPersonArea = {};

    // ========== 新增：统计相关私有成员变量 ==========
    // 将这些移入私有区域，并在cpp中初始化
private:
    long long m_total_all_pixels = 0;
    long long m_total_dynamic_pixels = 0;
    int m_frame_cnt = 0;
};

#endif //YOLO_DETECT_H

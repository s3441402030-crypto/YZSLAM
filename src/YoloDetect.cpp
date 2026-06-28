#include "YoloDetect.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <iomanip>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <sys/stat.h>
#include <sys/types.h>

using torch::indexing::Slice;
using torch::indexing::None;

// ===============================
// VOC/SBD 20 类模型
// ===============================
const int NUM_CLASSES = 20;

// VOC 20 类中 person 的类别 ID 是 14
const int PERSON_CLASS_ID = 14;

// 可视化开关：false 关闭窗口，true 显示 YOLO mask
static bool g_ShowVis = false;

// 模型和类别文件路径
static const std::string MODEL_PATH = "/home/zncksys/WQ/ORB_SLAM3-with-YOLO123/best.torchscript";
static const std::string CLASS_PATH = "/home/zncksys/WQ/ORB_SLAM3-with-YOLO123/voc.names";

// 输入尺寸
// 如果你的模型输出是 preds=[1,56,6300], protos=[1,32,120,160]，这里保持 640x480
static const int INPUT_W = 640;
static const int INPUT_H = 480;

// NMS 参数
static const float CONF_THRES = 0.35f;
static const float IOU_THRES  = 0.45f;

// mask 阈值
static const float MASK_THRES = 0.5f;

// mask 膨胀核
static const int DILATE_K = 7;

// 自动选择 GPU / CPU
static torch::Device GetTorchDevice()
{
    if (torch::cuda::is_available())
    {
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

static torch::Device g_Device = GetTorchDevice();

static bool FileExists(const std::string& path)
{
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
}

YoloDetection::YoloDetection()
{
    // 新版 LibTorch 不再支持 torch::jit::setTensorExprFuserEnabled(false)，不要使用

    if (!FileExists(MODEL_PATH))
    {
        std::cerr << "ERROR: YOLO model file does not exist: "
                  << MODEL_PATH << std::endl;
        throw std::runtime_error("YOLO TorchScript model not found.");
    }

    mModule = torch::jit::load(MODEL_PATH);
    mModule.eval();
    mModule.to(g_Device);
    mModule.to(torch::kFloat);

    if (!FileExists(CLASS_PATH))
    {
        std::cerr << "ERROR: class name file does not exist: "
                  << CLASS_PATH << std::endl;
        throw std::runtime_error("YOLO class name file not found.");
    }

    std::ifstream f(CLASS_PATH);
    std::string name;
    while (std::getline(f, name))
    {
        if (!name.empty())
        {
            mClassnames.push_back(name);
        }
    }

    if (mClassnames.size() != NUM_CLASSES)
    {
        std::cerr << "WARNING: class number mismatch. "
                  << "mClassnames.size() = " << mClassnames.size()
                  << ", NUM_CLASSES = " << NUM_CLASSES << std::endl;
    }

    // 只去除人体
    mvDynamicNames = {"person"};
}

YoloDetection::~YoloDetection() {}

void YoloDetection::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}

void YoloDetection::ClearArea()
{
    mvDynamicArea.clear();
    mOutMask.release();
}

void YoloDetection::ClearImage()
{
    mRGB.release();
}

bool YoloDetection::Detect()
{
    torch::NoGradGuard no_grad;

    if (mRGB.empty())
    {
        return false;
    }

    // ===============================
    // 1. 图像预处理
    // ===============================
    cv::Mat img_bgr;
    cv::Mat img_rgb;

    if (mRGB.channels() == 1)
    {
        cv::cvtColor(mRGB, img_bgr, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img_bgr = mRGB.clone();
    }

    cv::resize(img_bgr, img_bgr, cv::Size(INPUT_W, INPUT_H));

    // OpenCV 默认 BGR，YOLO 一般使用 RGB
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    torch::Tensor imgTensor = torch::from_blob(
        img_rgb.data,
        {1, INPUT_H, INPUT_W, 3},
        torch::kUInt8
    );

    imgTensor = imgTensor.permute({0, 3, 1, 2}).contiguous();
    imgTensor = imgTensor.to(g_Device);
    imgTensor = imgTensor.to(torch::kFloat).div_(255.0);

    // ===============================
    // 2. YOLO GPU 推理
    // ===============================
    auto output_ivalue = mModule.forward({imgTensor});
    auto outputs = output_ivalue.toTuple();

    torch::Tensor preds_gpu  = outputs->elements()[0].toTensor();
    torch::Tensor protos_gpu = outputs->elements()[1].toTensor();

    // ===============================
    // 3. NMS 在 CPU 执行
    // nms() 内部使用 data_ptr，所以 preds 必须转 CPU
    // protos 保持 GPU，后续 mask 计算使用 GPU
    // ===============================
    torch::Tensor preds_cpu = preds_gpu.detach().to(torch::kCPU).contiguous();

    auto results = non_max_suppression(preds_cpu, CONF_THRES, IOU_THRES);

    mOutMask = cv::Mat::zeros(mRGB.size(), CV_8UC1);
    mvDynamicArea.clear();

    cv::Mat mask_vis;
    if (g_ShowVis)
    {
        mask_vis = mRGB.clone();
    }

    if (results.size(0) == 0 || results[0].size(0) == 0)
    {
        mvDynamicArea.push_back(cv::Rect2i(1, 1, 1, 1));
        return true;
    }

    torch::Tensor dets = results[0].contiguous();

    // ===============================
    // 4. 只保留 person 检测结果
    // ===============================
    torch::Tensor cls_ids = dets.index({Slice(), 5}).to(torch::kLong);
    torch::Tensor person_indices = torch::nonzero(cls_ids == PERSON_CLASS_ID).squeeze();

    if (person_indices.numel() == 0)
    {
        mvDynamicArea.push_back(cv::Rect2i(1, 1, 1, 1));
        return true;
    }

    if (person_indices.dim() == 0)
    {
        person_indices = person_indices.unsqueeze(0);
    }

    torch::Tensor person_dets = dets.index_select(0, person_indices).contiguous();

    if (person_dets.size(0) == 0)
    {
        mvDynamicArea.push_back(cv::Rect2i(1, 1, 1, 1));
        return true;
    }

    // ===============================
    // 5. 记录 person bbox
    // ===============================
    float scale_w = static_cast<float>(mRGB.cols) / static_cast<float>(INPUT_W);
    float scale_h = static_cast<float>(mRGB.rows) / static_cast<float>(INPUT_H);

    for (int i = 0; i < person_dets.size(0); i++)
    {
        int left   = static_cast<int>(person_dets[i][0].item().toFloat() * scale_w);
        int top    = static_cast<int>(person_dets[i][1].item().toFloat() * scale_h);
        int right  = static_cast<int>(person_dets[i][2].item().toFloat() * scale_w);
        int bottom = static_cast<int>(person_dets[i][3].item().toFloat() * scale_h);

        left   = std::max(0, std::min(left,   mRGB.cols - 1));
        right  = std::max(0, std::min(right,  mRGB.cols - 1));
        top    = std::max(0, std::min(top,    mRGB.rows - 1));
        bottom = std::max(0, std::min(bottom, mRGB.rows - 1));

        if (right > left && bottom > top)
        {
            mvDynamicArea.push_back(cv::Rect2i(left, top, right - left, bottom - top));
        }
    }

    if (mvDynamicArea.empty())
    {
        mvDynamicArea.push_back(cv::Rect2i(1, 1, 1, 1));
        return true;
    }

    // ===============================
    // 6. 只对 person 计算 mask
    // mask 系数转 GPU，protos 保持 GPU
    // ===============================
    torch::Tensor mask_weights_gpu = person_dets.index({Slice(), Slice(6, None)})
                                         .to(g_Device)
                                         .contiguous();

    torch::Tensor bboxes_gpu = person_dets.index({Slice(), Slice(0, 4)})
                                   .to(g_Device)
                                   .contiguous();

    torch::Tensor masks_gpu = process_mask(
        protos_gpu[0],
        mask_weights_gpu,
        bboxes_gpu,
        cv::Size(INPUT_W, INPUT_H)
    );

    if (masks_gpu.size(0) == 0)
    {
        return true;
    }

    // ===============================
    // 7. 多个人体 mask 在 GPU 上合并
    // masks_gpu: [N, INPUT_H, INPUT_W]
    // ===============================
    torch::Tensor merged_mask_gpu = std::get<0>(masks_gpu.max(0));

    // ===============================
    // 8. GPU 上插值到原图大小
    // ===============================
    merged_mask_gpu = torch::nn::functional::interpolate(
        merged_mask_gpu.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({mRGB.rows, mRGB.cols}))
            .mode(torch::kBilinear)
            .align_corners(false)
    ).squeeze();

    // ===============================
    // 9. GPU 上二值化
    // 最终只把一张二值 mask 拷回 CPU
    // ===============================
    torch::Tensor binary_gpu = merged_mask_gpu.gt(MASK_THRES)
                                  .to(torch::kUInt8)
                                  .mul(255)
                                  .contiguous();

    torch::Tensor binary_cpu = binary_gpu.to(torch::kCPU).contiguous();

    cv::Mat temp_mask(
        mRGB.rows,
        mRGB.cols,
        CV_8UC1,
        binary_cpu.data_ptr<uint8_t>()
    );

    temp_mask.copyTo(mOutMask);

    // ===============================
    // 10. CPU 上做一次膨胀，扩大人体边界
    // ===============================
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(DILATE_K, DILATE_K)
    );

    cv::dilate(mOutMask, mOutMask, element);

    // ===============================
    // 11. 可视化，默认关闭
    // ===============================
    if (g_ShowVis)
    {
        cv::Mat color_mask = cv::Mat::zeros(mRGB.size(), mRGB.type());
        color_mask.setTo(cv::Scalar(0, 0, 255), mOutMask);

        cv::addWeighted(mask_vis, 1.0, color_mask, 0.5, 0, mask_vis);

        for (size_t i = 0; i < mvDynamicArea.size(); i++)
        {
            cv::rectangle(mask_vis, mvDynamicArea[i], cv::Scalar(0, 255, 0), 2);
            cv::putText(
                mask_vis,
                "person",
                cv::Point(mvDynamicArea[i].x, std::max(0, mvDynamicArea[i].y - 5)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 0),
                2
            );
        }

        cv::imshow("YOLO Segmentation Mask", mask_vis);
        cv::waitKey(1);
    }

    return true;
}

torch::Tensor YoloDetection::process_mask(
    const torch::Tensor& protos,
    const torch::Tensor& masks_in,
    const torch::Tensor& bboxes,
    const cv::Size& shape)
{
    // 注意：
    // 这个函数必须保留 4 个参数，因为你的 YoloDetect.h 里就是这样声明的。
    // bboxes 当前不参与 crop，只是为了兼容接口。

    auto c  = protos.size(0);
    auto mh = protos.size(1);
    auto mw = protos.size(2);

    torch::Tensor proto_flat = protos.contiguous().view({c, -1});
    torch::Tensor mask_coeff = masks_in.contiguous();

    if (mask_coeff.size(1) != c)
    {
        return torch::zeros(
            {mask_coeff.size(0), shape.height, shape.width},
            masks_in.options().dtype(torch::kFloat)
        );
    }

    torch::Tensor masks = mask_coeff.matmul(proto_flat).view({-1, mh, mw});
    masks = masks.sigmoid();

    masks = torch::nn::functional::interpolate(
        masks.unsqueeze(1),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({shape.height, shape.width}))
            .mode(torch::kBilinear)
            .align_corners(false)
    ).squeeze(1);

    return masks.contiguous();
}

torch::Tensor YoloDetection::non_max_suppression(
    torch::Tensor& prediction,
    float conf_thres,
    float iou_thres,
    int max_det)
{
    // prediction: [batch, channels, num_boxes] -> [batch, num_boxes, channels]
    prediction = prediction.transpose(-1, -2).contiguous();

    auto bs = prediction.size(0);
    auto nm = 32;

    prediction.index_put_(
        {"...", Slice(None, 4)},
        xywh2xyxy(prediction.index({"...", Slice(None, 4)}))
    );

    std::vector<torch::Tensor> output;

    for (int b = 0; b < bs; b++)
    {
        auto x = prediction[b].contiguous();

        // VOC/SBD 20 类：类别分数范围 [4, 24)
        auto cls_scores = x.index({Slice(), Slice(4, 4 + NUM_CLASSES)});
        auto conf_j = cls_scores.max(1, true);

        torch::Tensor conf = std::get<0>(conf_j);
        torch::Tensor j    = std::get<1>(conf_j);

        x = torch::cat(
            {
                x.index({Slice(), Slice(0, 4)}),
                conf,
                j.to(torch::kFloat),
                x.index({Slice(), Slice(4 + NUM_CLASSES, None)})
            },
            1
        ).contiguous();

        x = x.index({conf.view(-1) > conf_thres}).contiguous();

        if (x.size(0) == 0)
        {
            output.push_back(torch::zeros({0, 6 + nm}, x.options()));
            continue;
        }

        auto indices = nms(
            x.index({Slice(), Slice(None, 4)}).contiguous(),
            x.index({Slice(), 4}).contiguous(),
            iou_thres
        );

        int keep_num = std::min(static_cast<int>(indices.size(0)), max_det);
        output.push_back(x.index({indices.slice(0, 0, keep_num)}).contiguous());
    }

    return torch::stack(output);
}

torch::Tensor YoloDetection::xywh2xyxy(const torch::Tensor& x)
{
    auto y = torch::empty_like(x);

    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);

    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);

    return y;
}

torch::Tensor YoloDetection::nms(
    const torch::Tensor& bboxes,
    const torch::Tensor& scores,
    float iou_threshold)
{
    if (bboxes.numel() == 0)
    {
        return torch::empty({0}, bboxes.options().dtype(torch::kLong));
    }

    torch::Tensor boxes_cpu = bboxes.detach().to(torch::kCPU).contiguous();
    torch::Tensor scores_cpu = scores.detach().to(torch::kCPU).contiguous();

    auto x1 = boxes_cpu.select(1, 0).contiguous();
    auto y1 = boxes_cpu.select(1, 1).contiguous();
    auto x2 = boxes_cpu.select(1, 2).contiguous();
    auto y2 = boxes_cpu.select(1, 3).contiguous();

    torch::Tensor areas = ((x2 - x1).clamp_min(0)) * ((y2 - y1).clamp_min(0));

    auto order = std::get<1>(scores_cpu.sort(0, true)).contiguous();

    auto ndets = boxes_cpu.size(0);

    torch::Tensor suppressed = torch::zeros(
        {ndets},
        torch::TensorOptions().dtype(torch::kUInt8)
    );

    torch::Tensor keep = torch::zeros(
        {ndets},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto s_ptr = suppressed.data_ptr<uint8_t>();
    auto k_ptr = keep.data_ptr<int64_t>();
    auto o_ptr = order.data_ptr<int64_t>();

    auto x1_p = x1.data_ptr<float>();
    auto y1_p = y1.data_ptr<float>();
    auto x2_p = x2.data_ptr<float>();
    auto y2_p = y2.data_ptr<float>();
    auto a_p  = areas.data_ptr<float>();

    int64_t count = 0;

    for (int64_t _i = 0; _i < ndets; _i++)
    {
        auto i = o_ptr[_i];

        if (s_ptr[i] == 1)
        {
            continue;
        }

        k_ptr[count++] = i;

        for (int64_t _j = _i + 1; _j < ndets; _j++)
        {
            auto j = o_ptr[_j];

            if (s_ptr[j] == 1)
            {
                continue;
            }

            float xx1 = std::max(x1_p[i], x1_p[j]);
            float yy1 = std::max(y1_p[i], y1_p[j]);
            float xx2 = std::min(x2_p[i], x2_p[j]);
            float yy2 = std::min(y2_p[i], y2_p[j]);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);

            float inter = w * h;
            float denom = a_p[i] + a_p[j] - inter;

            float iou = 0.0f;
            if (denom > 0.0f)
            {
                iou = inter / denom;
            }

            if (iou > iou_threshold)
            {
                s_ptr[j] = 1;
            }
        }
    }

    return keep.narrow(0, 0, count).contiguous();
}

/**
* This file is part of ORB-SLAM3
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* ...
*/

#ifndef SYSTEM_H
#define SYSTEM_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <opencv2/core/core.hpp>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "ImuTypes.h"
#include "Settings.h"
#include "YoloDetect.h" 

namespace ORB_SLAM3
{

class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET=0,
        VERBOSITY_NORMAL=1,
        VERBOSITY_VERBOSE=2,
        VERBOSITY_VERY_VERBOSE=3,
        VERBOSITY_DEBUG=4
    };
    static eLevel th;

public:
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            cout << str << endl;
    }
    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

// --- 前向声明 ---
class Viewer;
class FrameDrawer;
class MapDrawer;
class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;
class Settings;
class DenseMapping; // <--- 新增：稠密建图类前向声明

class System
{
public:
    // 传感器类型
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_MONOCULAR=3,
        IMU_STEREO=4,
        IMU_RGBD=5,
    };

    // 文件类型
    enum FileType{
        TEXT_FILE=0,
        BINARY_FILE=1,
    };

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // 初始化 SLAM 系统，启动各个线程
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true, const int initFr = 0, const string &strSequence = std::string());

    // 跟踪函数接口
    Sophus::SE3f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");
    Sophus::SE3f TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");
    Sophus::SE3f TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // 模式切换
    void ActivateLocalizationMode();
    void DeactivateLocalizationMode();

    bool MapChanged();
    void Reset();
    void ResetActiveMap();

    // 退出管理
    void Shutdown();
    bool isShutDown();

    // 轨迹保存相关函数
    void SaveTrajectoryTUM(const string &filename);
    void SaveKeyFrameTrajectoryTUM(const string &filename);
    void SaveTrajectoryEuRoC(const string &filename);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename);
    void SaveTrajectoryEuRoC(const string &filename, Map* pMap);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename, Map* pMap);
    void SaveDebugData(const int &iniIdx);
    void SaveTrajectoryKITTI(const string &filename);

    // 状态获取
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

    double GetTimeFromIMUInit();
    bool isLost();
    bool isFinished();
    void ChangeDataset();
    float GetImageScale();

#ifdef REGISTER_TIMES
    void InsertRectTime(double& time);
    void InsertResizeTime(double& time);
    void InsertTrackTime(double& time);
#endif

public:
    // --- 模块对象指针 (放在 Public 方便 Tracking 访问) ---
    Tracking* mpTracker;
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopCloser;
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    
    // YOLO 分割检测器
    YoloDetection* mpDetector;

    // 稠密建图模块
    DenseMapping* mpDenseMapper; // <--- 新增：稠密建图对象指针

private:
    void SaveAtlas(int type);
    bool LoadAtlas(int type);
    string CalculateCheckSum(string filename, int type);

    // 成员变量
    eSensor mSensor;
    ORBVocabulary* mpVocabulary;
    KeyFrameDatabase* mpKeyFrameDatabase;
    Atlas* mpAtlas;

    // --- 系统线程管理 ---
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;
    std::thread* mptDenseMapping; // <--- 新增：稠密建图独立线程指针

    std::mutex mMutexReset;
    bool mbReset;
    bool mbResetActiveMap;

    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    bool mbShutDown;

    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;

    string mStrLoadAtlasFromFile;
    string mStrSaveAtlasToFile;
    string mStrVocabularyFilePath;

    Settings* settings_;

};

} // namespace ORB_SLAM3

#endif // SYSTEM_H

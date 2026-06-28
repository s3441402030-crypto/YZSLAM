/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef FRAME_H
#define FRAME_H

#include<vector>
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "Thirdparty/Sophus/sophus/geometry.hpp"
#include "ImuTypes.h"
#include "ORBVocabulary.h"
#include "Converter.h"
#include "Settings.h"
#include <mutex>
#include <opencv2/opencv.hpp>
#include "Eigen/Core"
#include "sophus/se3.hpp"

namespace ORB_SLAM3
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;
class ConstraintPoseImu;
class GeometricCamera;
class ORBextractor;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras. 
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Constructor for RGB-D cameras. (Added mask)
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib(), const cv::Mat &mask = cv::Mat());

    // Constructor for Monocular cameras. 
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Constructor for Stereo Fisheye cameras. 
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr, Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Extract ORB on the image. 0 for left image and 1 for right image.
    // Passed mask to filter dynamic keypoints
    void ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1, const cv::Mat &mask = cv::Mat());

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Setters and Getters
    void SetPose(const Sophus::SE3<float> &Tcw);
    void SetVelocity(Eigen::Vector3f Vw);
    Eigen::Vector3f GetVelocity() const;
    void SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb);
    Eigen::Matrix<float,3,1> GetImuPosition() const;
    Eigen::Matrix<float,3,3> GetImuRotation();
    Sophus::SE3<float> GetImuPose();
    Sophus::SE3f GetRelativePoseTrl();
    Sophus::SE3f GetRelativePoseTlr();
    Eigen::Matrix3f GetRelativePoseTlr_rotation();
    Eigen::Vector3f GetRelativePoseTlr_translation();
    void SetNewBias(const IMU::Bias &b);

    // Geometry/Projection functions
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);
    bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);
    Eigen::Vector3f inRefCoordinates(Eigen::Vector3f pCw);
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel=-1, const int maxLevel=-1, const bool bRight = false) const;

    // Stereo specific
    void ComputeStereoMatches();
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);
    bool UnprojectStereo(const int &i, Eigen::Vector3f &x3D);
    void ComputeStereoFishEyeMatches();
    bool isInFrustumChecks(MapPoint* pMP, float viewingCosLimit, bool bRight = false);
    Eigen::Vector3f UnprojectStereoFishEye(const int &i);

    // IMU related
    bool imuIsPreintegrated();
    void setIntegrated();
    bool isSet() const;

    // Matrices updates
    void UpdatePoseMatrices();
    inline Eigen::Vector3f GetCameraCenter(){ return mOw; }
    inline Eigen::Matrix3f GetRotationInverse(){ return mRwc; }
    inline Sophus::SE3<float> GetPose() const { return mTcw; }
    inline Eigen::Matrix3f GetRwc() const { return mRwc; }
    inline Eigen::Vector3f GetOw() const { return mOw; }
    inline bool HasPose() const { return mbHasPose; }
    inline bool HasVelocity() const { return mbHasVelocity; }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ORBVocabulary* mpORBvocabulary;
    ORBextractor *mpORBextractorLeft, *mpORBextractorRight;
    double mTimeStamp;
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    static float fx, fy, cx, cy, invfx, invfy;
    cv::Mat mDistCoef;
    float mbf, mb, mThDepth;

    // Semantic Mask for dynamic object removal
    cv::Mat mDynamicMask;

    int N;
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<MapPoint*> mvpMapPoints;
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;
    cv::Mat mDescriptors, mDescriptorsRight;
    std::vector<bool> mvbOutlier;
    int mnCloseMPs;

    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    IMU::Bias mPredBias, mImuBias;
    IMU::Calib mImuCalib;
    IMU::Preintegrated* mpImuPreintegrated;
    KeyFrame* mpLastKeyFrame;
    Frame* mpPrevFrame;
    IMU::Preintegrated* mpImuPreintegratedFrame;

    static long unsigned int nNextId;
    long unsigned int mnId;
    KeyFrame* mpReferenceKF;

    int mnScaleLevels;
    float mfScaleFactor, mfLogScaleFactor;
    vector<float> mvScaleFactors, mvInvScaleFactors, mvLevelSigma2, mvInvLevelSigma2;
    static float mnMinX, mnMaxX, mnMinY, mnMaxY;
    static bool mbInitialComputations;

    map<long unsigned int, cv::Point2f> mmProjectPoints;
    map<long unsigned int, cv::Point2f> mmMatchedInImage;
    string mNameFile;
    int mnDataset;
    ConstraintPoseImu* mpcpi;

#ifdef REGISTER_TIMES
    double mTimeORB_Ext;
    double mTimeStereoMatch;
#endif

    GeometricCamera *mpCamera, *mpCamera2;
    int Nleft, Nright;
    int monoLeft, monoRight;
    std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;
    static cv::BFMatcher BFmatcher;
    std::vector<Eigen::Vector3f> mvStereo3Dpoints;

    cv::Mat imgLeft, imgRight;

    void PrintPointDistribution();

private:
    void UndistortKeyPoints();
    void ComputeImageBounds(const cv::Mat &imLeft);
    void AssignFeaturesToGrid();

    Sophus::SE3<float> mTcw;
    Eigen::Matrix<float,3,3> mRwc, mRcw;
    Eigen::Matrix<float,3,1> mOw, mtcw;
    bool mbHasPose;

    Sophus::SE3<float> mTlr, mTrl;
    Eigen::Matrix<float,3,3> mRlr;
    Eigen::Vector3f mtlr;

    Eigen::Vector3f mVw;
    bool mbHasVelocity;
    bool mbIsSet, mbImuPreintegrated;
    std::mutex *mpMutexImu;

    
};

} // namespace ORB_SLAM3

#endif // FRAME_H

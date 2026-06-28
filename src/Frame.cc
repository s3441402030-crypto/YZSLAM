/**
* This file is part of ORB-SLAM3
* 修改说明：第三级判别升级为 Z-Score 自适应分布判别，彻底解决 std::bad_alloc 问题
*/

#include "Frame.h"
#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"
#include <thread>
#include <cmath>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{}

// 拷贝构造函数
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(frame.mK_), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mTcw(frame.mTcw), mbHasPose(frame.mbHasPose), mbHasVelocity(frame.mbHasVelocity)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0) mGridRight[i][j] = frame.mGridRight[i][j];
        }
    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;
}

// =========================================================================================
// 核心逻辑：RGB-D 构造函数 (Z-Score 分布判别版)
// =========================================================================================
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib, const cv::Mat &mask)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    mnId=nNextId++;
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // 1. 全图提点（此处不传mask）
    ExtractORB(0, imGray, 0, 0, cv::Mat()); 

    N = mvKeys.size();
    if(mvKeys.empty()) return;

    UndistortKeyPoints();
    ComputeStereoFromRGBD(imDepth); 

    // 2. 第三级：Z-Score 分布判别逻辑 (极致省内存，不使用 vector)
    if(!mask.empty())
    {
        float sum_d = 0.0f;
        int count_d = 0;

        // Pass A: 计算 Mask 内特征点的均值 (mu)
        for(int i=0; i<N; i++) {
            int px = cvRound(mvKeys[i].pt.x); int py = cvRound(mvKeys[i].pt.y);
            if(px>=0 && px<mask.cols && py>=0 && py<mask.rows) {
                if(mask.at<uchar>(py, px) > 0 && mvDepth[i] > 0.1f && mvDepth[i] < 6.0f) {
                    sum_d += mvDepth[i];
                    count_d++;
                }
            }
        }

        if(count_d > 5) // 至少有5个样本才进行统计分析
        {
            float mu = sum_d / count_d;
            float sum_sq_diff = 0.0f;

            // Pass B: 计算标准差 (sigma)
            for(int i=0; i<N; i++) {
                int px = cvRound(mvKeys[i].pt.x); int py = cvRound(mvKeys[i].pt.y);
                if(px>=0 && px<mask.cols && py>=0 && py<mask.rows) {
                    if(mask.at<uchar>(py, px) > 0 && mvDepth[i] > 0.1f && mvDepth[i] < 6.0f) {
                        float diff = mvDepth[i] - mu;
                        sum_sq_diff += diff * diff;
                    }
                }
            }
            float sigma = std::sqrt(sum_sq_diff / count_d);
            if(sigma < 0.05f) sigma = 0.05f; // 防止除零或过分敏感

            // Pass C: 根据 Z-Score 执行剔除与平反
            for(int i=0; i<N; i++) {
                int px = cvRound(mvKeys[i].pt.x); int py = cvRound(mvKeys[i].pt.y);
                if(px>=0 && px<mask.cols && py>=0 && py<mask.rows) {
                    if(mask.at<uchar>(py, px) > 0) {
                        float D_p = mvDepth[i];
                        if(D_p > 0) {
                            float z_score = std::abs(D_p - mu) / sigma;
                            // Z < 2.0：属于动态物体主分布 -> 剔除 [cite: 350]
                            if(z_score < 2.0f) {
                                mvDepth[i] = -1.0f; mvuRight[i] = -1.0f;
                            }
                            // Z >= 2.0：属于背景离群值 -> 平反保留
                        } else {
                            // Mask 内无深度点，直接设为无效
                            mvDepth[i] = -1.0f; mvuRight[i] = -1.0f;
                        }
                    }
                }
            }
        }
    }

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mmProjectPoints.clear(); mmMatchedInImage.clear(); mvbOutlier = vector<bool>(N,false);

    if(mbInitialComputations) {
        ComputeImageBounds(imGray);
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);
        fx = K.at<float>(0,0); fy = K.at<float>(1,1); cx = K.at<float>(0,2); cy = K.at<float>(1,2);
        invfx = 1.0f/fx; invfy = 1.0f/fy;
        mbInitialComputations=false;
    }
    mb = mbf/fx;
    if(pPrevF && pPrevF->HasVelocity()) SetVelocity(pPrevF->GetVelocity());
    else mVw.setZero();
    mpMutexImu = new std::mutex();
    Nleft = -1; Nright = -1; AssignFeaturesToGrid();
}

// 鱼眼双目构造函数
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
     mbHasPose(false), mbHasVelocity(false)
{
    mnId=nNextId++;
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1],cv::Mat());
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1],cv::Mat());
    threadLeft.join(); threadRight.join();
    Nleft = mvKeys.size(); Nright = mvKeysRight.size(); N = Nleft + Nright;
    if(N == 0) return;
    if(mbInitialComputations) {
        ComputeImageBounds(imLeft);
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
        fx = K.at<float>(0,0); fy = K.at<float>(1,1); cx = K.at<float>(0,2); cy = K.at<float>(1,2);
        invfx = 1.0f/fx; invfy = 1.0f/fy;
        mbInitialComputations=false;
    }
    mb = mbf / fx; mTlr = Tlr; mTrl = mTlr.inverse();
    ComputeStereoFishEyeMatches();
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);
    AssignFeaturesToGrid();
    mpMutexImu = new std::mutex(); UndistortKeyPoints();
}

// 标准双目构造函数
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)   
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    mnId=nNextId++;
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0,cv::Mat());
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0,cv::Mat());
    threadLeft.join(); threadRight.join();
    N = mvKeys.size();
    if(mvKeys.empty()) return;
    UndistortKeyPoints(); ComputeStereoMatches();
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    if(mbInitialComputations) {
        ComputeImageBounds(imLeft);
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
        fx = K.at<float>(0,0); fy = K.at<float>(1,1); cx = K.at<float>(0,2); cy = K.at<float>(1,2);
        invfx = 1.0f/fx; invfy = 1.0f/fy;
        mbInitialComputations=false;
    }
    mb = mbf/fx;
    if(pPrevF && pPrevF->HasVelocity()) SetVelocity(pPrevF->GetVelocity());
    else mVw.setZero();
    mpMutexImu = new std::mutex();
    Nleft = -1; Nright = -1; AssignFeaturesToGrid();
}

// 单目构造函数
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    mnId=nNextId++;
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
    ExtractORB(0,imGray,0,1000);
    N = mvKeys.size();
    if(mvKeys.empty()) return;
    UndistortKeyPoints();
    mvuRight = vector<float>(N,-1); mvDepth = vector<float>(N,-1);
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mmProjectPoints.clear(); mvbOutlier = vector<bool>(N,false);
    if(mbInitialComputations) {
        ComputeImageBounds(imGray);
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);
        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx; invfy = 1.0f/fy;
        mbInitialComputations=false;
    }
    mb = mbf/fx;
    if(pPrevF && pPrevF->HasVelocity()) SetVelocity(pPrevF->GetVelocity());
    else mVw.setZero();
    mpMutexImu = new std::mutex(); AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].clear(); mGrid[i][j].reserve(nReserve);
            if(Nleft != -1) { mGridRight[i][j].clear(); mGridRight[i][j].reserve(nReserve); }
        }
    for(int i=0;i<N;i++) {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i] : (i < Nleft) ? mvKeys[i] : mvKeysRight[i - Nleft];
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft) mGrid[nGridPosX][nGridPosY].push_back(i);
            else mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices; vIndices.reserve(N);
    const int nMinCellX = std::max(0,(int)std::floor((x-mnMinX-r)*mfGridElementWidthInv));
    const int nMaxCellX = std::min((int)FRAME_GRID_COLS-1,(int)std::ceil((x-mnMinX+r)*mfGridElementWidthInv));
    const int nMinCellY = std::max(0,(int)std::floor((y-mnMinY-r)*mfGridElementHeightInv));
    const int nMaxCellY = std::min((int)FRAME_GRID_ROWS-1,(int)std::ceil((y-mnMinY+r)*mfGridElementHeightInv));

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++) {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            for(size_t j=0; j<vCell.size(); j++) {
                const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[vCell[j]] : (!bRight) ? mvKeys[vCell[j]] : mvKeysRight[vCell[j]];
                if(minLevel>0 && kp.octave<minLevel) continue;
                if(maxLevel>=0 && kp.octave>maxLevel) continue;
                if(std::abs(kp.pt.x-x)<r && std::abs(kp.pt.y-y)<r) vIndices.push_back(vCell[j]);
            }
        }
    return vIndices;
}

void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1, const cv::Mat &mask)
{
    vector<int> vLapping = {x0, x1};
    if(flag == 0) monoLeft = (*mpORBextractorLeft)(im, mask, mvKeys, mDescriptors, vLapping);
    else monoRight = (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight, vLapping);
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) { mTcw = Tcw; UpdatePoseMatrices(); mbIsSet = true; }
void Frame::UpdatePoseMatrices() {
    Sophus::SE3<float> Twc = mTcw.inverse(); mRwc = Twc.rotationMatrix(); mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix(); mtcw = mTcw.translation();
}
void Frame::SetVelocity(Eigen::Vector3f Vwb) { mVw = Vwb; mbHasVelocity = true; }
Eigen::Vector3f Frame::GetVelocity() const { return mVw; }
void Frame::SetNewBias(const IMU::Bias &b) { mImuBias = b; if(mpImuPreintegrated) mpImuPreintegrated->SetNewBias(b); }
bool Frame::isSet() const { return mbIsSet; }

void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1); mvDepth = vector<float>(N,-1);
    for(int i=0; i<N; i++) {
        const float d = imDepth.at<float>(mvKeys[i].pt.y, mvKeys[i].pt.x);
        if(d>0) { mvDepth[i] = d; mvuRight[i] = mvKeysUn[i].pt.x-mbf/d; }
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0) { mvKeysUn=mvKeys; return; }
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++) { mat.at<float>(i,0)=mvKeys[i].pt.x; mat.at<float>(i,1)=mvKeys[i].pt.y; }
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++) {
        cv::KeyPoint kp = mvKeys[i]; kp.pt.x=mat.at<float>(i,0); kp.pt.y=mat.at<float>(i,1); mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0) {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);
        mnMinX = std::min(mat.at<float>(0,0),mat.at<float>(2,0)); mnMaxX = std::max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = std::min(mat.at<float>(0,1),mat.at<float>(1,1)); mnMaxY = std::max(mat.at<float>(2,1),mat.at<float>(3,1));
    } else {
        mnMinX = 0.0f; mnMaxX = imLeft.cols; mnMinY = 0.0f; mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f); mvDepth = vector<float>(N,-1.0f);
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());
    for(int iR=0; iR<(int)mvKeysRight.size(); iR++) {
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        for(int yi=std::floor(mvKeysRight[iR].pt.y-r);yi<=std::ceil(mvKeysRight[iR].pt.y+r);yi++) vRowIndices[yi].push_back(iR);
    }
    const float minD = 0, maxD = mbf/mb;
    for(int iL=0; iL<N; iL++) {
        const cv::KeyPoint &kpL = mvKeys[iL]; const vector<size_t> &vCand = vRowIndices[kpL.pt.y];
        if(vCand.empty()) continue;
        int bestDist = ORBmatcher::TH_HIGH; size_t bestIdxR = 0;
        for(size_t iR : vCand) {
            if(mvKeysRight[iR].octave<kpL.octave-1 || mvKeysRight[iR].octave>kpL.octave+1) continue;
            if(mvKeysRight[iR].pt.x >= kpL.pt.x-maxD && mvKeysRight[iR].pt.x <= kpL.pt.x-minD) {
                int dist = ORBmatcher::DescriptorDistance(mDescriptors.row(iL), mDescriptorsRight.row(iR));
                if(dist<bestDist) { bestDist = dist; bestIdxR = iR; }
            }
        }
        if(bestDist< (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2) {
            mvDepth[iL]=mbf/(kpL.pt.x-mvKeysRight[bestIdxR].pt.x); mvuRight[iL] = mvKeysRight[bestIdxR].pt.x;
        }
    }
}

void Frame::ComputeStereoFishEyeMatches() {
    vector<cv::KeyPoint> sL(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> sR(mvKeysRight.begin() + monoRight, mvKeysRight.end());
    cv::Mat sDL = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat sDR = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);
    mvLeftToRightMatch = vector<int>(Nleft,-1); mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f); mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    vector<vector<cv::DMatch>> matches; BFmatcher.knnMatch(sDL,sDR,matches,2);
    for(auto &m : matches){
        if(m.size() >= 2 && m[0].distance < m[1].distance * 0.7){
            Eigen::Vector3f p3D;
            float sigma1 = mvLevelSigma2[mvKeys[m[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[m[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[m[0].queryIdx + monoLeft],mvKeysRight[m[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[m[0].queryIdx+monoLeft] = m[0].trainIdx+monoRight;
                mvRightToLeftMatch[m[0].trainIdx+monoRight] = m[0].queryIdx+monoLeft;
                mvStereo3Dpoints[m[0].queryIdx+monoLeft] = p3D; mvDepth[m[0].queryIdx+monoLeft] = depth;
            }
        }
    }
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        pMP->mbTrackInView = false; Eigen::Vector3f P = pMP->GetWorldPos();
        const Eigen::Vector3f Pc = mRcw * P + mtcw;
        if(Pc(2)<0.0f) return false;
        const Eigen::Vector2f uv = mpCamera->project(Pc);
        if(uv(0)<mnMinX || uv(0)>mnMaxX || uv(1)<mnMinY || uv(1)>mnMaxY) return false;
        float dist = (P - mOw).norm();
        if(dist<pMP->GetMinDistanceInvariance() || dist>pMP->GetMaxDistanceInvariance()) return false;
        if((P - mOw).dot(pMP->GetNormal())/dist < viewingCosLimit) return false;
        pMP->mbTrackInView = true; pMP->mTrackProjX = uv(0); pMP->mTrackProjY = uv(1);
        pMP->mTrackProjXR = uv(0) - mbf/Pc(2); pMP->mTrackDepth = Pc.norm();
        pMP->mnTrackScaleLevel= pMP->PredictScale(dist,this); pMP->mTrackViewCos = (P - mOw).dot(pMP->GetNormal())/dist;
        return true;
    } else {
        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);
        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    Eigen::Vector3f P = pMP->GetWorldPos(); Eigen::Matrix3f mR; Eigen::Vector3f mt, twc;
    if(bRight){
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix(); mR = Rrl * mRcw; mt = Rrl * mtcw + mTrl.translation();
        twc = mRwc * mTlr.translation() + mOw;
    } else { mR = mRcw; mt = mtcw; twc = mOw; }
    Eigen::Vector3f Pc = mR * P + mt; if(Pc(2)<0.0f) return false;
    Eigen::Vector2f uv = bRight ? mpCamera2->project(Pc) : mpCamera->project(Pc);
    if(uv(0)<mnMinX || uv(0)>mnMaxX || uv(1)<mnMinY || uv(1)>mnMaxY) return false;
    float dist = (P - twc).norm(); if(dist<pMP->GetMinDistanceInvariance() || dist>pMP->GetMaxDistanceInvariance()) return false;
    if((P - twc).dot(pMP->GetNormal()) / dist < viewingCosLimit) return false;
    return true;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv); posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);
    return !(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS);
}

bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D) {
    float z = mvDepth[i]; if(z>0) {
        float x = (mvKeysUn[i].pt.x-cx)*z*invfx; float y = (mvKeysUn[i].pt.y-cy)*z*invfy;
        x3D = mRwc * Eigen::Vector3f(x, y, z) + mOw; return true;
    } return false;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb) {
    mVw = Vwb; mbHasVelocity = true;
    Sophus::SE3f Twb(Rwb, twb); Sophus::SE3f Tbw = Twb.inverse();
    mTcw = mImuCalib.mTcb * Tbw; UpdatePoseMatrices();
    mbIsSet = true; mbHasPose = true;
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const { return mRwc * mImuCalib.mTcb.translation() + mOw; }
Eigen::Matrix<float,3,3> Frame::GetImuRotation() { return mRwc * mImuCalib.mTcb.rotationMatrix(); }
Sophus::SE3<float> Frame::GetImuPose() { return mTcw.inverse() * mImuCalib.mTcb; }
bool Frame::imuIsPreintegrated() { unique_lock<std::mutex> lock(*mpMutexImu); return mbImuPreintegrated; }
void Frame::setIntegrated() { unique_lock<std::mutex> lock(*mpMutexImu); mbImuPreintegrated = true; }
Sophus::SE3f Frame::GetRelativePoseTrl() { return mTrl; }
Sophus::SE3f Frame::GetRelativePoseTlr() { return mTlr; }
Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){ return mRwc * mvStereo3Dpoints[i] + mOw; }

void Frame::ComputeBoW() {
    if(mBowVec.empty()) {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

} // namespace ORB_SLAM3

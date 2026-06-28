/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <algorithm> 

#include "ORBextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
    {
        int m_01 = 0, m_10 = 0;
        const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));
        for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];
        int step = (int)image.step1();
        for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
        {
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }
        return fastAtan2((float)m_01, (float)m_10);
    }

    const float factorPI = (float)(CV_PI/180.f);
    static void computeOrbDescriptor(const KeyPoint& kpt,
                                     const Mat& img, const Point* pattern,
                                     uchar* desc)
    {
        float angle = (float)kpt.angle*factorPI;
        float a = (float)cos(angle), b = (float)sin(angle);
        const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
        const int step = (int)img.step;

#define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]

        for (int i = 0; i < 32; ++i, pattern += 16)
        {
            int t0, t1, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            val = t0 < t1;
            t0 = GET_VALUE(2); t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;
            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;
            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;
            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;
            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;
            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;
            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;
            desc[i] = (uchar)val;
        }
#undef GET_VALUE
    }

    static int bit_pattern_31_[256*4] = { 8,-3, 9,5, 4,2, 7,-12, -11,9, -8,2, 7,-12, 12,-13, 2,-13, 2,12, 1,-7, 1,6, -2,-10, -2,-4, -13,-13, -11,-8, -13,-3, -12,-9, 10,4, 11,9, -13,-8, -8,-9, -11,7, -9,12, 7,7, 12,6, -4,-5, -3,0, -13,2, -12,-3, -9,0, -7,5, 12,-6, 12,-1, -3,6, -2,12, -6,-13, -4,-8, 11,-13, 12,-8, 4,7, 5,1, 5,-3, 10,-3, 3,-7, 6,12, -8,-7, -6,-2, -2,11, -1,-10, -13,12, -8,10, -7,3, -5,-3, -4,2, -3,7, -10,-12, -6,11, 5,-12, 6,-7, 5,-6, 7,-1, 1,0, 4,-5, 9,11, 11,-13, 4,7, 4,12, 2,-1, 4,4, -4,-12, -2,7, -8,-5, -7,-10, 4,11, 9,12, 0,-8, 1,-13, -13,-2, -8,2, -3,-2, -2,3, -6,9, -4,-9, 8,12, 10,7, 0,9, 1,3, 7,-5, 11,-10, -13,-6, -11,0, 10,7, 12,1, -6,-3, -6,12, 10,-9, 12,-4, -13,8, -8,-12, -13,0, -8,-4, 3,3, 7,8, 5,7, 10,-7, -1,7, 1,-12, 3,-10, 5,6, 2,-4, 3,-10, -13,0, -13,5, -13,-7, -12,12, -13,3, -11,8, -7,12, -4,7, 6,-10, 12,8, -9,-1, -7,-6, -2,-5, 0,12, -12,5, -7,5, 3,-10, 8,-13, -7,-7, -4,5, -3,-2, -1,-7, 2,9, 5,-11, -11,-13, -5,-13, -1,6, 0,-1, 5,-3, 5,2, -4,-13, -4,12, -9,-6, -9,6, -12,-10, -8,-4, 10,2, 12,-3, 7,12, 12,12, -7,-13, -6,5, -4,9, -3,4, 7,-1, 12,2, -7,6, -5,1, -13,11, -12,5, -3,7, -2,-6, 7,-8, 12,-7, -13,-7, -11,-12, 1,-3, 12,12, 2,-6, 3,0, -4,3, -2,-13, -1,-13, 1,9, 7,1, 8,-6, 1,-1, 3,12, 9,1, 12,6, -1,-9, -1,3, -13,-13, -10,5, 7,7, 10,12, 12,-5, 12,9, 6,3, 7,11, 5,-13, 6,10, 2,-12, 2,3, 3,8, 4,-6, 2,6, 12,-13, 9,-12, 10,3, -8,4, -7,9, -11,12, -4,-6, 1,12, 2,-8, 6,-9, 7,-4, 2,3, 3,-2, 6,3, 11,0, 3,-3, 8,-8, 7,8, 9,3, -11,-5, -6,-4, -10,11, -5,10, -5,-8, -3,12, -10,5, -9,0, 8,-1, 12,-6, 4,-6, 6,-11, -10,12, -8,7, 4,-2, 6,7, -2,0, -2,12, -5,-8, -5,2, 7,-6, 10,12, -9,-13, -8,-8, -5,-13, -5,-2, 8,-8, 9,-13, -9,-11, -9,0, 1,-8, 1,-2, 7,-4, 9,1, -2,1, -1,-4, 11,-6, 12,-11, -12,-9, -6,4, 3,7, 7,12, 5,5, 10,8, 0,-4, 2,8, -9,12, -5,-13, 0,7, 2,12, -1,2, 1,7, 5,11, 7,-9, 3,5, 6,-8, -13,-4, -8,9, -5,9, -3,-3, -4,-7, -3,-12, 6,5, 8,0, -7,6, -6,12, -13,6, -5,-2, 1,-10, 3,10, 4,1, 8,-4, -2,-2, 2,-13, 2,-12, 12,12, -2,-13, 0,-6, 4,1, 9,3, -6,-10, -3,-5, -3,-13, -1,1, 7,5, 12,-11, 4,-2, 5,-7, -13,9, -9,-5, 7,1, 8,6, 7,-8, 7,6, -7,-4, -7,1, -8,11, -7,-8, -13,6, -12,-8, 2,4, 3,9, 10,-5, 12,3, -6,-5, -6,7, 8,-3, 9,-8, 2,-12, 2,8, -11,-2, -10,3, -12,-13, -7,-9, -11,0, -10,-5, 5,-3, 11,8, -2,-13, -1,12, -1,-8, 0,9, -13,-11, -12,-5, -10,-2, -10,11, -3,9, -2,-13, 2,-3, 3,2, -9,-13, -4,0, -4,6, -3,-10, -4,12, -2,-7, -6,-11, -4,9, 6,-3, 6,11, -13,11, -5,5, 11,11, 12,6, 7,-5, 12,-2, -1,12, 0,7, -4,-8, -3,-2, -7,1, -6,7, -13,-12, -8,-13, -7,-2, -6,-8, -8,5, -6,-9, -5,-1, -4,5, -13,7, -8,10, 1,5, 5,-13, 1,0, 10,-13, 9,12, 10,-1, 5,-8, 10,-9, -1,11, 1,-13, -9,-3, -6,2, -1,-10, 1,12, -13,1, -8,-10, 8,-11, 10,-6, 2,-13, 3,-6, 7,-13, 12,-9, -10,-10, -5,-7, -10,-8, -8,-13, 4,-6, 8,5, 3,12, 8,-13, -4,2, -3,-3, 5,-13, 10,-12, 4,-13, 5,-1, -9,9, -4,3, 0,3, 3,-9, -12,1, -6,1, 3,2, 4,-8, -10,-10, -10,9, 8,-13, 12,12, -8,-12, -6,-5, 2,2, 3,7, 10,6, 11,-8, 6,8, 8,-12, -7,10, -6,5, -3,-9, -3,9, -1,-13, -1,5, -3,-7, -3,4, -8,-2, -8,3, 4,2, 12,12, 2,-5, 3,11, 6,-9, 11,-13, 3,-1, 7,12, 11,-1, 12,4, -3,0, -3,6, 4,-11, 4,12, 2,-4, 2,1, -10,-6, -8,1, -13,7, -11,1, -13,12, -11,-13, 6,0, 11,-13, 0,-1, 1,4, -13,3, -9,-2, -9,8, -6,-3, -13,-6, -8,-2, 5,-9, 8,10, 2,7, 3,-9, -1,-6, -1,-1, 9,5, 11,-2, 11,-3, 12,-8, 3,0, 3,5, -1,4, 0,10, 3,-6, 4,5, -13,0, -10,5, 5,8, 12,11, 8,9, 9,-6, 7,-4, 8,-12, -10,4, -10,9, 7,3, 12,4, 9,-7, 10,-2, 7,0, 12,-2, -1,-6, 0,-11 };

    ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST):
                nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
                iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0]=1.0f;
        mvLevelSigma2[0]=1.0f;
        
        mnLastNumRejected = 0; 

        for(int i=1; i<nlevels; i++)
        {
            mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
            mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for(int i=0; i<nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);
        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));
        int sumFeatures = 0;
        for( int level = 0; level < nlevels-1; level++ )
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

        const int npoints = 512;
        const Point* pattern0 = (const Point*)bit_pattern_31_;
        std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

        umax.resize(HALF_PATCH_SIZE + 1);
        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }
    }

    static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
    {
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
            keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }

    void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
        const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);
        n1.UL = UL; n1.UR = cv::Point2i(UL.x+halfX,UL.y); n1.BL = cv::Point2i(UL.x,UL.y+halfY); n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY); n1.vKeys.reserve(vKeys.size());
        n2.UL = n1.UR; n2.UR = UR; n2.BL = n1.BR; n2.BR = cv::Point2i(UR.x,UL.y+halfY); n2.vKeys.reserve(vKeys.size());
        n3.UL = n1.BL; n3.UR = n1.BR; n3.BL = BL; n3.BR = cv::Point2i(n1.BR.x,BL.y); n3.vKeys.reserve(vKeys.size());
        n4.UL = n3.UR; n4.UR = n2.BR; n4.BL = n3.BR; n4.BR = BR; n4.vKeys.reserve(vKeys.size());
        for(size_t i=0;i<vKeys.size();i++)
        {
            const cv::KeyPoint &kp = vKeys[i];
            if(kp.pt.x<n1.UR.x)
            {
                if(kp.pt.y<n1.BR.y) n1.vKeys.push_back(kp);
                else n3.vKeys.push_back(kp);
            }
            else if(kp.pt.y<n1.BR.y) n2.vKeys.push_back(kp);
            else n4.vKeys.push_back(kp);
        }
        if(n1.vKeys.size()==1) n1.bNoMore = true;
        if(n2.vKeys.size()==1) n2.bNoMore = true;
        if(n3.vKeys.size()==1) n3.bNoMore = true;
        if(n4.vKeys.size()==1) n4.bNoMore = true;
    }

    static bool compareNodes(pair<int,ExtractorNode*>& e1, pair<int,ExtractorNode*>& e2){
        if(e1.first < e2.first) return true;
        else if(e1.first > e2.first) return false;
        else return e1.second->UL.x < e2.second->UL.x;
    }

    vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
    {
        const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));
        const float hX = static_cast<float>(maxX-minX)/nIni;
        list<ExtractorNode> lNodes;
        vector<ExtractorNode*> vpIniNodes;
        vpIniNodes.resize(nIni);
        for(int i=0; i<nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX*static_cast<float>(i),0); ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0); ni.BL = cv::Point2i(ni.UL.x,maxY-minY); ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
            ni.vKeys.reserve(vToDistributeKeys.size());
            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }
        for(size_t i=0;i<vToDistributeKeys.size();i++)
        {
            const cv::KeyPoint &kp = vToDistributeKeys[i];
            vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
        }
        list<ExtractorNode>::iterator lit = lNodes.begin();
        while(lit!=lNodes.end())
        {
            if(lit->vKeys.size()==1) { lit->bNoMore=true; lit++; }
            else if(lit->vKeys.empty()) lit = lNodes.erase(lit);
            else lit++;
        }
        bool bFinish = false;
        int iteration = 0;
        vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size()*4);
        while(!bFinish)
        {
            iteration++;
            int prevSize = lNodes.size();
            lit = lNodes.begin();
            int nToExpand = 0;
            vSizeAndPointerToNode.clear();
            while(lit!=lNodes.end())
            {
                if(lit->bNoMore) { lit++; continue; }
                else
                {
                    ExtractorNode n1,n2,n3,n4;
                    lit->DivideNode(n1,n2,n3,n4);
                    if(n1.vKeys.size()>0) { lNodes.push_front(n1); if(n1.vKeys.size()>1) { nToExpand++; vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); } }
                    if(n2.vKeys.size()>0) { lNodes.push_front(n2); if(n2.vKeys.size()>1) { nToExpand++; vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); } }
                    if(n3.vKeys.size()>0) { lNodes.push_front(n3); if(n3.vKeys.size()>1) { nToExpand++; vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); } }
                    if(n4.vKeys.size()>0) { lNodes.push_front(n4); if(n4.vKeys.size()>1) { nToExpand++; vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); } }
                    lit=lNodes.erase(lit);
                    continue;
                }
            }
            if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize) bFinish = true;
            else if(((int)lNodes.size()+nToExpand*3)>N)
            {
                while(!bFinish)
                {
                    prevSize = lNodes.size();
                    vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();
                    sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end(),compareNodes);
                    for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                    {
                        ExtractorNode n1,n2,n3,n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);
                        if(n1.vKeys.size()>0) { lNodes.push_front(n1); if(n1.vKeys.size()>1) { vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); } }
                        if(n2.vKeys.size()>0) { lNodes.push_front(n2); if(n2.vKeys.size()>1) { vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); } }
                        if(n3.vKeys.size()>0) { lNodes.push_front(n3); if(n3.vKeys.size()>1) { vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); } }
                        if(n4.vKeys.size()>0) { lNodes.push_front(n4); if(n4.vKeys.size()>1) { vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); } }
                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);
                        if((int)lNodes.size()>=N) break;
                    }
                    if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize) bFinish = true;
                }
            }
        }
        vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(nfeatures); 
        for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
        {
            vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
            cv::KeyPoint* pKP = &vNodeKeys[0];
            float maxResponse = pKP->response;
            for(size_t k=1;k<vNodeKeys.size();k++)
            {
                if(vNodeKeys[k].response>maxResponse) { pKP = &vNodeKeys[k]; maxResponse = vNodeKeys[k].response; }
            }
            vResultKeys.push_back(*pKP);
        }
        return vResultKeys;
    }

    void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
    {
        allKeypoints.resize(nlevels);
        const float W = 35; 
        
        int nThisPassRejected = 0; // 本次 Pass 剔除的点数计数器

        for (int level = 0; level < nlevels; ++level)
        {
            const int minBorderX = EDGE_THRESHOLD-3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
            const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

            vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures*10); 

            const float width = (maxBorderX-minBorderX);
            const float height = (maxBorderY-minBorderY);
            const int nCols = width/W;
            const int nRows = height/W;
            const int wCell = ceil(width/nCols);
            const int hCell = ceil(height/nRows);

            for(int i=0; i<nRows; i++)
            {
                const float iniY = minBorderY + i * hCell;
                float maxY = iniY + hCell + 6;
                if(iniY >= maxBorderY - 3) continue;
                if(maxY > maxBorderY) maxY = maxBorderY;

                for(int j=0; j<nCols; j++)
                {
                    const float iniX = minBorderX + j * wCell;
                    float maxX = iniX + wCell + 6;
                    if(iniX >= maxBorderX - 6) continue;
                    if(maxX > maxBorderX) maxX = maxBorderX;

                    vector<cv::KeyPoint> vKeysCell;
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, iniThFAST, true);
                    if(vKeysCell.empty()) FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, minThFAST, true);

                    if(!vKeysCell.empty())
                    {
                        for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end(); vit++)
                        {
                            // --- 第一级核心改进：点级别掩码过滤 ---
                            if(!mSegmentationMask.empty())
                            {
                                // 计算在原图尺度下的绝对像素坐标
                                float scale = mvScaleFactor[level];
                                int px = cvRound(((*vit).pt.x + j*wCell + minBorderX) * scale);
                                int py = cvRound(((*vit).pt.y + i*hCell + minBorderY) * scale);

                                // 越界安全检查
                                if(px >= 0 && px < mSegmentationMask.cols && py >= 0 && py < mSegmentationMask.rows)
                                {
                                    // 只要像素不为 0 (不管是 255 还是其他非零类索引)，直接剔除
                                    if(mSegmentationMask.data[py * mSegmentationMask.step + px] > 0) 
                                    {
                                        nThisPassRejected++;
                                        continue; 
                                    }
                                }
                            }

                            (*vit).pt.x += j * wCell;
                            (*vit).pt.y += i * hCell;
                            vToDistributeKeys.push_back(*vit);
                        }
                    }
                } 
            }

            allKeypoints[level] = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX, minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);
            const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];
            const int nkps = allKeypoints[level].size();
            for(int i=0; i<nkps ; i++)
            {
                allKeypoints[level][i].pt.x += minBorderX;
                allKeypoints[level][i].pt.y += minBorderY;
                allKeypoints[level][i].octave = level;
                allKeypoints[level][i].size = scaledPatchSize;
            }
        } 
        
        // 关键：将当前提取过程中的剔除总数累加到类成员中
        mnLastNumRejected += nThisPassRejected;

        for (int level = 0; level < nlevels; ++level)
            computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
    }

    void ORBextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint> > &allKeypoints)
    {
        // (省略了 Old 方法的实现，保持原始即可)
    }

    static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                                   const vector<Point>& pattern)
    {
        descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);
        for (size_t i = 0; i < keypoints.size(); i++)
            computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
    }

    int ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors, std::vector<int> &vLappingArea, InputArray _dynamicMask)
    {
        if(_image.empty()) return -1;
        
        // --- 核心防空值保护逻辑 ---
        // 只有当传入了有效的动态掩码时，才覆盖成员变量。
        // 这防止了 Frame.cc 中的默认调用（传空值）抹掉 Tracking.cc 中设置好的 Mask。
        if(!_dynamicMask.empty())
        {
            mSegmentationMask = _dynamicMask.getMat().clone();
        }

        // 每次 operator 启动时，重置上一帧的计数器
        mnLastNumRejected = 0; 

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 );
        ComputePyramid(image);
        vector < vector<KeyPoint> > allKeypoints; 
        
        // 第一次特征点提取
        ComputeKeyPointsOctTree(allKeypoints);

        int nkeypoints_pass1 = 0;
        for (int level = 0; level < nlevels; ++level)
            nkeypoints_pass1 += (int)allKeypoints[level].size();

        // 如果第一遍点不够，降低阈值提取第二遍（注意计数器会累加）
        if (nkeypoints_pass1 < nfeatures)
        {
            int original_iniThFAST = iniThFAST;
            int original_minThFAST = minThFAST;
            iniThFAST = original_minThFAST; 
            minThFAST = std::max(1, original_minThFAST / 2); 
            ComputeKeyPointsOctTree(allKeypoints);
            iniThFAST = original_iniThFAST;
            minThFAST = original_minThFAST;
        }

        Mat descriptors;
        int nkeypoints = 0;
        for (int level = 0; level < nlevels; ++level)
            nkeypoints += (int)allKeypoints[level].size();

        if( nkeypoints == 0 ) _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, 32, CV_8U);
            descriptors = _descriptors.getMat();
        }

        _keypoints.assign(nkeypoints, KeyPoint()); 

        int monoIndex = 0;
        int stereoIndex = nkeypoints-1; 
        if (nkeypoints == 0 && stereoIndex < 0) { stereoIndex = -1; }

        for (int level = 0; level < nlevels; ++level)
        {
            vector<KeyPoint>& keypoints = allKeypoints[level];
            int nkeypointsLevel = (int)keypoints.size();
            if(nkeypointsLevel==0) continue;
            Mat workingMat = mvImagePyramid[level].clone();
            GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
            Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
            computeDescriptors(workingMat, keypoints, desc, pattern);
            float scale = mvScaleFactor[level];
            int i = 0; 
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){
                if (level != 0) keypoint->pt *= scale;
                if(keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]){
                     if (stereoIndex >= 0 && stereoIndex < nkeypoints) {
                        _keypoints.at(stereoIndex) = (*keypoint);
                        desc.row(i).copyTo(descriptors.row(stereoIndex));
                        stereoIndex--;
                    }
                }
                else{
                    if (monoIndex >= 0 && monoIndex < nkeypoints) {
                        _keypoints.at(monoIndex) = (*keypoint);
                        desc.row(i).copyTo(descriptors.row(monoIndex));
                        monoIndex++;
                    }
                }
                i++;
            }
        }
        return monoIndex; 
    }

    void ORBextractor::ComputePyramid(cv::Mat image)
    {
        for (int level = 0; level < nlevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
            Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
            Mat temp(wholeSize, image.type());
            mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
            if( level != 0 )
            {
                resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
                copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101+BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101);
            }
        }
    }
} //namespace ORB_SLAM3

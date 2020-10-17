
#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>


#define PRINTF_ERROR(fmt, ...) printf(("%s(%d)\t" fmt "\n"), __FUNCTION__, __LINE__, ##__VA_ARGS__)

class GrayCodeDecoding
{
public:
    GrayCodeDecoding();
    ~GrayCodeDecoding();
    typedef std::shared_ptr<GrayCodeDecoding> Ptr;
    

    /**
    * @brief        根据正全部格雷码，算出 Contrast Code.
    * @param[in]    frames      :   全部输入图像，包括：纯亮/纯暗/格雷码/inverst 格雷码
    * @param[out]   contrastcode:   contrast code.
    */
    void ComputeContrastCode(std::vector<cv::Mat> &frames, cv::Mat &contrastcode);

    /**
    * @brief        找出相机pixel与投影仪pixel的对应关系；给每个pixel指定颜色
    * @param[in]    frames:     输入42张图.
    * @param[out]   data_rp:    投影仪中每个pixel对应的相机pixel坐标.
    * @param[out]   data_color: 颜色.
    */
    void Matching_P_C(std::vector<cv::Mat> &frames, cv::Mat &data_rp, cv::Mat &data_color);

    /**
    * @brief        找出相机pixel与投影仪pixel的对应关系；给每个pixel指定颜色
    * @param[in]    frames:     输入42张图.
    * @param[in]    undist:     畸变校正矩阵.
    * @param[out]   data_rp:    投影仪中每个pixel对应的相机pixel坐标.
    * @param[out]   data_color: 颜色.
    */
    void Matching_P_C(std::vector<cv::Mat> &frames, cv::Mat &undist, cv::Mat &data_rp, cv::Mat &data_color);

    /**
    * @brief        set threshold
    * @param[in]    thresh:     threshold, 0.002 ~ 0.2
    */
    void setDecodingThreshold(float thresh);

    /**
    * @brief        get decoding score. Depending on: 1. the percentage of valid decoding; 2...
    */
    float getDecodingScore();

    /**
    * @brief        导出投影仪像素与相机像素的对应关系
    * @param[in]    data_rp:   投影仪像素与相机像素的对应关系.
    */
    void save_pro_matching(cv::Mat data_rp);


    float               theshhold;          //阈值，用于判断该点是亮点还是暗点还是无效点
    
private:
    /**
    * @brief        根据正反两张格雷码，算出 Contrast Code.
    * @param        index       :   position that you want to compute contrast code
    * @param[in]    frame       :   格雷码
    * @param[in]    frame_inv   :   反色格雷码
    * @param[out]   contrastcode:   contrast code.
    */
    void ComputeContrastCode(unsigned int index, cv::Mat &frame, cv::Mat &frame_inv, cv::Mat &contrastcode);
    void generateForm();

    int                 NUM_OF_FRAMES;      //输入图片数，包括：纯亮/纯暗/格雷码/inverst 格雷码
    int                 NUM_OF_INDEX;       //正向格雷码的总数
    int                 Frame_Row;          //输入图像的row 
    int                 Frame_Col;          //输入图像的col
    int                 Proj_Row;           //投影仪的row
    int                 Proj_Col;           //投影仪的col
    int                 offset[2];          //由于投影仪像素并非是2^n，需要平移
    int                 form_gray[1024];    //查表：code与投影仪像素的对应关系
    float               thresh_max;
    float               thresh_min;
    float               decoding_score;

    float               cont1 = 0;          //没啥用，用于计算 （I-I_inv）/(I+I_inv) 的过程中
    float               cont2 = 0;          //没啥用，用于计算 （I-I_inv）/(I+I_inv) 的过程中
    unsigned int        flag_Y=0;           //标记当前图像是X方向编码还是Y方向编码
    int                 count_invalid = 0;  //count the number of invalid points 
    float               invalid_percentage = 0; // percentage of invalid points
};
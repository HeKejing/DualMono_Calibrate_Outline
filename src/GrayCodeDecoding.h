
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
    * @brief        ������ȫ�������룬��� Contrast Code.
    * @param[in]    frames      :   ȫ������ͼ�񣬰���������/����/������/inverst ������
    * @param[out]   contrastcode:   contrast code.
    */
    void ComputeContrastCode(std::vector<cv::Mat> &frames, cv::Mat &contrastcode);

    /**
    * @brief        �ҳ����pixel��ͶӰ��pixel�Ķ�Ӧ��ϵ����ÿ��pixelָ����ɫ
    * @param[in]    frames:     ����42��ͼ.
    * @param[out]   data_rp:    ͶӰ����ÿ��pixel��Ӧ�����pixel����.
    * @param[out]   data_color: ��ɫ.
    */
    void Matching_P_C(std::vector<cv::Mat> &frames, cv::Mat &data_rp, cv::Mat &data_color);

    /**
    * @brief        �ҳ����pixel��ͶӰ��pixel�Ķ�Ӧ��ϵ����ÿ��pixelָ����ɫ
    * @param[in]    frames:     ����42��ͼ.
    * @param[in]    undist:     ����У������.
    * @param[out]   data_rp:    ͶӰ����ÿ��pixel��Ӧ�����pixel����.
    * @param[out]   data_color: ��ɫ.
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
    * @brief        ����ͶӰ��������������صĶ�Ӧ��ϵ
    * @param[in]    data_rp:   ͶӰ��������������صĶ�Ӧ��ϵ.
    */
    void save_pro_matching(cv::Mat data_rp);


    float               theshhold;          //��ֵ�������жϸõ������㻹�ǰ��㻹����Ч��
    
private:
    /**
    * @brief        �����������Ÿ����룬��� Contrast Code.
    * @param        index       :   position that you want to compute contrast code
    * @param[in]    frame       :   ������
    * @param[in]    frame_inv   :   ��ɫ������
    * @param[out]   contrastcode:   contrast code.
    */
    void ComputeContrastCode(unsigned int index, cv::Mat &frame, cv::Mat &frame_inv, cv::Mat &contrastcode);
    void generateForm();

    int                 NUM_OF_FRAMES;      //����ͼƬ��������������/����/������/inverst ������
    int                 NUM_OF_INDEX;       //��������������
    int                 Frame_Row;          //����ͼ���row 
    int                 Frame_Col;          //����ͼ���col
    int                 Proj_Row;           //ͶӰ�ǵ�row
    int                 Proj_Col;           //ͶӰ�ǵ�col
    int                 offset[2];          //����ͶӰ�����ز�����2^n����Ҫƽ��
    int                 form_gray[1024];    //���code��ͶӰ�����صĶ�Ӧ��ϵ
    float               thresh_max;
    float               thresh_min;
    float               decoding_score;

    float               cont1 = 0;          //ûɶ�ã����ڼ��� ��I-I_inv��/(I+I_inv) �Ĺ�����
    float               cont2 = 0;          //ûɶ�ã����ڼ��� ��I-I_inv��/(I+I_inv) �Ĺ�����
    unsigned int        flag_Y=0;           //��ǵ�ǰͼ����X������뻹��Y�������
    int                 count_invalid = 0;  //count the number of invalid points 
    float               invalid_percentage = 0; // percentage of invalid points
};

#pragma once

#include <stdio.h>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>

#include "GrayCodeDecoding.h"


// output switch
#ifdef  QT_NO_DEBUG
    #define PRINTF_INFO(fmt, ...) /* do nothing */
#else
    #define PRINTF_INFO(fmt, ...) printf(("%s(%d)\t" fmt "\n"), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif
#define PRINTF_ERROR(fmt, ...) printf(("%s(%d)\t" fmt "\n"), __FUNCTION__, __LINE__, ##__VA_ARGS__)

class KJCalibration
{
public:
    KJCalibration(int boardWidth, int boardHeight, int squareSize, int projectorWidth, int projectorHeight);
    ~KJCalibration();
    typedef std::shared_ptr<KJCalibration> Ptr;
    using callbackLog = std::function<void(std::string)>;

    /**
    * @brief       set images of one position
    * @param       leftImages: images of left camera. rightImages: images of right camera.
    * @return      false: set failed. true: images are ok.
    */
    bool setOnePosition(std::vector<cv::Mat> &leftImages, std::vector<cv::Mat> &rightImages);

    /**
    * @brief       calibrate the system
    * @return      false: failed. true: success and calibration result are generated.
    * @attention   This function should be called after setOnePosition() and position number is enough.
    */
    bool calibrate();

    /**
    * @brief       clear positions that set before
    */
    void clear();

    /**
    * @brief       check whether the image has corners
    * @param       image to check
    * @return      true: image has corners; false: image have no corners
    * @attention   image should be the gray format
    */
    bool fastCheckCorners(cv::Mat &image);
    bool checkAndDrawCorners(cv::Mat &image);
    void setGreenArea(cv::Mat &cornerImage, cv::Mat &greenArea);

    /**
    * @brief       change chessboard size.
    */
    void setBoardWidth(int boardWidth);
    void setBoardHeight(int boardHeight);
    void setBoardSquareSize(int squareSize);

    /**
    * @brief       set resolution of projector.
    * @attention   only 912*1140 resolution is supported temporary.
    */
    void setProjectorResolution(int projectorWidth, int projectorHeight);

    /**
    * @brief       set position number. default is 9.
    * @attention   image number should be more than 9
    */
    void setPositionNums(int positionNums);
    
    void registerCallBackLog(callbackLog externalLog);
    void callExternalLog(char const* const fmt, ...);


    KJCalibration() = delete;
    KJCalibration(const KJCalibration&) = delete;

    void extractCameraCorners(cv::Mat &image, std::vector<cv::Point2f> &corners);
    bool extractProjectorCorners(std::vector<cv::Mat> &images, std::vector<cv::Point2f> &cameraCorners, std::vector<cv::Point2f> &projectorCorners);
    cv::Point2f sub_decode_corner_H(cv::Mat &code, cv::Point2f &corner);

    void calcObjectPoints(std::vector<std::vector<cv::Point3f> > &objectPoints);
    bool checkProjectorCornersSequence(std::vector<cv::Point2f> &preCorners, std::vector<cv::Point2f> &postCorners);


    GrayCodeDecoding::Ptr   gcdecoding;

    const int               PATTERN_NUMS  = 42;
    int                     positionNums;
    int                     cornerNums;
    int                     squareSize;
    cv::Size                board_size;
    cv::Size                camera_size;
    cv::Size                project_size;

    std::string     RESULT_DIRECTORY_NAME   = "calibration_result";
    std::string     result_directory_path;
    std::string     left_yml                = "Myleft.yml";
    std::string     right_yml               = "Myright.yml";
    std::string     rotation_txt            = "R2LI.txt";

    std::vector<cv::Point2f>                camL_points;
    std::vector<cv::Point2f>                camR_points;
    std::vector<std::vector<cv::Point2f> >  camL_corners_all;
    std::vector<std::vector<cv::Point2f> >  camR_corners_all;
    std::vector<cv::Point2f>                project_points;
    std::vector<std::vector<cv::Point2f> >  projectL_corners_all;
    std::vector<std::vector<cv::Point2f> >  projectR_corners_all;
    std::vector<std::vector<cv::Point2f> >  camL_corners_all_filtered;
    std::vector<std::vector<cv::Point2f> >  camR_corners_all_filtered;
    std::vector<std::vector<cv::Point2f> >  project_corners_all_filtered;

    callbackLog externalLog;
};



#include "KJCalibration.h"
#include <filesystem>
#include <algorithm>


KJCalibration::KJCalibration(int boardWidth, int boardHeight, int squareSize, int projectorWidth, int projectorHeight)
    : squareSize(squareSize)
{    
    positionNums = 9;
    cornerNums = boardWidth * boardHeight;

    board_size = cv::Size(boardWidth, boardHeight);
    project_size = cv::Size(projectorWidth, projectorHeight);
    gcdecoding = GrayCodeDecoding::Ptr(new GrayCodeDecoding());

    clear();

    // create result directory
    std::tr2::sys::path current_path = std::tr2::sys::current_path();
    result_directory_path = current_path.string() + "/" + RESULT_DIRECTORY_NAME;
    std::replace(result_directory_path.begin(), result_directory_path.end(), '\\', '/');
    std::tr2::sys::path result_path = std::tr2::sys::path(result_directory_path);
    bool doesResultDirExist = std::tr2::sys::exists(result_path);
    if (false == doesResultDirExist) {
        doesResultDirExist = std::tr2::sys::create_directories(result_path);
        if (false == doesResultDirExist) {
            PRINTF_ERROR("create result directory(%s) failed", result_directory_path.c_str());
        }
    }
    if (doesResultDirExist) {
        left_yml =  left_yml;
        right_yml = right_yml;
        rotation_txt =  rotation_txt;
    }

    externalLog = NULL;
}

KJCalibration::~KJCalibration()
{
}

bool KJCalibration::setOnePosition(std::vector<cv::Mat> &leftImages, std::vector<cv::Mat> &rightImages)
{
    camera_size = cv::Size(leftImages[0].cols, leftImages[0].rows);
    

    // left cam_corner
    extractCameraCorners(leftImages[0], camL_points);


    // right cam_corner
    extractCameraCorners(rightImages[0], camR_points);


    // left projector decode 
    if (false == extractProjectorCorners(leftImages, camL_points, project_points)) 
    {
        std:: cout << "Left extractProjectorCorners error";
        return false;
    }
    //projectL_corners_all.push_back(project_points);   // use later

    std::vector<cv::Point2f> projectorCornersL;
    if (false == checkProjectorCornersSequence(project_points, projectorCornersL)) 
    {
        std::cout << "Match_Left Error!";
        return false;
    }


    // right projector decode 
    if (false == extractProjectorCorners(rightImages, camR_points, project_points))
    {
        std::cout << "Right extractProjectorCorners error";
        return false;
    }
    //projectR_corners_all.push_back(project_points); // use later

    std::vector<cv::Point2f> projectorCornersR;
    if (false == checkProjectorCornersSequence(project_points, projectorCornersR)) {
        std::cout << "Match_Right Error!";
        return false;
    }
    
    // push_back data
    camL_corners_all.push_back(camL_points);
    camR_corners_all.push_back(camR_points);
    projectL_corners_all.push_back(projectorCornersL);
    projectR_corners_all.push_back(projectorCornersR);

    return true;
}

bool KJCalibration::calibrate()
{
    if (camL_corners_all.size() < positionNums) {
        std::cout << "positions are not enough.";
        return false;
    }

    std::cout << ("Start YML build!");
    
    std::vector<std::vector<cv::Point3f> > objectPointsAll;
    std::vector<std::vector<cv::Point3f> > objectPointsAll_filtered;
    std::vector<cv::Point3f> object_points;
    calcObjectPoints(objectPointsAll);

    camL_corners_all_filtered.clear();
    camR_corners_all_filtered.clear();
    project_corners_all_filtered.clear();
    objectPointsAll_filtered.clear();
    
    for (int k = 0; k < positionNums; k++) {
        camL_points.clear();
        camR_points.clear();
        project_points.clear();
        object_points.clear();
        for (int i = 0; i < cornerNums; i++) {
            double dist = 0;
            cv::Point2f pro_point, proR_point;
            pro_point = projectL_corners_all[k][i];
            proR_point = projectR_corners_all[k][i];

            dist = sqrt((pro_point.x - proR_point.x)*(pro_point.x - proR_point.x) + (pro_point.y - proR_point.y)*(pro_point.y - proR_point.y));

            //not good Point
            if (dist > 1)
            {
                std::cout << "distance between projector corners: Pos: " << k << "Corner : " << i << " is two big" << std::endl;
                continue;
            }
            camL_points.push_back(camL_corners_all[k][i]);
            camR_points.push_back(camR_corners_all[k][i]);
            project_points.push_back(pro_point);
            object_points.push_back(objectPointsAll[k][i]);
        }
        camL_corners_all_filtered.push_back(camL_points);
        camR_corners_all_filtered.push_back(camR_points);
        project_corners_all_filtered.push_back(project_points);
        objectPointsAll_filtered.push_back(object_points);
    }

    cv::Mat cameraMatrix_L =    cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Cd_L           =    cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0));
    cv::Mat cameraMatrix_R =    cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Cd_R           =    cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0));
    cv::Mat projectMatrix  =    cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Pd             =    cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0));
    cv::Mat P1, P2, P3, P4, P5, P6, E, F;
    std::vector<cv::Mat> Rc_L, Tc_L, Rp, Tp, Rc_R, Tc_R;
    cv::Mat R_L, T_L, R_R, T_R;
    double camL_error    = cv::calibrateCamera(objectPointsAll_filtered, camL_corners_all_filtered, camera_size, cameraMatrix_L, Cd_L, Rc_L, Tc_L, 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, DBL_EPSILON));
    double pro_error     = cv::calibrateCamera(objectPointsAll_filtered, project_corners_all_filtered, project_size, projectMatrix, Pd, Rp, Tp, 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, DBL_EPSILON));
    double camR_error    = cv::calibrateCamera(objectPointsAll_filtered, camR_corners_all_filtered, camera_size, cameraMatrix_R, Cd_R, Rc_R, Tc_R, 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, DBL_EPSILON));
    double cam_two_error = cv::stereoCalibrate(objectPointsAll_filtered, camL_corners_all_filtered, camR_corners_all_filtered, cameraMatrix_L, Cd_L, cameraMatrix_R, Cd_R, camera_size, P5, P6, E, F, CV_CALIB_FIX_INTRINSIC, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 150, DBL_EPSILON));
	/* To get better fitness, left system use CV_CALIB_USE_INTRINSIC_GUESS*/
    double stereoL_error = cv::stereoCalibrate(objectPointsAll_filtered, camL_corners_all_filtered, project_corners_all_filtered, cameraMatrix_L, Cd_L, projectMatrix, Pd, camera_size, R_L, T_L, E, F, CV_CALIB_USE_INTRINSIC_GUESS, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 150, DBL_EPSILON));
	/* To cooperate with left system, right system use CV_CALIB_FIX_INTRINSIC. The sequence can not be overturned*/
    double stereoR_error = cv::stereoCalibrate(objectPointsAll_filtered, camR_corners_all_filtered, project_corners_all_filtered, cameraMatrix_R, Cd_R, projectMatrix, Pd, camera_size, R_R, T_R, E, F, CV_CALIB_FIX_INTRINSIC, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 150, DBL_EPSILON));

    //callExternalLog("stereoL_error : %llf", stereoL_error);
    //callExternalLog("stereoR_error : %llf", stereoR_error);

    // save_calibration_yml
    cv::FileStorage fs(left_yml, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cout << ("fs.isOpened() failed.");
    }
    else {
        fs << "cam_K" << cameraMatrix_L << "cam_kc" << Cd_L
            << "proj_K" << projectMatrix << "proj_kc" << Pd
            << "R" << R_L << "T" << T_L << "cam_error" << camL_error << "proj_error" << pro_error << "stereo_error" << stereoL_error;
        fs.release();
    }

    cv::FileStorage fs1(right_yml, cv::FileStorage::WRITE);
    if (!fs1.isOpened()) {
        std::cout << ("fs1.isOpened() failed.");
    }
    else {
        fs1 << "cam_K" << cameraMatrix_R << "cam_kc" << Cd_R
            << "proj_K" << projectMatrix << "proj_kc" << Pd
            << "R" << R_R << "T" << T_R << "cam_error" << camR_error << "proj_error" << pro_error << "stereo_error" << stereoR_error;
        fs1.release();    
    }

    // set registry

    std::cout << "stereoL_error:  " << stereoL_error <<std::endl;
    std::cout << "stereoR_error:  " << stereoR_error << std::endl;
    std::cout << ("YML save Complete!");

    return true;
}

void KJCalibration::clear()
{
    camL_corners_all.clear();
    camR_corners_all.clear();
    projectL_corners_all.clear();
    projectR_corners_all.clear();
}

bool KJCalibration::fastCheckCorners(cv::Mat &image)
{
    std::vector<cv::Point2f> corners;
    return cv::findChessboardCorners(image, board_size, corners, cv::CALIB_CB_FAST_CHECK);
}

bool KJCalibration::checkAndDrawCorners(cv::Mat &image)
{
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(image, board_size, corners, cv::CALIB_CB_FAST_CHECK);
    if (found) { drawChessboardCorners(image, board_size, corners, found); }

    return found;
}

void KJCalibration::setGreenArea(cv::Mat &cornerImage, cv::Mat &greenArea)
{
    std::vector<cv::Point2f> corners;
    cv::findChessboardCorners(cornerImage, board_size, corners, cv::CALIB_CB_FAST_CHECK);
    std::vector<cv::Point> leftPoints;
    const int index[4] = { 0, board_size.width - 1, board_size.width*board_size.height - 1, board_size.width*board_size.height - board_size.width };
    for (int i = 0; i < 4; i++) {
        leftPoints.push_back(corners[index[i]]);
    }

    // fill poly
    const int greenChannelValue = 80;
    const int numOfPoints = 4;
    cv::Point pointsLeft[1][numOfPoints];

    for (int i = 0; i < numOfPoints; i++) {
        pointsLeft[0][i] = leftPoints[i];
    }

    int npt[1] = { numOfPoints };
    const cv::Point *pptLeft[1] = { pointsLeft[0] };

    fillPoly(greenArea, pptLeft, npt, 1, cv::Scalar(0, greenChannelValue, 0));
}

void KJCalibration::setBoardWidth(int boardWidth)
{
    board_size = cv::Size(boardWidth, board_size.height);
}

void KJCalibration::setBoardHeight(int boardHeight)
{
    board_size = cv::Size(board_size.width, boardHeight);
}

void KJCalibration::setBoardSquareSize(int squareSize)
{
    this->squareSize = squareSize;
}

void KJCalibration::setProjectorResolution(int projectorWidth, int projectorHeight)
{
    project_size = cv::Size(projectorWidth, projectorHeight);
}

void KJCalibration::setPositionNums(int positionNums)
{
    const int MINIMUM_POSITION_NUMS = 9;
    if (positionNums < MINIMUM_POSITION_NUMS) {
        std::cout << ("positionNums is less than MINIMUM_POSITION_NUMS(%d)", MINIMUM_POSITION_NUMS);
        return; 
    }

    this->positionNums = positionNums;
}

void KJCalibration::registerCallBackLog(callbackLog externalLog)
{
    this->externalLog = externalLog;
}

void KJCalibration::callExternalLog(char const* const fmt, ...)
{
    if (!externalLog) { return; }

    char content[400];  // attention 400 limit
    va_list arg;
    va_start(arg, fmt);
    vsprintf(content, fmt, arg);
    va_end(arg);

    externalLog(content);
}

void KJCalibration::extractCameraCorners(cv::Mat &image, std::vector<cv::Point2f> &corners)
{
    cv::findChessboardCorners(image, board_size, corners);
    cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
}

bool KJCalibration::extractProjectorCorners(std::vector<cv::Mat> &images, std::vector<cv::Point2f> &cameraCorners, std::vector<cv::Point2f> &projectorCorners)
{
    cv::Mat contrastCode;
    // Projector Decode
    gcdecoding->ComputeContrastCode(images, contrastCode);

    // Projector Corner Output
    projectorCorners.swap(std::vector<cv::Point2f>());
    projectorCorners.resize(cornerNums, cv::Point2f(0, 0));
    for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
            int num_index = i*board_size.width + j; 
            cv::Point2f Pro_corner_temp;
            Pro_corner_temp = sub_decode_corner_H(contrastCode, cameraCorners[num_index]);
            if (Pro_corner_temp.x != 0 || Pro_corner_temp.y != 0) {
                projectorCorners[num_index] = Pro_corner_temp;
            }
            else {
                std::cout << ("extract projector error.");
                return false;
            }
        }
    }

    return true;
}

cv::Point2f KJCalibration::sub_decode_corner_H(cv::Mat &code, cv::Point2f &corner)
{
    cv::Point2f Pro_corner(0, 0);
    std::vector<cv::Point2f> img_points, proj_points;
    int center_x, center_y, L = 20, index = 0;
    int min_index = (2*L)*(2*L) / 10;
    //Initial
    Pro_corner.x = 0;
    Pro_corner.y = 0;
    center_x = round(corner.x);
    center_y = round(corner.y);
    // Estimate Homography Matrix H
    for (int k = -L; k < L + 1; k++) {
        for (int m = -L; m < L + 1; m++) {
            if ((center_x + m) > -1 && (center_x + m) < code.cols && (center_y + k) > -1 && (center_y + k) < code.rows) {
                if (code.at<cv::Vec2i>(center_y + k, center_x + m)[0] >= 0 
                    && code.at<cv::Vec2i>(center_y + k, center_x + m)[0] < project_size.width 
                    && code.at<cv::Vec2i>(center_y + k, center_x + m)[1] >= 0 
                    && code.at<cv::Vec2i>(center_y + k, center_x + m)[1] < project_size.height) {
                    img_points.push_back(cv::Point2f(center_x + k, center_y + m));
                    proj_points.push_back(cv::Point2f(code.at<cv::Vec2i>(center_y + k, center_x + m)[0], code.at<cv::Vec2i>(center_y + k, center_x + m)[1]));
                    index += 1;
                }
            }
        }
    }
    if (index < min_index) {
        std::cout << ("sub_decode_corner_H index is too small");
        return Pro_corner;
    }

    cv::Mat H = cv::findHomography(img_points, proj_points, CV_RANSAC);
    if (H.empty()) {
        std::cout << ("Homography matrix is empty");
        return Pro_corner;
    }

    H.convertTo(H, CV_32FC1);
    cv::Mat Q(3, 1, CV_32FC1), oldp(3, 1, CV_32FC1);

    oldp.at<float>(0, 0) = corner.x;
    oldp.at<float>(1, 0) = corner.y;
    oldp.at<float>(2, 0) = 1.0;
    Q = H * oldp;

    Pro_corner.x = Q.at<float>(0, 0) / Q.at<float>(2, 0);
    Pro_corner.y = Q.at<float>(1, 0) / Q.at<float>(2, 0);
    return Pro_corner;
}

void KJCalibration::calcObjectPoints(std::vector<std::vector<cv::Point3f> > &objectPoints)
{
    std::vector<cv::Point3f> object_xyz;
    for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
            object_xyz.push_back(cv::Point3f(float(j * squareSize), float(i * squareSize), 0));
        }
    }
    for (int i = 0; i < positionNums; ++i) {
        objectPoints.push_back(object_xyz);
    }
}
bool KJCalibration::checkProjectorCornersSequence(std::vector<cv::Point2f> &preCorners, std::vector<cv::Point2f> &postCorners)
{
    postCorners.clear();
    for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
            int num_index = i * board_size.width + j;
            if (preCorners[num_index].x != 0 || preCorners[num_index].y != 0) {
                postCorners.push_back(preCorners[num_index]);
            }
            else {
                std::cout << ("x or y of projector corner  is 0");
                return false;
            }
        }
    }
    return true;
}


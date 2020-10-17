#include "GrayCodeDecoding.h"
#include "KJCalibration.h"

using namespace std;
using namespace cv;
int main()
{
    string File_path = "G:\\Phd_Research\\2020\\colorcamera_calibate_img\\calib\\";
    string ImgL_nameformat = "\\l_";
    string ImgR_nameformat = "\\r_";
    vector<cv::Mat> leftImages, rightImages;
    int PositionNums = 11;

    KJCalibration Calib_Function(11, 8, 30, 912, 1140 / 2);
    Calib_Function.setPositionNums(PositionNums);
    //Calib_Function.
    for (int i = 0; i < PositionNums; i++)
    {
        cout << "PostionNumber Processing: " << i << endl;
        for (int k = 0; k < 50; k++)
        {
            Mat leftImage = imread(File_path + std::to_string(i) + ImgL_nameformat + std::to_string(k) + ".bmp",0);
            Mat rightImage = imread(File_path + std::to_string(i) + ImgR_nameformat + std::to_string(k) + ".bmp", 0);
            //cout << File_path + ImgR_nameformat + std::to_string(k) + ".bmp" << endl;
            //cout << rightImage.size() << endl;
            leftImages.push_back(leftImage);
            rightImages.push_back(rightImage);
        }
        cout << "Image Reading Finish " << i << endl;

        Calib_Function.setOnePosition(leftImages, rightImages);

        leftImages.clear();
        rightImages.clear();
    }

    Calib_Function.calibrate();

    cout << "1" << endl;
    system("pause");
    return 0;
}

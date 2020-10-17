
#include "GrayCodeDecoding.h"

GrayCodeDecoding::GrayCodeDecoding()
{
    NUM_OF_FRAMES = 42; //total number of frames, including X,Y,inverse frames
    NUM_OF_INDEX=10; //total number of index, NOT including Y and inverse frames

    Proj_Row = 1140 / 2;
    Proj_Col = 912 ;

    thresh_max = 0.2;
    thresh_min = 0.002;
    theshhold = 0.01;
    
    offset[0] =  ((1 << NUM_OF_INDEX) - Proj_Col) / 2;
    offset[1] =  ((1 << (NUM_OF_INDEX )) - Proj_Row) / 2;

    generateForm();
}

GrayCodeDecoding::~GrayCodeDecoding()
{
}

void GrayCodeDecoding::ComputeContrastCode(std::vector<cv::Mat> &frames, cv::Mat & contrastcode)
{
    //���˴����봿���⣬�������ͼ�������ܱ�4�������򷵻�
    int num = frames.size();
    if (50 != num) {
        PRINTF_ERROR("Number of frames should be %d!", 50);
        return;
    }

    //XY����ֱ����Contrast code
    Frame_Row = frames[0].rows;
    Frame_Col = frames[0].cols;
    contrastcode = cv::Mat(Frame_Row, Frame_Col, CV_32SC2, cv::Scalar::all(0));
    flag_Y = 0;
    for (int i = 0; i < NUM_OF_INDEX; i++) {
        ComputeContrastCode(i, frames[2 * i + 2], frames[2 * i + 3], contrastcode);
    }
    flag_Y = 1;
    for (int i = 0; i < NUM_OF_INDEX ; i++) {
        ComputeContrastCode(i, frames[2 * i + 22], frames[2 * i + 23], contrastcode);
    }
}

// frame frame_inv CV_8UC1
void GrayCodeDecoding::ComputeContrastCode(unsigned int index, cv::Mat &frame, cv::Mat &frame_inv, cv::Mat & contrastcode)
{
    if (index >= NUM_OF_INDEX) {
        PRINTF_ERROR("index is out of the range");
        return;
    }
    for (int i = 0; i < Frame_Row; i++) {
        uchar *data_A = frame.ptr<uchar>(i); // 
        uchar *data_B = frame_inv.ptr<uchar>(i);
        int *data_code = contrastcode.ptr<int>(i);
        for (int j = 0; j < Frame_Col; j++) {
            cont2 = float(data_A[j] + data_B[j]);

            // ���㣨I-I_inv��/(I+I_inv)
            if (cont2 == 0) {
                cont1 = 0; // I+I_inv=0�����
            }
            else {
                cont1 = float(data_A[j] - data_B[j]) / cont2; // ���㣨I-I_inv��/(I+I_inv)
            }

            // �жϣ��� or �������
            if (data_code[2 * j + flag_Y] < -1) {                       // the point has already been labeled to be invalid
                continue;
            }
            else if ((cont1 < theshhold) && (cont1 > -theshhold)) {    // |cont1|< threshold
                data_code[2 * j + flag_Y] = -10; //��Ϊ��Ч��
                count_invalid++;
                continue;
            }
            else if (cont1 >= theshhold) {
                data_code[2 * j + flag_Y] += (1 << (NUM_OF_INDEX - index - 1)); // ��Ч����1����(NUM_OF_INDEX - index - 1)λ�����൱��2��(NUM_OF_INDEX - index - 1)�η���
            }

            // ������תΪcode
            if (index == (NUM_OF_INDEX  - 1) && data_code[2 * j + flag_Y] > 0) {
                data_code[2 * j + flag_Y] = form_gray[data_code[2 * j + flag_Y]] - offset[flag_Y]; // ���
            }
        }
    }
    
}

void GrayCodeDecoding::Matching_P_C(std::vector<cv::Mat> &frames, cv::Mat & data_rp, cv::Mat & data_color)
{
    //����contrast code
    cv::Mat contrastcode;
    ComputeContrastCode(frames, contrastcode);
    
    for (int i = 0; i < Frame_Row; i++) {
        int *data_code = contrastcode.ptr<int>(i);
        uchar *frame_data = frames[0].ptr<uchar>(i); 
        for (int j = 0; j < Frame_Col; j++) {
            if (data_code[2 * j] > 0 && data_code[2 * j + 1] > 0) {
                if (data_code[2 * j] < Proj_Col && data_code[2 * j + 1] < Proj_Row) {
                    float *data_pro = data_rp.ptr<float>(data_code[2 * j + 1]);
                    uchar *point_color = data_color.ptr<uchar>(data_code[2 * j + 1]);

                    data_pro[3 * data_code[2 * j]] += j;            //ƥ��X����
                    data_pro[3 * data_code[2 * j] + 1] += i;        //ƥ��Y����
                    data_pro[3 * data_code[2 * j] + 2] += 1;        //ƥ�����ظ�����1
                    point_color[data_code[2 * j]] = frame_data[j];  //��¼��ɫ
                }
            }
        }
    }
}

void GrayCodeDecoding::Matching_P_C(std::vector<cv::Mat> &frames, cv::Mat &undist, cv::Mat &data_rp, cv::Mat &data_color)
{
    //����contrast code
    cv::Mat contrastcode;
    ComputeContrastCode(frames, contrastcode);

    for (int i = 0; i < Frame_Row; i++) {
        float *data_xy = undist.ptr<float>(i);//����reference
        int *data_code = contrastcode.ptr<int>(i);
        uchar *frame_data = frames[0].ptr<uchar>(i);
        for (int j = 0; j < Frame_Col; j++) {
            if (data_code[2 * j] > 0 && data_code[2 * j + 1] > 0) {
                if (data_code[2 * j] < Proj_Col && data_code[2 * j + 1] < Proj_Row) {
                    float *data_pro = data_rp.ptr<float>(data_code[2 * j + 1]);
                    uchar *point_color = data_color.ptr<uchar>(data_code[2 * j + 1]); // 

                    data_pro[3 * data_code[2 * j]] += data_xy[2 * j + 1];//j; //ƥ��X����
                    data_pro[3 * data_code[2 * j] + 1] += data_xy[2 * j]; //i; //ƥ��Y����
                    data_pro[3 * data_code[2 * j] + 2] += 1; //ƥ�����ظ�����1
                    point_color[data_code[2 * j]] = frame_data[j];  //��ɫ��¼
                }
            }
        }
    }
}

void GrayCodeDecoding::setDecodingThreshold(float thresh)
{
    if (thresh > thresh_max || thresh<thresh_min) {
        PRINTF_ERROR("threshold should be withn [%f, %f]",thresh_min,thresh_max);
        return;
    }
    theshhold = thresh;
}

float GrayCodeDecoding::getDecodingScore()
{
    invalid_percentage = float(count_invalid) / float(Frame_Row*Frame_Col*2);
    decoding_score = (1.0- invalid_percentage)*100;

    return decoding_score;
}

void GrayCodeDecoding::generateForm()
{
    for (int i = 0; i < 1024; ++i) {
        int x = i;
        int y = x;
        while (x >>= 1) {
            y ^= x;
        }

        form_gray[i] = y;
    }
}

void GrayCodeDecoding::save_pro_matching(cv::Mat data_rp)
{
    cv::FileStorage fs2("match_projector.xml", cv::FileStorage::WRITE);
    fs2 << "match_projector" << data_rp;
    fs2.release();
}

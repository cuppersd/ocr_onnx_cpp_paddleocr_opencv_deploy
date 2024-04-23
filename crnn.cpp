#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <numeric>

using namespace std;


void crnn_resize_img(cv::Mat& src_img, cv::Mat& resize_img) {
    int imgC, imgH, imgW;
    imgC = 3;
    imgH = 48;
    imgW = 320;
    float ratio = float(src_img.cols) / float(src_img.rows);
    int resize_w, resize_h;
    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));
    cv::resize(src_img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,cv::INTER_LINEAR);
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,int(imgW - resize_img.cols), cv::BORDER_CONSTANT,{ 0.0, 0.0, 0.0 });
}

cv::Mat norm_image(cv::Mat& src_img) {
    cv::Mat input_image;
    src_img.convertTo(input_image, CV_32FC3);
    input_image = input_image / 255.0;
    input_image = input_image - cv::Scalar(0.5f, 0.5f, 0.5f);
    input_image = input_image / cv::Scalar(0.5f, 0.5f, 0.5f);
    return input_image;
}

vector<string> get_alphabet(const std::string& pathTxt) {
    ifstream ifs(pathTxt);
    vector<string> alphabet;
    string line;
    while (getline(ifs, line))
    {
        alphabet.push_back(line);
    }
    alphabet.push_back(" ");
    return alphabet;
}


string postprocess(cv::Mat& output, vector<string>& alphabet) {
    int i = 0, j = 0;
    int h = output.size[2];
    int w = output.size[1];
    vector<int> preb_label;
    preb_label.resize(w);
    for (i = 0; i < w; i++)
    {
        int one_label_idx = 0;
        float max_data = -10000;
        for (j = 0; j < h; j++)
        {
            float data_ = output.at<float>(0, i, j);
            if (data_ > max_data)
            {
                max_data = data_;
                one_label_idx = j;
            }
        }
        preb_label[i] = one_label_idx;
    }
    vector<int> no_repeat_blank_label;
    for (size_t elementIndex = 0; elementIndex < w; ++elementIndex)
    {
        if (preb_label[elementIndex] != 0 && !(elementIndex > 0 && preb_label[elementIndex - 1] == preb_label[elementIndex]))
        {
            no_repeat_blank_label.push_back(preb_label[elementIndex] - 1);
        }
    }
    int len_s = no_repeat_blank_label.size();
    string plate_text;
    for (i = 0; i < len_s; i++)
    {
        plate_text += alphabet[no_repeat_blank_label[i]];
    }
    return plate_text;
}


int main()
{
    cv::dnn::Net net = cv::dnn::readNetFromONNX("D:/CPP/opencv_pro/opencv03/new_model.onnx");
    vector<string> alphabet = get_alphabet("D:/CPP/opencv_pro/opencv03/ppocr_keys_v1.txt");
    cv::Mat origin_image = cv::imread("D:/CPP/opencv_pro/opencv03/test/nin_3_2022-06-25F2AB7581-D8E2-4382-B821-12A25E2B5BC5.jpg");

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    crnn_resize_img(origin_image, origin_image);
    cv::Mat input_image = norm_image(origin_image);

    input_image = cv::dnn::blobFromImage(input_image, 1.0, cv::Size{ 320, 48 }, cv::Scalar(), true, false, CV_32F);
    net.setInput(input_image);
    std::vector<cv::Mat> output;
    net.forward(output, net.getUnconnectedOutLayersNames());

    string res = postprocess(output[0], alphabet);
    cout << res << endl;

    return 0;
}
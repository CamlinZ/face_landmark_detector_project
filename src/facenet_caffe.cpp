//
// Created by ftdhaowang on 18-6-9.
//

#define CPU_ONLY
#include "facenet_caffe.h"
#include "caffe/caffe.hpp"
#include "fstream"

using namespace caffe;

//static  shared_ptr<caffe::Net<float> > net_;

void FacenetCaffe::SetMean(const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_value.empty()) {
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) << "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                            cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

void FacenetCaffe::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

//预处理函数
void FacenetCaffe::Preprocess(const cv::Mat& img,
                          std::vector<cv::Mat>* input_channels) {
    //颜色通道转化
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    //将图像数据写入网路
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}


int FacenetCaffe::initModel(const string &model_path, const string &weights_path, const string &mean_value){
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    net_.reset(new Net<float>(model_path, caffe::TEST));
    net_->CopyTrainedLayersFrom(weights_path);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    //网路设置均值
    SetMean(mean_value);
    
    cout << "box init ok" << endl;
    return 1;
}

vector< vector<float> > FacenetCaffe::detectFace(const cv::Mat &img) {
    vector< vector<float> > detections;
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);

//网路forward操作
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);

    net_->Forward();

    //打印结果
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
//    cout << "num_det " << num_det << endl;
//    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
//    cout << "detections size: " << detections.size() << endl;
    return detections;
}

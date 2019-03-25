//
//  main.cpp
//  landmark_detec
//
//  Created by Camlin Zhang on 2018/7/13.
//

#include "face_landmark_detection.h"

int main() {

    FacenetCaffe fc_box;
    //const string& caffe_model_path ="../model/faceboxes_0718.prototxt";
    //const string& caffe_weights_path = "../model/faceboxes_0718.caffemodel";
    //const string& caffe_model_path ="../model/faceboxe_0724.prototxt";
    //const string& caffe_weights_path = "../model/faceboxe_0724.caffemodel";
    const string& caffe_model_path ="../model/final/faceboxes_deploy.prototxt";
    const string& caffe_weights_path = "../model/final/faceboxes_iter_120000.caffemodel";  
    const string& caffe_mean_value = "104,117,123";
    
    fc_box.initModel(caffe_model_path, caffe_weights_path, caffe_mean_value);
    
    FaceLandmarkDetector fc_point;
    const string dlib_model_path = "../model/predictor.dat"; 
    const string vedio_root = "../image_examples/001/";
    const string img_path = "../image_examples/test.jpg";
    const string pts_path = "../image_examples/test.pts";
    const string output_path = "../image_examples/result.jpg";
 
    fc_point.init(dlib_model_path);
    //fc_point.TestImageLandmark(fc_box, img_path, pts_path, output_path);
    fc_point.TestVedioLandmark(fc_box, vedio_root);
    //delete fc_box;
    //fc_box = NULL;

}





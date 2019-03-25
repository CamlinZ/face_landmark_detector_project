#include "face_landmark_detection.h"

FaceLandmarkDetector::FaceLandmarkDetector() { }

FaceLandmarkDetector::~FaceLandmarkDetector() { }

void FaceLandmarkDetector::init(const string model_path) {
   
    // Load pose estimation models.
    dlib::deserialize(model_path) >> pose_model;
    
    // init kalman filter param
    kalman_bounding = cv::KalmanFilter(6,2,0);
    kalman_eyes = cv::KalmanFilter(6,2,0);
    kalman_mouth = cv::KalmanFilter(6,2,0);
    kalman_other_part = cv::KalmanFilter(6,2,0);
    
    bounding_reset = true;
    mouth_reset = true;
    eyes_reset = true;
    other_part_reset = true;
    count = 0;
    
    cout << "init ok" << endl;
}

int FaceLandmarkDetector::GetFaceBox(FacenetCaffe fc_box, cv::Mat image, std::vector<std::vector<int>> & face_box, float confidence_threshold, double & time_detec){
   
    //compute detect run time(time unit is "ms")
    timeval start, end;
    gettimeofday(&start, NULL); 
    std::vector<std::vector<float>> face_box_temp = fc_box.detectFace(image);
    gettimeofday(&end, NULL);
    long cost = (end.tv_sec  - start.tv_sec) * 1000 + (end.tv_usec- start.tv_usec)/1000;
    time_detec = double(cost)/1000;

    std::vector<int> face_box_single(4, 0);    
    // detect result is: [image_id, label, score, xmin, ymin, xmax, ymax].
    for (int i = 0; i < face_box_temp.size(); ++i){
        if (face_box_temp[i][2] > confidence_threshold) {
            face_box_single[0] = int(face_box_temp[i][3] * image.cols);
            face_box_single[1] = int(face_box_temp[i][4] * image.rows);
            face_box_single[2] = int(face_box_temp[i][5] * image.cols);
            face_box_single[3] = int(face_box_temp[i][6] * image.rows);
            face_box.push_back(face_box_single);
        }
    }
    
    // check if the face_box contain some empty bounding box
    for (int n = 0 ; n < face_box.size(); ++n) {
       if (face_box[n][0] == 0 && face_box[n][1] == 0 && face_box[n][2] == 0 && face_box[n][3] == 0) {
            cout << "not detect this face!!!" << endl;
            //return -1;
        }
    }
    
    face_box_temp.clear();
    face_box_single.clear();
    return 0;
}

int FaceLandmarkDetector::AdjustFaceBox(cv::Mat image, std::vector<std::vector<int>> & face_box){
    
    for (int i = 0; i < face_box.size(); ++i) {
        int left_x = face_box[i][0];
        int top_y = face_box[i][1];
        int right_x = face_box[i][2];
        int bottom_y = face_box[i][3];
        
        int box_width = right_x - left_x;
        int box_height = bottom_y - top_y;
        int diff = box_height - box_width;
        int delta = int(abs(diff) / 2);
        
        // move bounding box "(height - width)/2"
        top_y += delta;
        bottom_y += delta;
        
        // make bounding box square
        if(diff == 0)
            return 0;
        else if(diff > 0){
            left_x -= delta;
            right_x += delta;
            if(diff % 2 == 1) right_x += 1;
        }
        else{
            top_y -= delta;
            bottom_y += delta;
            if(diff % 2 == 1) bottom_y += 1;
        }
        
        // check if the box is in image
        if (left_x < 0 || top_y < 0 || right_x > image.cols || bottom_y > image.rows) {
            // try to move the bounding box and make it in image
            if(right_x - left_x <= image.cols && bottom_y - top_y <= image.rows){
                if(left_x < 0){  // left edge crossed, move right.
                    right_x += abs(left_x);
                    left_x = 0;
                }
                if(right_x > image.cols){  // right edge crossed, move left.
                    left_x -= (right_x - image.cols);
                    right_x = image.cols;
                }
                if(top_y < 0) { // top edge crossed, move down.
                    bottom_y += abs(top_y);
                    top_y = 0;
                }
                if(bottom_y > image.rows){  // bottom edge crossed, move up.
                    top_y -= (bottom_y - image.rows);
                    bottom_y = image.rows;
                }
                // check if the box is in image
                if (left_x < 0 || top_y < 0 || right_x > image.cols || bottom_y > image.rows) {
                    cout << "input image is illegal！！！" << endl;
                    //return -1;
                }
            }
        }
        face_box[i][0] = left_x;
        face_box[i][1] = top_y;
        face_box[i][2] = right_x;
        face_box[i][3] = bottom_y;
    }
    return 0;
}

int FaceLandmarkDetector::ImageStandard(float & rate_w, float & rate_h, cv::Mat frame, std::vector<std::vector<int>> face_box, std::vector<cv::Mat> & img_resize_vector) {

    // crop face image according to the bounding box and resize the crop image to 224x224
    for (int i = 0; i < face_box.size(); ++i) {
        cv::Mat img_roi = frame(cv::Rect(face_box[i][0], face_box[i][1], face_box[i][2] - face_box[i][0], face_box[i][3] - face_box[i][1]));
        cv::Mat img_resize;
        int cols = face_box[i][2] - face_box[i][0];
        int rows = face_box[i][3] - face_box[i][1];
        if (cols != 224 || rows != 224) {
            cv::resize(img_roi, img_resize, Size(224, 224));
            rate_w = double(224) / double(cols);
            rate_h = double(224) / double(rows);
        }
        img_resize_vector.push_back(img_resize);
        
        img_roi.release();
        img_resize.release();
    }
}

int FaceLandmarkDetector::detectFaceLandmark(float & rate_w, float & rate_h, std::vector<int> face_box, cv::Mat image,  std::vector<std::vector<int>> & landmark, bool flag, double & time)
{
    // load image of dlib type
    dlib::cv_image<rgb_pixel> img(image);
    dlib::rectangle rec = dlib::rectangle(0, 0, 224, 224);
    
    // compute detect run time(time unit is "ms")
    timeval start, end;
    gettimeofday(&start, NULL);
    full_object_detection shape = pose_model(img, rec);
    gettimeofday(&end, NULL);
    long cost = (end.tv_sec  - start.tv_sec) * 1000 + (end.tv_usec- start.tv_usec)/1000;
    time = double(cost)/1000;    

    // put the predict 68 face landmark to landmark_temp and map it to the original image, landmark is the final result
    std::vector<std::vector<int>> landmark_temp(68, std::vector<int>(2, 0));
    for (int j = 0; j < 68; ++j) {
        landmark_temp[j][0] = int(shape.part(j).x());
        landmark_temp[j][1] = int(shape.part(j).y());
        //cout << "part:" << j << " : [" << landmark_temp[j][0] << " " << landmark_temp[j][1] << "]" << endl;
    }
    GetNewPoints(rate_w, rate_h, face_box, landmark_temp, landmark);
    landmark_temp.clear();

    // if flag equal to 1, it means using filter processing,
    if (flag){
        cv::Mat landmark_mat(68, 2, CV_32F);
        cv::Mat landmark_mat_trans(2, 68, CV_32F);
        for (int i = 0; i < 68; ++i)
            for (int j = 0; j < 2; ++j)
                landmark_mat.at<float>(i, j) = landmark[i][j];
        // compute the transposition of landmark_mat, because kalman filter need 2x68 matrix rather than 68x2 matrix
        landmark_mat_trans = landmark_mat.t();
        landmark.clear();
        landmark = FaceLandmarkDetector::KalmanFilter(landmark_mat_trans, float(face_box[3] - face_box[1]));
        landmark_mat.release();
        landmark_mat_trans.release();
    }
    return 0;
}

void FaceLandmarkDetector::GetNewPoints(float rate_w, float rate_h, std::vector<int> face_box, std::vector<std::vector<int>> & landmark_pre, std::vector<std::vector<int>> & landmark_pre_ori){
    
    int x = face_box[0];
    int y = face_box[1];
    
    // map 68 face landmarks detected from crop image to original image
    for (int i = 0; i < 68; ++i) {
        landmark_pre_ori[i][0] = int(landmark_pre[i][0] / rate_w + x);
        landmark_pre_ori[i][1] = int(landmark_pre[i][1] / rate_h + y);
    }
}


std::vector<std::vector<int>> FaceLandmarkDetector::KalmanFilter(cv::Mat landmark_pre, float height) {
    
    // init variable
    std::vector<std::vector<int>> landmark_final(68, std::vector<int>(2, 0));
    cv::Mat landmarks_bounding_after_filter(2, 17, CV_32F);
    cv::Mat landmarks_other_part_after_filter(2, 19, CV_32F);
    cv::Mat landmarks_eyes_after_filter(2, 12, CV_32F);
    cv::Mat landmarks_mouth_after_filter(2, 20, CV_32F);
    cv:Mat landmarks_after_filter(2, 68, CV_32F);
    cv::Mat landmark_copy(2, 68, CV_32F);
    landmark_copy = landmark_pre;
    
    // ----------------------------------------------- Bounding ------------------------------------------------
    if (bounding_reset) {
        landmarks_bounding_after_filter = landmark_copy(cv::Range(0, 2), cv::Range(0, 17));
        cv::Mat zero_bounding = Mat::zeros(4, 17, CV_32F);
        cv::Mat statePre_bounding(6, 17, CV_32F);
        cv::vconcat(landmarks_bounding_after_filter, zero_bounding, statePre_bounding);
        kalman_bounding.statePre = statePre_bounding.clone();
        kalman_bounding.measurementMatrix = (Mat_<float>(2, 6) <<
                                             1, 0, 0, 0, 0, 0,
                                             0, 1, 0, 0, 0, 0);
        kalman_bounding.transitionMatrix = (Mat_<float>(6, 6) <<
                                            1, 0, 0.04, 0, 0.0008, 0,
                                            0, 1, 0, 0.04, 0, 0.0008,
                                            0, 0, 1, 0, 0.04, 0,
                                            0, 0, 0, 1, 0, 0.04,
                                            0, 0, 0, 0, 1, 0,
                                            0, 0, 0, 0, 0, 1);
        kalman_bounding.processNoiseCov = (Mat_<float>(6, 6) <<
                                           0.00000064, 0, 0.000032, 0, 0.0008, 0,
                                           0, 0.00000064, 0, 0.000032, 0, 0.0008,
                                           0.000032, 0, 0.0016, 0, 0.04, 0,
                                           0, 0.000032, 0, 0.0016, 0, 0.04,
                                           0.0008, 0, 0.04, 0, 1, 0,
                                           0, 0.0008, 0, 0.04, 0, 1)*3;
        kalman_bounding.measurementNoiseCov = (Mat_<float>(2, 2) <<
                                               10, 0,
                                               0, 10);
        bounding_reset = false;
    }
    else {
        cv::Mat landmarks_cp_bounding = landmark_copy(cv::Range(0, 2), cv::Range(0, 17));
        kalman_bounding.correct(landmarks_cp_bounding);
        cv::Mat landmarks_bounding_after_filter_total = kalman_bounding.predict();
        landmarks_bounding_after_filter = landmarks_bounding_after_filter_total(cv::Range(0, 2), cv::Range(0, 17));
        cv::Mat temp_bounding_mat = (landmarks_cp_bounding - landmarks_bounding_after_filter).mul(landmarks_cp_bounding - landmarks_bounding_after_filter);
        cv::Mat dis_bounding_temp(1, 17, CV_32F);
        
        for (int i = 0; i < temp_bounding_mat.cols; i++)
            dis_bounding_temp.at<float>(i) = sqrt(temp_bounding_mat.at<float>(0, i) + temp_bounding_mat.at<float>(1, i));
        cv::Scalar dis_mean_bounding = cv::mean(dis_bounding_temp);
        // 归一化处理
        float error_bounding = dis_mean_bounding[0] / height;
        if (error_bounding > 0.02) {
            landmarks_bounding_after_filter = landmarks_cp_bounding;
            bounding_reset = true;
        }
    }
    
    // ----------------------------------------------- eyes ------------------------------------------------
    if (eyes_reset) {
        landmarks_eyes_after_filter = landmark_copy(cv::Range(0, 2), cv::Range(36, 48));
        cv::Mat zero_eyes = Mat::zeros(4, 12, CV_32F);
        cv::Mat statePre_eyes(6, 12, CV_32F);
        cv::vconcat(landmarks_eyes_after_filter, zero_eyes, statePre_eyes);
        
        kalman_eyes.statePre = statePre_eyes;
        kalman_eyes.measurementMatrix = (Mat_<float>(2, 6) << 1, 0, 0, 0, 0, 0,
                                         0, 1, 0, 0, 0, 0);
        kalman_eyes.transitionMatrix = (Mat_<float>(6, 6) << 1, 0, 0.04, 0, 0.0008, 0,
                                        0, 1, 0, 0.04, 0, 0.0008,
                                        0, 0, 1, 0, 0.04, 0,
                                        0, 0, 0, 1, 0, 0.04,
                                        0, 0, 0, 0, 1, 0,
                                        0, 0, 0, 0, 0, 1);
        kalman_eyes.processNoiseCov = (Mat_<float>(6, 6) << 0.00000064, 0, 0.000032, 0, 0.0008, 0,
                                       0, 0.00000064, 0, 0.000032, 0, 0.0008,
                                       0.000032, 0, 0.0016, 0, 0.04, 0,
                                       0, 0.000032, 0, 0.0016, 0, 0.04,
                                       0.0008, 0, 0.04, 0, 1, 0,
                                       0, 0.0008, 0, 0.04, 0, 1) * 3;
        kalman_eyes.measurementNoiseCov = (Mat_<float>(2, 2) <<
                                           4, 0,
                                           0, 4);
        eyes_reset = false;
    }
    else {
        cv::Mat landmarks_cp_eyes = landmark_copy(cv::Range(0, 2), cv::Range(36, 48));
        kalman_eyes.correct(landmarks_cp_eyes);
        
        cv::Mat landmarks_eyes_after_filter_total = kalman_eyes.predict();
        landmarks_eyes_after_filter = landmarks_eyes_after_filter_total(cv::Range(0, 2), cv::Range(0, 12));
        
        cv::Mat temp_eyes_mat = (landmarks_cp_eyes - landmarks_eyes_after_filter).mul(landmarks_cp_eyes - landmarks_eyes_after_filter);
        Mat dis_eyes_temp(1, 12, CV_32F);
        for (int i = 0; i < temp_eyes_mat.cols; i++)
            dis_eyes_temp.at<float>(0, i) = sqrt(temp_eyes_mat.at<float>(0, i) + temp_eyes_mat.at<float>(1, i));
        cv::Scalar dis_eye_mean = cv::mean(dis_eyes_temp);
        float error_eyes = dis_eye_mean[0] / height;
        if (error_eyes> 0.01) {
            landmarks_eyes_after_filter = landmarks_cp_eyes;
            eyes_reset = true;
        }
    }
    
    // ----------------------------------------------- Mouth ------------------------------------------------
    if (mouth_reset) {
        landmarks_mouth_after_filter = landmark_copy(cv::Range(0, 2), cv::Range(48, 68));
        cv::Mat zero_mouth = Mat::zeros(4, 20, CV_32F);
        cv::Mat statePre_mouth(6, 20, CV_32F);
        cv::vconcat(landmarks_mouth_after_filter, zero_mouth, statePre_mouth);
        
        kalman_mouth.statePre = statePre_mouth;
        kalman_mouth.measurementMatrix = (Mat_<float>(2, 6) <<
                                          1, 0, 0, 0, 0, 0,
                                          0, 1, 0, 0, 0, 0);
        kalman_mouth.transitionMatrix = (Mat_<float>(6, 6) <<
                                         1, 0, 0.04, 0, 0.0008, 0,
                                         0, 1, 0, 0.04, 0, 0.0008,
                                         0, 0, 1, 0, 0.04, 0,
                                         0, 0, 0, 1, 0, 0.04,
                                         0, 0, 0, 0, 1, 0,
                                         0, 0, 0, 0, 0, 1);
        kalman_mouth.processNoiseCov = (Mat_<float>(6, 6) <<
                                        0.00000064, 0, 0.000032, 0, 0.0008, 0,
                                        0, 0.00000064, 0, 0.000032, 0, 0.0008,
                                        0.000032, 0, 0.0016, 0, 0.04, 0,
                                        0, 0.000032, 0, 0.0016, 0, 0.04,
                                        0.0008, 0, 0.04, 0, 1, 0,
                                        0, 0.0008, 0, 0.04, 0, 1)*3;
        kalman_mouth.measurementNoiseCov = (Mat_<float>(2, 2) <<
                                            4, 0,
                                            0, 4);
        mouth_reset = false;
    }
    else {
        cv::Mat landmarks_cp_mouth = landmark_copy(cv::Range(0, 2), cv::Range(48, 68));
        kalman_mouth.correct(landmarks_cp_mouth);
        cv::Mat landmarks_mouth_after_filter_total = kalman_mouth.predict();
        landmarks_mouth_after_filter = landmarks_mouth_after_filter_total(cv::Range(0, 2), cv::Range(0, 20));
        cv::Mat temp_mat_mouth = (landmarks_cp_mouth - landmarks_mouth_after_filter).mul(landmarks_cp_mouth - landmarks_mouth_after_filter);
        cv::Mat dis_temp_mouth(1, 20, CV_32F);
        
        for (int i = 0; i < temp_mat_mouth.cols; i++)
            dis_temp_mouth.at<float>(i) = sqrt(temp_mat_mouth.at<float>(0, i) + temp_mat_mouth.at<float>(1, i));
        cv::Scalar dis_mean_mouth = cv::mean(dis_temp_mouth);
        float error_mouth = dis_mean_mouth[0] / height;
        if (error_mouth > 0.01) {
            landmarks_mouth_after_filter = landmarks_cp_mouth;
            mouth_reset = true;
        }
    }
    
    // ----------------------------------------------- other part ------------------------------------------------
    if (other_part_reset) {
        landmarks_other_part_after_filter = landmark_copy(cv::Range(0, 2), cv::Range(17, 36));
        cv::Mat zero_other_part = Mat::zeros(4, 19, CV_32F);
        cv::Mat statePre_other_part(6, 19, CV_32F);
        cv::vconcat(landmarks_other_part_after_filter, zero_other_part, statePre_other_part);
        
        kalman_other_part.statePre = statePre_other_part;
        kalman_other_part.measurementMatrix = (Mat_<float>(2, 6) <<
                                               1, 0, 0, 0, 0, 0,
                                               0, 1, 0, 0, 0, 0);
        kalman_other_part.transitionMatrix = (Mat_<float>(6, 6) <<
                                              1, 0, 0.04, 0, 0.0008, 0,
                                              0, 1, 0, 0.04, 0, 0.0008,
                                              0, 0, 1, 0, 0.04, 0,
                                              0, 0, 0, 1, 0, 0.04,
                                              0, 0, 0, 0, 1, 0,
                                              0, 0, 0, 0, 0, 1);
        kalman_other_part.processNoiseCov = (Mat_<float>(6, 6) <<
                                             0.00000064, 0, 0.000032, 0, 0.0008, 0,
                                             0, 0.00000064, 0, 0.000032, 0, 0.0008,
                                             0.000032, 0, 0.0016, 0, 0.04, 0,
                                             0, 0.000032, 0, 0.0016, 0, 0.04,
                                             0.0008, 0, 0.04, 0, 1, 0,
                                             0, 0.0008, 0, 0.04, 0, 1)*3;
        kalman_other_part.measurementNoiseCov = (Mat_<float>(2, 2) <<
                                                 4, 0,
                                                 0, 4);
        other_part_reset = false;
    }
    else {
        cv::Mat landmarks_cp_other_part = landmark_copy(cv::Range(0, 2), cv::Range(17, 36));
        kalman_other_part.correct(landmarks_cp_other_part);
        cv::Mat landmarks_other_part_after_filter_total = kalman_other_part.predict();
        landmarks_other_part_after_filter = landmarks_other_part_after_filter_total(cv::Range(0, 2), cv::Range(0, 19));
        cv::Mat temp_mat_other_part = (landmarks_cp_other_part - landmarks_other_part_after_filter).mul(landmarks_cp_other_part - landmarks_other_part_after_filter);
        cv::Mat dis_temp_other_part(1, 19, CV_32F);
        
        for (int i = 0; i < temp_mat_other_part.cols; i++)
            dis_temp_other_part.at<float>(i) = sqrt(temp_mat_other_part.at<float>(0, i) + temp_mat_other_part.at<float>(1, i));
        cv::Scalar dis_mean_other_part = cv::mean(dis_temp_other_part);
        float error_other_part = dis_mean_other_part[0] / height;
        if (error_other_part > 0.02) {
            landmarks_other_part_after_filter = landmarks_cp_other_part;
            other_part_reset = true;
        }
    }
    cv::Mat landmarks_after_filter_temp1(2, 36, CV_32F);
    cv::hconcat(landmarks_bounding_after_filter, landmarks_other_part_after_filter, landmarks_after_filter_temp1);
    cv::Mat landmarks_after_filter_temp2(2, 48, CV_32F);
    cv::hconcat(landmarks_after_filter_temp1, landmarks_eyes_after_filter, landmarks_after_filter_temp2);
    cv::hconcat(landmarks_after_filter_temp2, landmarks_mouth_after_filter, landmarks_after_filter);
    
    if (count == 0) {
        landmarks_after_filter = landmark_copy;
        count++;
    }
    
    // transform 68x2 matrix to 2x68 two dimension vector
    cv::Mat landmarks_after_filter_temp = landmarks_after_filter.t();
    for (int i = 0; i < 68; ++i)
        for (int j = 0; j < 2; ++j)
            landmark_final[i][j] = landmarks_after_filter_temp.at<float>(i, j);
    
    landmarks_bounding_after_filter.release();
    landmarks_other_part_after_filter.release();
    landmarks_eyes_after_filter.release();
    landmarks_mouth_after_filter.release();
    landmarks_after_filter.release();
    landmark_copy.release();

    return landmark_final;
}

void FaceLandmarkDetector::drawLandmarks(cv::Mat &image, std::vector<std::vector<int>> landmark, cv::Scalar color, int radius) {
    for (int i = 0; i < 68; i++) {
        cv::Point p = cv::Point(landmark[i][0], landmark[i][1]);
        cv::circle(image, p, radius, color, -1);
    }
}

// test vedio
void FaceLandmarkDetector::TestVedioLandmark(FacenetCaffe fc_box, const string root){
    
    //init test variable
    double time, time_detec, mser, mser_norm;
    double time_detec_sum = 0;
    double time_sum = 0;
    double mser_sum = 0;
    double mser_norm_sum = 0;
    int id_sum = 0;
    
    // init default path
    string vedio_anno_dir = root + "annot/";
    string vedio_dir = root + "vid.avi";
    string img_out_path = root + "result/";
    
    mkdir(img_out_path.c_str(), S_IRWXU);
    
    // read all pts file in vedio_anno_dir
    std::vector<string> file_list;
    struct dirent* ptr;
    DIR* file_dir = opendir(vedio_anno_dir.c_str());
    while((ptr = readdir(file_dir)) != NULL)
    {
        string filename = ptr->d_name;
        //get file suffix
        string suffixStr = filename.substr(filename.find_last_of('.') + 1);
        if (strcmp(suffixStr.c_str(), "pts") == 0) {
            file_list.push_back(filename);
        }
    }
    // sort all of file according to file name
    sort(file_list.begin(), file_list.begin() + int(file_list.size()));
    closedir(file_dir);
    
    // open vedio
    cv::VideoCapture capture;
    capture.open(vedio_dir);
    if (!capture.isOpened()) {
        cout << "vedio is empty!" << endl;
        return;
    }
    
    // test every frame according to pts file
    for (auto iter = file_list.begin(); iter != file_list.end(); ++iter) {
        
        id_sum++;
        
        cv::Mat frame;
        capture >> frame;
        float rate_w = 1;
        float rate_h = 1;
        cv::Mat img;
        double NormDist = 1;
        
        string pts_path = vedio_anno_dir + *iter;
        std::vector<std::vector<int>> face_box;
        std::vector<cv::Mat> img_resize_vector;
        std::vector<std::vector<int>> landmark_gt(68, std::vector<int>(2, 0));
        std::vector<std::vector<int>> landmark_pre(68, std::vector<int>(2, 0));
        std::vector<std::vector<int>> landmark_filter(68, std::vector<int>(2, 0));
        string img_out_path_name = img_out_path + std::to_string(id_sum) + ".jpg";        

        // get face_box from every frame
        GetFaceBox(fc_box, frame, face_box, 0.7, time_detec);
        // adjust face_box
        AdjustFaceBox(frame, face_box);
        // crop frame according the adjust face_box
        ImageStandard(rate_w, rate_h, frame, face_box, img_resize_vector);
        
        if (face_box.size() <= 0) {
            cout << "detect none!!!" << endl;
        }
        
        // detect 68 face landmark of every face_box
        for (int i = 0; i < face_box.size(); ++i) {
            detectFaceLandmark(rate_w, rate_h, face_box[i], img_resize_vector[i], landmark_pre, true, time);
            
            // draw face_box
            cv::Point pt1,pt2;
            pt1.x = face_box[i][0];
            pt1.y = face_box[i][1];
            pt2.x = face_box[i][2];
            pt2.y = face_box[i][3];
            cout << "face number " << i << " : [" << pt1.x << ", " << pt1.y << ", " << pt2.x << ", " << pt2.y << "]" << endl;
            cv::rectangle(frame, pt1, pt2, cvScalar(0, 255, 0), 2, 8, 0);

            // draw landmark
            drawLandmarks(frame, landmark_pre, cv::Scalar(0, 255, 0), 2); 
            
            // test index according to the ground truth of pts file
            landmark_gt = ReadPoints(pts_path);
            // compute normalized param
            NormDist = ComputeNormDist(landmark_gt);
            
            // compute mean square error ,normalized mean square error, detect face box time and detect landmark time
            mser = ComputeAccuracy(landmark_gt, landmark_pre);
            mser_norm = mser / NormDist;    // normalized mean square error
            
            mser_sum += mser;
            mser_norm_sum += mser_norm;
            time_sum += time;
            time_detec_sum += time_detec;
            
            cout << "Frame number: " << id_sum << endl;
            cout << "Detect time consuming: " << time_detec << endl;
            cout << "Landmark time consuming: " << time << endl;
            cout << "NormDist: " << NormDist << endl;
            cout << "Mean Square Error: " << mser << endl;
            cout << "Normalized Mean Square Error: " << mser_norm << endl;
            cout << endl;
        }
        cv::imwrite(img_out_path_name, frame);
        
        face_box.clear();
        img_resize_vector.clear();
        landmark_gt.clear();
        landmark_pre.clear();
        landmark_filter.clear();

    }
    cout << "Average Time consuming: " << time_sum / id_sum << endl;
    cout << "Average Mean Square Error: " << mser_sum / id_sum << endl;
    cout << "Average Normalized Mean Square Error: " << mser_norm_sum / id_sum << endl;
}

// test image
void FaceLandmarkDetector::TestImageLandmark(FacenetCaffe fc_box, const string img_path, const string pts_path, const string output_path) {
    
    // init
    double time, time_detec, mser, mser_norm;
    
    cv::Mat img = cv::imread(img_path);
    float rate_w = 1;
    float rate_h = 1;
    double NormDist = 1;

    std::vector<std::vector<int>> face_box;
    std::vector<cv::Mat> img_resize_vector;
    std::vector<std::vector<int>> landmark_gt(68, std::vector<int>(2, 0));
    std::vector<std::vector<int>> landmark_pre(68, std::vector<int>(2, 0));
    std::vector<std::vector<int>> landmark_filter(68, std::vector<int>(2, 0));
    
    // same as above
    GetFaceBox(fc_box, img, face_box, 0.7, time_detec);
    AdjustFaceBox(img, face_box);
    ImageStandard(rate_w, rate_h, img, face_box, img_resize_vector);
    
    if (face_box.size() <= 0) {
        cout << "detect none!!!" << endl;
        return;
    }
    for (int i = 0; i < face_box.size(); ++i) {
        detectFaceLandmark(rate_w, rate_h, face_box[i], img_resize_vector[i], landmark_pre, false, time);
        
        // draw face_box
        cv::Point pt1,pt2;
        pt1.x = face_box[i][0];
        pt1.y = face_box[i][1];
        pt2.x = face_box[i][2];
        pt2.y = face_box[i][3];
        cout << "face number " << i << " : [" << pt1.x << ", " << pt1.y << ", " << pt2.x << ", " << pt2.y << "]" << endl;
        cv::rectangle(img, pt1, pt2, cvScalar(0, 255, 0), 2, 8, 0);
        
        // draw landmark
        drawLandmarks(img, landmark_pre, cv::Scalar(0, 255, 0), 2);

        landmark_gt = ReadPoints(pts_path);
        NormDist = ComputeNormDist(landmark_gt);
        
        mser = ComputeAccuracy(landmark_gt, landmark_pre);
        mser_norm = mser / NormDist;
        
        cout << "Detect time consuming: " << time_detec << endl;
        cout << "Landmark time consuming: " << time << endl;
        cout << "NormDist: " << NormDist << endl;
        cout << "Mean Square Error: " << mser << endl;
        cout << "Normalized Mean Square Error: " << mser_norm << endl;
        cout << endl;
    }
    cv::imwrite(output_path, img);
    
    face_box.clear();
    img_resize_vector.clear();
    landmark_gt.clear();
    landmark_pre.clear();
    landmark_filter.clear();
}

// ----------------------------------------------test code--------------------------------------
const std::vector<int> FaceLandmarkDetector::split(const string& s, const char& c)
{
    string buff{ "" };
    std::vector<int> v;
    for (auto n : s)
    {
        if (n != c) buff += n;
        else
            if (n == c && buff != "") {
                v.push_back(stoi(buff)); buff = "";
            }
    }
    if (buff != "") v.push_back(stoi(buff));
    return v;
}

std::vector<std::vector<int>> FaceLandmarkDetector::ReadPoints(string filename){
    std::vector<std::vector<int>> points;
    ifstream file;
    char buf[1024];
    const char * message;
    file.open(filename.c_str());
    if (file.is_open()) {
        while(file.good() && !file.eof())
        {
            memset(buf,0,1024);
            file.getline(buf,1204);
            message = buf;
            if (strcmp(message, "{") != 0 && strcmp(message, "}") != 0 && strcmp(message, "version: 1") != 0 && strcmp(message, "n_points: 68") != 0 && strcmp(message, "{\r") != 0 && strcmp(message, "}\r") != 0 && strcmp(message, "version: 1\r") != 0 && strcmp(message, "n_points: 68\r") != 0) {
                std::vector<int> v{ split(message, ' ') };
                //                for (auto n : v) cout << n <<endl;
                points.push_back(v);
            }
        }
        file.close();
    }
    else
        cout << "file is empty!!!" << endl;
    return points;
}

double FaceLandmarkDetector::ComputeNormDist(std::vector<std::vector<int>> landmark){
    double NormDist = 1;
    // turn vector into Mat
    cv::Mat landmark_mat(68, 2, CV_32FC1);
    for (int i = 0; i < 68; i++) {
        for (int j = 0; j < 2; j++) {
            landmark_mat.at<float>(i, j) = landmark[i][j];
        }
    }
    //    cout << landmark_mat << endl;
    // compute mean value
    cv::Mat landmark_mat_x = landmark_mat.col(0);
    cv::Mat landmark_mat_y = landmark_mat.col(1);
    cv::Scalar mean1_x_temp = cv::mean(landmark_mat_x(cv::Range(36, 42), cv::Range(0, 1)));
    double mean1_x = mean1_x_temp[0];
    cv::Scalar mean1_y_temp = cv::mean(landmark_mat_y(cv::Range(36, 42), cv::Range(0, 1)));
    double mean1_y = mean1_y_temp[0];
    cv::Scalar mean2_x_temp = cv::mean(landmark_mat_x(cv::Range(42, 48), cv::Range(0, 1)));
    double mean2_x = mean2_x_temp[0];
    cv::Scalar mean2_y_temp = cv::mean(landmark_mat_y(cv::Range(42, 48), cv::Range(0, 1)));
    double mean2_y = mean2_y_temp[0];
    
    // compute mean error
    NormDist = sqrt(pow((mean2_x - mean1_x), 2) + pow((mean2_y - mean1_y), 2));
    
    //    landmark_mat.release();
    return NormDist;
}

string FaceLandmarkDetector::pts_to_img_name(const string root, string file, string pts_file_name){
    string img_file_name;
    string prefixStr = pts_file_name.substr(0, pts_file_name.find_last_of('.'));
    string img_file_name_jpg = root + file + "/" + prefixStr + ".jpg";
    string img_file_name_jpg_temp = prefixStr + ".jpg";
    string img_file_name_png = root + file + "/" + prefixStr + ".png";
    string img_file_name_png_temp = prefixStr + ".png";
    string img_file_name_jpeg = root + file + "/" + prefixStr + ".jpeg";
    string img_file_name_jpeg_temp = prefixStr + ".jpeg";
    
    ifstream fin_jpg(img_file_name_jpg);
    if (fin_jpg) {
        img_file_name = img_file_name_jpg_temp;
    }
    ifstream fin_png(img_file_name_png);
    if (fin_png) {
        img_file_name = img_file_name_png_temp;
    }
    ifstream fin_jpeg(img_file_name_jpeg);
    if (fin_jpeg) {
        img_file_name = img_file_name_jpeg_temp;
    }
    return img_file_name;
}

double FaceLandmarkDetector::ComputeAccuracy(std::vector<std::vector<int>> landmark_gt, std::vector<std::vector<int>> landmark_pre) {
    double landmark_sum = 0;
    double mser;
    for (int i = 0; i < 68; i++) {
        landmark_sum += (sqrt(pow(double(landmark_gt[i][0]) - double(landmark_pre[i][0]), 2) + pow(double(landmark_gt[i][1]) - double(landmark_pre[i][1]), 2)));
    }
    mser = landmark_sum / 68;
    return mser;
}

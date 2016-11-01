#include "convnets.h"

QVector<cv::Mat> convGenerator(int size, int number){
    QVector<cv::Mat> convnetList;
    cv::Mat convnet;
    double r;

    qsrand(1);
    for (int i = 0; i < number; i++){
        convnet(size, size, CV_64FC4, cv::Scalar(0, 0, 0, 0));

        for (int k = 0; k < size; k++){
            for (int l = 0; l < size; l++){
                for (int c = 0; c < 4; c++){
                    r = (double)qrand()/RAND_MAX;
                    convnet.at<cv::Vec4d>(k, l)[c] = r;
                }
            }
        }

        convnetList.push_back(convnet);
    }

    return convnetList;
}

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

QVector<cv::Mat> zeropadding1(cv::Mat rgbd, int padding){
    cv::copyMakeBorder( rgbd, rgbd, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0) );
}

QVector<cv::Mat> zeropadding2(cv::Mat rgbd, int padding){
    cv::copyMakeBorder( rgbd, rgbd, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0,0,0,0,0) );
}

cv::Mat relu(cv::Mat res){
    QVector<cv::Mat> splitLayer;
    cv::split(res, splitLayer);
    for(int i = 0; i < splitLayer.length(); i++){
        for(int x = 0; x < splitLayer.at(i).rows; x++){
            for(int y = 0; y < splitLayer.at(i).cols; y++){
                if (splitLayer.at(i).at<cv::Vec4d>(x,y)[0] < 0){
                    splitLayer.at(i).at<cv::Vec4d>(x,y)[0] = 0;
                }
            }
        }
    }
    cv::merge(splitLayer, res);
    return res;
}

#include "convnets.h"

QVector<cv::Mat> convGenerator(int size, int number, int epoch){
    QVector<cv::Mat> convnetList;
    cv::Mat convnet;
    double r;

    qsrand(1); //choose seed
    for (int i = 0; i < number; i++){
        convnet = cv::Mat(size, size, CV_64FC(4*epoch), cv::Scalar(0, 0, 0, 0));

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
    cv::copyMakeBorder( rgbd, rgbd, padding, padding, padding, padding, cv::BORDER_CONSTANT );
}

QVector<cv::Mat> zeropadding2(cv::Mat rgbd, int padding){
    cv::copyMakeBorder( rgbd, rgbd, padding, padding, padding, padding, cv::BORDER_CONSTANT );
}

cv::Mat relu(cv::Mat res){
    std::vector<cv::Mat> splitLayer;
    cv::split(res, splitLayer);
    for(int i = 0; i < splitLayer.size(); i++){
        for(int x = 0; x < splitLayer[i].rows; x++){
            for(int y = 0; y < splitLayer[i].cols; y++){
                if (splitLayer[i].at<cv::Vec4d>(x,y)[0] < 0){
                    splitLayer[i].at<cv::Vec4d>(x,y)[0] = 0;
                }
            }
        }
    }
    cv::merge(splitLayer, res);
    return res;
}

cv::Mat convolution(cv::Mat rgbd, QVector<cv::Mat> kernel, int filterCount){
    double delta = 0;
    std::vector<cv::Mat> desList(kernel.length());
    cv::Mat des;
    cv::Mat output;
    for(int i = 0; i < kernel.length(); i++){
        cv::filter2D(rgbd, des, filterCount, kernel.at(i), cv::Point(-1, -1), delta, cv::BORDER_CONSTANT);
        desList[i] = des;
    }
    cv::merge(desList, output);
    return output;
}

cv::Mat pooling(cv::Mat res, int size, int epoch){
    cv::Mat des = res;
    pyrDown( res, des, cv::Size( res.cols/(3+epoch), res.rows/(3+epoch) ));
    return des;
}

cv::Mat reshape(QVector<cv::Mat> res){
    cv::Mat output(150, 400, CV_64FC1, cv::Scalar(0));
    cv::Mat currRes;
    std::vector<cv::Mat> splitLayer;
    for(int i = 0; i < res.length(); i++){
        currRes = res.at(i);
        cv::split(currRes, splitLayer);
        for(int j = 0; j < splitLayer.size(); j++){
            for(int x = 0; x < 5; x++){
                for(int y = 0; y < 5; y++){
                    output.at<int>(i, j*25 + x*5 + y);
                }
            }
        }
    }
    return output;
}

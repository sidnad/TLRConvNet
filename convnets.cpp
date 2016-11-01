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

cv::Mat zeropadding(cv::Mat rgbd, int padding){
    cv::copyMakeBorder( rgbd, rgbd, padding, padding, padding, padding, cv::BORDER_CONSTANT );
    std::cout << "dimension after zero padding: " << rgbd.rows << std::endl;
    return rgbd;
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

cv::Mat convolution(cv::Mat rgbd, QVector<cv::Mat> kernel, int filterCount, int epoch){

    std::cout << "dimension of the input: " << rgbd.rows << std::endl;
    std::cout << "dimension of each kernel: " << kernel.at(0).rows << std::endl;

    double delta = 0;
    std::vector<cv::Mat> resList;
    int size;
    if (epoch == 1){
        size = 100;
    }
    else if (epoch == 2){
        size = 25;
    }
    cv::Mat res(size, size, CV_64F);
    cv::Mat output;
    double tempval;
    for(int i = 0; i < kernel.length(); i++){ // 8 or 16 loops
        for(int x = 0; x < rgbd.rows - kernel.at(i).cols + 1; x++){
            for(int y = 0; y < rgbd.cols - kernel.at(i).cols + 1; y++){
                tempval = 0;
                for(int layer = 0; layer < 4; layer++){
                    for(int u = 0; u < kernel.at(i).rows; u++){
                        for(int v = 0; v < kernel.at(i).cols; v++){
                            tempval = tempval + kernel.at(i).at<cv::Vec4d>(u,v)[layer] * rgbd.at<cv::Vec4d>(x+u, y+v)[layer];
                        }
                    }
                }
                tempval = tempval/(kernel.at(i).rows*kernel.at(i).cols)/4; // val/25
                res.at<double>(x, y) = tempval;
            }
        }
        /*
        cv::namedWindow("PuRSE");
        cv::convertScaleAbs(res,res,1,0);
        cv::imshow("PuRSE", res);
        cv::waitKey(1000);*/

        resList.push_back(res);
    }

    /*
    cv::namedWindow("PuRSE");
    cv::convertScaleAbs(res,res,1,0);
    cv::imshow("PuRSE", res);
    cv::waitKey(1000000000000);
    */

    cv::merge(resList, output);
    return output;
}

cv::Mat pooling(cv::Mat res, int size, int epoch){
    cv::Mat des = res;
    std::cout << "epoch: " << epoch << " old dimension before pooling: " << res.rows << std::endl;
    pyrDown( res, des, cv::Size( res.cols/(2), res.rows/(2) ));
    pyrDown( des, des, cv::Size( des.cols/(2), des.rows/(2) ));
    std::cout << "epoch: " << epoch << " new dimension after pooling: " << des.rows << std::endl;
    return des;
}

cv::Mat reshape(QVector<cv::Mat> res){
    cv::Mat output(51, 576, CV_64FC1, cv::Scalar(0));
    cv::Mat currRes;
    std::vector<cv::Mat> splitLayer;
    for(int i = 0; i < res.length(); i++){
        currRes = res.at(i);
        cv::split(currRes, splitLayer);
        for(int j = 0; j < splitLayer.size(); j++){
            for(int x = 0; x < 6; x++){
                for(int y = 0; y < 6; y++){
                    output.at<int>(i, j*25 + x*5 + y);
                }
            }
        }
    }
    return output;
}

#include "mainwindow.h"
#include "filereader.h"
#include "convnets.h"
#include <QApplication>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QVector<QVector<QString>> data = freader();
    std::vector<cv::Mat> rgbdList = farranger_rgbd(data, 0.07, 0); // former 7% for training.
    std::vector<int> groundTruth = groundTruthVector(data, 0.07);
    std::vector<std::vector<double>> probabilityMat = probMatInit(groundTruth);
    QVector<int> rgbd_gt = farranger_gt(data, 0.07);//ground truth, 1 for TL, 0 for other.

    std::vector<cv::Mat> conv1 = convGenerator(7, 8, 1); // 7x7, 8 filters. The inside QVector is RGBD 4 layer, second QVector is # of filters.
    std::vector<cv::Mat> conv2 = convGenerator(5, 16, 2); // 5x5, 16 filters, 8 layers each.
    cv::Mat res1, res2;
    QVector<cv::Mat> res2list;
    cv::Mat rgbd(100, 100, CV_64FC4, cv::Scalar(0, 0, 0, 0));
    int learningrate = 3;
    double correctPercentage = 0;

    for (int i = 0; i < rgbdList.size(); i++){
        std::cout << "iteration: " << i << "  -------------epoch 1-------------" << std::endl;
        rgbd = rgbdList.at(i);
        //ZERO PADDING 1
        rgbd = zeropadding(rgbd, 3);
        //CONV1
        res1 = convolution(rgbd, conv1, 8, 1);
        //RELU LAYER
        res1 = relu(res1, 1);
        showImage(res1, "");
        //POOLING
        res1 = pooling(res1, 4, 1);
        std::cout << "iteration: " << i << "  -------------epoch 2-------------" << std::endl;
        //ZERO PADDING 2
        res1 = zeropadding(res1, 2);
        //CONV2
        res2 = convolution(res1, conv2, 16, 2);

        showImage(res2, "../TLRConvNet/images/");

        //RELU LAYER
        res2 = relu(res2, 2);
        //POOLING
        res2 = pooling(res2, 4, 2);

        //showImage(res2, "../TLRConvNet/images_pooling2/");

        res2list.push_back(res2);
    }

    //RESHAPE
    cv::Mat res3;
    res3 = reshape(res2list); //522 * (6*6*16) = 150*576
    double val;
    int val2;
    cv::Mat outputfinal(51,576,CV_8UC1);

    //cv::convertScaleAbs(res3,res3,1,0);
    for(int i = 0; i < 51; i++){
        for(int j = 0; j < 576; j++){
            val = res3.at<double>(i,j);
            outputfinal.at<uchar>(i,j) = (int)(val*10);
            val2 = (int)outputfinal.at<uchar>(i,j);
            //if(val != 0)
                //std::cout << val2 << std::endl;
        }
    }

    //cv::Mat testMat = cv::Mat( outputfinal ).reshape( 0, 256 );
    //testMat.convertTo( testMat, CV_8UC1 );

    cv::namedWindow("PuRSE");
    cv::imshow("PuRSE", outputfinal);
    cv::waitKey(10);

    int iteration = 0;
    while(iteration < 50){
    //while(correctPercentage!=0.85 || correctPercentage>0.85){
        //CALCULATION
        std::vector<std::vector<double>> errorList = calculation(res3, probabilityMat, groundTruth);
        double correctPercentage = correctPercentageCalculation(errorList, groundTruth);
        std::cout << "Correct Precentage: " << correctPercentage << std::endl;

        std::vector<std::vector<double>> delta = calculateDelta(res3,errorList,learningrate);
        probabilityMat = trainFCL(probabilityMat,delta);
        /*for(int i=0;i<probabilityMat.at(0).size();i++){
            std::cout<<probabilityMat.at(0).at(i)<<" ";
            std::cout<<probabilityMat.at(1).at(i)<<" ";
        }*/
        iteration++;
    }

    //TESTING
    std::vector<cv::Mat> rgbdList2 = farranger_rgbd(data, 0.03, 50); // 7% for testing.
    std::vector<int> groundTruth2 = groundTruthVector(data, 0.03);
    QVector<cv::Mat> res2list2;
    for (int i = 0; i < rgbdList2.size(); i++){
        std::cout << "iteration: " << i << "  -------------epoch 1-------------" << std::endl;
        rgbd = rgbdList2.at(i);
        //ZERO PADDING 1
        rgbd = zeropadding(rgbd, 3);
        //CONV1
        res1 = convolution(rgbd, conv1, 8, 1);
        //RELU LAYER
        res1 = relu(res1, 1);
        showImage(res1, "");
        //POOLING
        res1 = pooling(res1, 4, 1);
        std::cout << "iteration: " << i << "  -------------epoch 2-------------" << std::endl;
        //ZERO PADDING 2
        res1 = zeropadding(res1, 2);
        //CONV2
        res2 = convolution(res1, conv2, 16, 2);

        showImage(res2, "../TLRConvNet/images/");

        //RELU LAYER
        res2 = relu(res2, 2);
        //POOLING
        res2 = pooling(res2, 4, 2);

        //showImage(res2, "../TLRConvNet/images_pooling2/");

        res2list2.push_back(res2);
    }

    //RESHAPE
    cv::Mat res4;
    res4 = reshape_testing(res2list2); //522 * (6*6*16) = 150*576
    //cv::convertScaleAbs(res3,res3,1,0);

    //CALCULATION
    std::vector<std::vector<double>> errorList2 = calculation(res4, probabilityMat, groundTruth2);
    correctPercentage = correctPercentageCalculation(errorList2, groundTruth2);
    std::cout << "Final Estimate Precentage: " << correctPercentage << std::endl;


    return a.exec();
}



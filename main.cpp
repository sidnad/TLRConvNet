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

    QVector<cv::Mat> rgbdList = farranger_rgbd(data, 0.7); // former 70% for training.
    QVector<int> rgbd_gt = farranger_gt(data, 0.7);//ground truth, 1 for TL, 0 for other.

    QVector<cv::Mat> conv1 = convGenerator(7, 8, 1); // 7x7, 8 filters. The inside QVector is RGBD 4 layer, second QVector is # of filters.
    QVector<cv::Mat> conv2 = convGenerator(5, 16, 2); // 5x5, 16 filters, 8 layers each.
    cv::Mat res1, res2;
    QVector<cv::Mat> res2list;
    cv::Mat rgbd(100, 100, CV_64FC4, cv::Scalar(0, 0, 0, 0));

    for (int i = 0; i < rgbdList.length(); i++){
        rgbd = rgbdList.at(i);
        //ZERO PADDING 1
        //rgbd = zeropadding1(rgbd, 2);
        //CONV1
        res1 = convolution(rgbd, conv1, 8);
        //RELU LAYER
        res1 = relu(res1);
        //POOLING
        res1 = pooling(res1, 4, 1);
        //ZERO PADDING 2
        //rgbd = zeropadding2(rgbd, 1);
        //CONV2
        res2 = convolution(res1, conv1, 16);
        //RELU LAYER
        res2 = relu(res2);
        //POOLING
        res2 = pooling(res2, 4, 2);
        res2list.push_back(res2);
    }
    //RESHAPE
    cv::Mat res3;
    res3 = reshape(res2list); //150 * (5*5*16) = 150*400
    double val;
    for(int i = 0; i < 150; i++){
        for(int j = 0; j < 400; j++){
            val = res3.at<double>(i,j);
            //if(val != 0)
                std::cout << val << std::endl;
        }
    }

    cv::namedWindow("PuRSE");
    cv::imshow("PuRSE", res3);
    cv::waitKey(10);

    /*
    //CLASSIFICATION
    QVector<QVector<double>> res4; // [0.89 0.54], ...
    cv::Mat fullyConnLayerWeight = (784, 2, CV_8UC1, cv::Scalar(0));

    res4 = matrixMult(res3, fullyConnLayerWeight);

    //COST
    QVector<double> cost;
    cost = calculateCost(res4, rgbd_gt);
    double totalCost;
    totalCost = sumUp(cost);

    //BP
    */


    return a.exec();
}

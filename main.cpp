#include "mainwindow.h"
#include "filereader.h"
#include <QApplication>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QVector<QVector<QString>> data = freader();

    QVector<QVector<QString>> rgbdList = farranger_rgbd(data, 0.7); // former 70% for training.
    QVector<int> rgbd_gt = farranger_gt(data, 0.7);//ground truth, 1 for TL, 0 for other.

    QVector<QVector<cv::Mat>> conv1 = convGenerator(7, 8); // 7x7, 8 filters. The inside QVector is RGBD 4 layer, second QVector is # of filters.
    QVector<QVector<cv::Mat>> conv2 = convGenerator(5, 16); // 5x5, 16 filters
    QVector<QVector<cv::Mat>> res1, res2;
    QVector<cv::Mat> rgbd;

    for (int i = 0; i < rgbdList.length(); i++){
        rgbd = rgbdList.at(i);
        //ZERO PADDING 1
        rgbd = zeropadding(rgbd, 2);
        //CONV1
        res1 = convolution(rgbd, conv1, 1);//stride = 1
        //RELU LAYER
        res1 = relu(res1);
        //POOLING
        res1 = pooling(res1, 4);
        //ZERO PADDING 2
        rgbd = zeropadding(rgbd, 1);
        //CONV2
        res2 = convolution(res1, conv1, 1);//stride = 1
        //RELU LAYER
        res2 = relu(res2);
        //POOLING
        res2 = pooling(res2, 4);
    }
    //RESHAPE
    cv::Mat res3;
    res3 = reshape(res2); //150 * (7*7*16) = 150*784

    //CLASSIFICATION
    QVector<QVector<double>> res4; // [0.89 0.54], ...
    cv::Mat fullyConnLayerWeight = (784, 2, CV_8UC3, cv::Scalar(0,0,0));

    res4 = matrixMult(res3, fullyConnLayerWeight);

    //COST
    QVector<double> cost;
    cost = calculateCost(res4, rgbd_gt);
    double totalCost;
    totalCost = sumUp(cost);

    //BP


    return a.exec();
}

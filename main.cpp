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

    QVector<QVector<QString>> rgbdList = farranger_rgbd();
    QVector<int> rgbd_gt = farranger_gt();//ground truth

    QVector<QVector<cv::Mat>> conv1 = convGenerator(7, 8); // 7x7, 8 filters. The inside QVector is RGBD 4 layer, second QVector is # of filters.
    QVector<QVector<cv::Mat>> conv2 = convGenerator(5, 16); // 5x5, 16 filters
    QVector<QVector<cv::Mat>> res1, res2;

    for (int i = 0; i < rgbdList.length(); i++){
        QVector<cv::Mat> rgbd = rgbdList.at(i);
        //ZERO PADDING 1
        rgbd = zeropadding(rgbd, 2);
        //CONV1
        res1 = convolution(rgbd, conv1, 1);//stride = 1
        //RELU LAYER
        res1 = relu(res1);
        //POOLING
        res1 = pooling(res1, 2);
        //ZERO PADDING 2
        rgbd = zeropadding(rgbd, 1);
        //CONV2
        res2 = convolution(res1, conv1, 1);//stride = 1
        //RELU LAYER
        res2 = relu(res2);
        //POOLING
        res2 = pooling(res2, 2);

    }





    /*
    cv::Mat img = cv::imread(imgPath);

    std::vector<Mat> planes(3);

    cv::namedWindow("PuRSE");
    cv::imshow("PuRSE", img);
    std::cout << (int)img.at<cv::Vec3b>(50,50)[0] << std::endl;
    cv::waitKey(10);*/
    return a.exec();
}

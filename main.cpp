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
    std::vector<cv::Mat> rgbdList = farranger_rgbd(data, 0.07); // former 7% for training.
    QVector<int> rgbd_gt = farranger_gt(data, 0.07);//ground truth, 1 for TL, 0 for other.

    std::vector<cv::Mat> conv1 = convGenerator(7, 8, 1); // 7x7, 8 filters. The inside QVector is RGBD 4 layer, second QVector is # of filters.
    std::vector<cv::Mat> conv2 = convGenerator(5, 16, 2); // 5x5, 16 filters, 8 layers each.
    cv::Mat res1, res2;
    QVector<cv::Mat> res2list;
    cv::Mat rgbd(100, 100, CV_64FC4, cv::Scalar(0, 0, 0, 0));

    for (int i = 0; i < rgbdList.size(); i++){
        std::cout << "iteration: " << i << "  -------------epoch 1-------------" << std::endl;
        rgbd = rgbdList.at(i);
        //ZERO PADDING 1
        rgbd = zeropadding(rgbd, 3);
        //CONV1
        res1 = convolution(rgbd, conv1, 8, 1);
        //RELU LAYER
        res1 = relu(res1, 1);
        //POOLING
        res1 = pooling(res1, 4, 1);
        std::cout << "iteration: " << i << "  -------------epoch 2-------------" << std::endl;
        //ZERO PADDING 2
        res1 = zeropadding(res1, 2);
        //CONV2
        res2 = convolution(res1, conv2, 16, 2);


        cv::namedWindow("PuRSE");
        std::vector<cv::Mat> rgbdlayer,rgblayer;
        cv::Mat rgb;
        cv::split(res2, rgbdlayer);
        rgblayer.push_back(rgbdlayer[0]);
        rgblayer.push_back(rgbdlayer[1]);
        rgblayer.push_back(rgbdlayer[2]);
        cv::merge(rgblayer, rgb);
        cv::convertScaleAbs(rgb,rgb,1,0);
        cv::imshow("PuRSE", rgb);
        cv::waitKey(300);
        std::string name = "../TLRConvNet/images/" +std::to_string(i)+ ".png";
        cv::imwrite(name, rgb);


        //RELU LAYER
        res2 = relu(res2, 2);
        //POOLING
        res2 = pooling(res2, 4, 2);

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

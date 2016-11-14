#ifndef CONVNETS_H
#define CONVNETS_H

#endif // CONVNETS_H
#include <iostream>
#include <QDir>
#include <QStringList>
#include <QApplication>
#include <QVector>
#include <stdio.h>      /* printf */
#include <math.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QTime>

std::vector<cv::Mat> convGenerator(int size, int number, int epoch);
cv::Mat zeropadding(cv::Mat rgbd, int padding);
cv::Mat relu(cv::Mat res, int epoch);
cv::Mat convolution(cv::Mat rgbd, std::vector<cv::Mat> kernel, int filterCount, int epoch);
cv::Mat pooling(cv::Mat res, int size, int epoch);
cv::Mat reshape(QVector<cv::Mat> res);
cv::Mat reshape_testing(QVector<cv::Mat> res);
std::vector<std::vector<double>> probMatInit(std::vector<int> groundTruth);
std::vector<std::vector<double>> calculation(cv::Mat res3, std::vector<std::vector<double>> probabilityMat, std::vector<int> groundTruth);
double correctPercentageCalculation(std::vector<std::vector<double>> errorList, std::vector<int> groundTruth);
std::vector<std::vector<double>> calculateDelta(cv::Mat res3, std::vector<std::vector<double>> errorlist,int learningrate);
std::vector<std::vector<double>> trainFCL(std::vector<std::vector<double>> probabilityMat,std::vector<std::vector<double>> delta);

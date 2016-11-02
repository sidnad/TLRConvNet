#ifndef FILEREADER_H
#define FILEREADER_H

#endif // FILEREADER_H

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

QVector<QVector<QString>> freader();
std::vector<cv::Mat> farranger_rgbd(QVector<QVector<QString>> data, double training_portion );
QVector<int> farranger_gt(QVector<QVector<QString>> data, double training_portion );

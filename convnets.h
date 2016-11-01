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

QVector<cv::Mat> convGenerator(int size, int number);

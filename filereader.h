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

QVector<QVector<QString>> freader();
QVector<QVector<QString>> farranger_rgbd(QVector<QVector<QString>> data, double training_portion );
QVector<int> farranger_gt(QVector<QVector<QString>> data, double training_portion );

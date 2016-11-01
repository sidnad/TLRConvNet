#include "filereader.h"

QVector<QVector<QString>> freader(){

    QString imgPath_positive = "../570CNN/cropped_data/reshape_positive_rgb/";
    QString depthPath_positive = "../570CNN/cropped_data/reshape_positive_depth/";
    QString imgPath_negative = "../570CNN/cropped_data/reshape_negative_rgb/";
    QString depthPath_negative = "../570CNN/cropped_data/reshape_negative_depth/";

    QDir imgPosDir(imgPath_positive);
    QDir depthPosDir(depthPath_positive);
    QDir imgNegDir(imgPath_negative);
    QDir depthNegDir(depthPath_negative);

    imgPosDir.setNameFilters(QStringList()<<"*.png");
    depthPosDir.setNameFilters(QStringList()<<"*.png");
    imgNegDir.setNameFilters(QStringList()<<"*.png");
    depthNegDir.setNameFilters(QStringList()<<"*.png");

    QStringList filesList1 = imgPosDir.entryList();
    QStringList filesList2 = depthPosDir.entryList();
    QStringList filesList3 = imgNegDir.entryList();
    QStringList filesList4 = depthNegDir.entryList();


    QVector<QString> fl1 = QVector<QString>();
    QVector<QString> fl2 = QVector<QString>();
    QVector<QString> fl3 = QVector<QString>();
    QVector<QString> fl4 = QVector<QString>();

    QVector<QVector<QString>> data = QVector< QVector<QString> >();

    for(int i = 0; i < filesList1.length(); i++){
        QString fn1 = filesList1.at(i);
        QString fn2 = filesList2.at(i);
        fl1.push_back(imgPath_positive + fn1);
        fl2.push_back(depthPath_positive + fn2);

        //std::cout << fl1.at(i).toStdString() << std::endl;
    }
    for(int i = 0; i < filesList3.length(); i++){
        QString fn3 = filesList3.at(i);
        QString fn4 = filesList4.at(i);

        fl3.push_back(imgPath_negative + fn3);
        fl4.push_back(depthPath_negative + fn4);

        //std::cout << fl3.at(i).toStdString() << std::endl;
    }

    data.push_back(fl1);
    data.push_back(fl2);
    data.push_back(fl3);
    data.push_back(fl4);

    return data;
}


QVector<QVector<QString>> farranger_rgbd(QVector<QVector<QString>> data, double training_portion){
    QVector<QVector<QString>> rgbdList = QVector<QVector<QString>>();
    int positiveCount, negativeCount, positive4train, negative4train;

    positiveCount = data.at(0).length();
    negativeCount = data.at(2).length();

    positive4train = floor(training_portion * positiveCount);
    negative4train = floor(training_portion * negativeCount);

    QVector<QString> fp_rgb, fp_depth;
    for (int i = 0; i < positive4train; i++){
        fp_rgb.push_back(data.at(0).at(i));
        fp_depth.push_back(data.at(1).at(i));
    }
    for (int i = 0; i < positive4train; i++){
        fp_rgb.push_back(data.at(2).at(i));
        fp_depth.push_back(data.at(3).at(i));
    }
    rgbdList.push_back(fp_rgb);
    rgbdList.push_back(fp_depth);
    return rgbdList;
}


QVector<int> farranger_gt(QVector<QVector<QString>> data, double training_portion){
    QVector<int> rgbd_gt = QVector<int>();
    int positiveCount, negativeCount, positive4train, negative4train;

    positiveCount = data.at(0).length();
    negativeCount = data.at(2).length();

    positive4train = floor(training_portion * positiveCount);
    negative4train = floor(training_portion * negativeCount);

    for (int i = 0; i < positive4train; i++){
        rgbd_gt.push_back(1);
    }
    for (int i = 0; i < positive4train; i++){
        rgbd_gt.push_back(0);
    }

    return rgbd_gt;
}

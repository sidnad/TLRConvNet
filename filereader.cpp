#include "filereader.h"

QVector<QVector<QString>> freader(){

    QString imgPath_positive = "../TLRConvNet/cropped_data/reshape_positive_rgb/";
    QString depthPath_positive = "../TLRConvNet/cropped_data/reshape_positive_depth256/";
    QString imgPath_negative = "../TLRConvNet/cropped_data/reshape_negative_rgb/";
    QString depthPath_negative = "../TLRConvNet/cropped_data/reshape_negative_depth256/";

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


std::vector<cv::Mat> farranger_rgbd(QVector<QVector<QString>> data, double training_portion){
    std::vector<cv::Mat> rgbdList;
    cv::Mat img, depth;
    cv::Mat rgbd;
    std::vector<cv::Mat> splitBGR, splitDepth;
    int positiveCount, negativeCount, positive4train, negative4train;

    positiveCount = data.at(0).length();
    negativeCount = data.at(2).length();

    positive4train = floor(training_portion * positiveCount);
    negative4train = floor(training_portion * negativeCount);

    std::cout << "Total data amount: " << positiveCount + negativeCount << ". Positive count: " << positiveCount << ". Negative count: " << negativeCount << std::endl;
    std::cout << "Total training data (7%) amount: " << positive4train + negative4train << std::endl;

    QVector<QString> fp_rgb, fp_depth;
    for (int i = 0; i < positive4train; i++){
        splitBGR.clear();
        splitDepth.clear();
        cv::Mat rgbd;
        //std::cout << data.at(0).at(i).toStdString() << std::endl;
        img = cv::imread(data.at(0).at(i).toStdString());
        depth = cv::imread(data.at(1).at(i).toStdString());
        depth.convertTo(depth, CV_8UC1);
        cv::split(img, splitBGR);
        cv::split(depth, splitDepth);
        splitBGR.push_back(splitDepth[0]);
        cv::merge(splitBGR, rgbd);
        rgbdList.push_back(rgbd);
/*
        cv::namedWindow("PuRSE");
        cv::imshow("PuRSE", img);
        cv::waitKey(100);*/
    }
    for (int i = 0; i < negative4train; i++){
        splitBGR.clear();
        splitDepth.clear();
        cv::Mat rgbd;
        //std::cout << data.at(2).at(i).toStdString() << std::endl;
        img = cv::imread(data.at(2).at(i).toStdString());
        depth = cv::imread(data.at(3).at(i).toStdString());
        depth.convertTo(depth, CV_8UC1);
        cv::split(img, splitBGR);
        cv::split(depth, splitDepth);
        splitBGR.push_back(splitDepth[0]);
        cv::merge(splitBGR, rgbd);
        rgbdList.push_back(rgbd);
    }

    return rgbdList;
}


std::vector<int> groundTruthVector(QVector<QVector<QString>> data, double training_portion ){
    int positiveCount, negativeCount, positive4train, negative4train;
    std::vector<int> groundTruth;
    positiveCount = data.at(0).length();
    negativeCount = data.at(2).length();

    positive4train = floor(training_portion * positiveCount);
    negative4train = floor(training_portion * negativeCount);
    std::cout << "Positive training data (7%) amount: " << positive4train << " Negative training data (7%) amount: " << negative4train << std::endl;
    for (int i = 0; i < positive4train; i++){
        groundTruth.push_back(1);
    }
    for (int i = 0; i < negative4train; i++){
        groundTruth.push_back(0);
    }
    return groundTruth;
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

void showImage(cv::Mat image, std::string saveDir){
    cv::namedWindow("image");
    std::vector<cv::Mat> rgbdlayer,rgblayer;
    cv::Mat rgb;
    cv::split(image, rgbdlayer);
    rgblayer.push_back(rgbdlayer[0]);
    rgblayer.push_back(rgbdlayer[1]);
    rgblayer.push_back(rgbdlayer[2]);
    cv::merge(rgblayer, rgb);
    cv::convertScaleAbs(rgb,rgb,1,0);
    cv::imshow("image", rgb);
    cv::waitKey(10);
    //std::string name = saveDir + std::to_string(i) + ".png";
    //cv::imwrite(name, rgb);
}

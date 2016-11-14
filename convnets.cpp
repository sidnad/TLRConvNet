#include "convnets.h"

std::vector<cv::Mat> convGenerator(int size, int number, int epoch){
    std::vector<cv::Mat> convnetList;
    cv::Mat convnet;
    double r;
    //qsrand(QTime::currentTime().msec());
    qsrand(1); //choose seed
    for (int i = 0; i < number; i++){
        if (epoch == 1){
            convnet = cv::Mat(size, size, CV_64FC(4), cv::Scalar(0, 0, 0, 0));
        }
        else if (epoch == 2){
            convnet = cv::Mat(size, size, CV_64FC(8), cv::Scalar(0, 0, 0, 0));
        }

        for (int k = 0; k < size; k++){
            for (int l = 0; l < size; l++){
                for (int c = 0; c < 4; c++){
                    r = (double)qrand()/RAND_MAX;
                    convnet.at<cv::Vec4d>(k, l)[c] = r;
                }
            }
        }
        convnetList.push_back(convnet);
    }

    return convnetList;
}

cv::Mat zeropadding(cv::Mat rgbd, int padding){
    cv::copyMakeBorder( rgbd, rgbd, padding, padding, padding, padding, cv::BORDER_CONSTANT );
    //std::cout << "dimension after zero padding: " << rgbd.rows << std::endl;
    return rgbd;
}

cv::Mat relu(cv::Mat res, int epoch){
    std::vector<cv::Mat> splitLayer;
    cv::split(res, splitLayer);
    for(int i = 0; i < splitLayer.size(); i++){

        for(int x = 0; x < splitLayer[i].rows; x++){
            for(int y = 0; y < splitLayer[i].cols; y++){
                if (splitLayer[i].at<double>(x,y) < 0){
                    splitLayer[i].at<double>(x,y) = 0;
                }
            }
        }
        /*
        if(epoch == 2){
            cv::namedWindow("PuRSE");
            cv::convertScaleAbs(splitLayer[i],splitLayer[i],1,0);
            cv::imshow("PuRSE", splitLayer[i]);
            cv::waitKey(300);
        }*/
    }
    cv::merge(splitLayer, res);
    return res;
}

cv::Mat convolution(cv::Mat rgbd, std::vector<cv::Mat> kernel, int filterCount, int epoch){
     //view rgb image
    /*
    if (epoch == 1){
        cv::namedWindow("PuRSE");
        std::vector<cv::Mat> rgbdlayer,rgblayer;
        cv::Mat rgb;
        cv::split(rgbd, rgbdlayer);
        rgblayer.push_back(rgbdlayer[0]);
        rgblayer.push_back(rgbdlayer[1]);
        rgblayer.push_back(rgbdlayer[2]);
        cv::merge(rgblayer, rgb);
        cv::convertScaleAbs(rgb,rgb,1,0);
        cv::imshow("PuRSE", rgb);
        cv::waitKey(300);
    }*/

    std::cout << "dimension of the input: " << rgbd.rows << std::endl;
    std::cout << "dimension of each kernel: " << kernel.at(0).rows << std::endl;

    double delta = 0;
    std::vector<cv::Mat> resList;
    int size;
    if (epoch == 1){
        size = 100;
    }
    else if (epoch == 2){
        size = 25;
    }
    cv::Mat res(size, size, CV_64F);
    cv::Mat output;
    double tempval;

    std::vector<cv::Mat> rgbdlist; // 4 layers / 8 layers
    cv::split(rgbd,rgbdlist);
    std::cout << "rgbd's dimension: " << rgbdlist.size() << std::endl;
    std::vector<cv::Mat> singleKernelList; // 4 layers / 8 layers

/*
    for(int u = 0; u < rgbdlist.at(0).rows; u++){
        for(int v = 0; v < rgbdlist.at(0).cols; v++){
            std::cout << "----------------------value: " << rgbd.at<cv::Vec4d>(u,v)[0] << std::endl;
        }
    }
*/


    for(int i = 0; i < kernel.size(); i++){ // 8 or 16 loops

        cv::split(kernel.at(i),singleKernelList);
        for(int x = 0; x < rgbd.rows - kernel.at(i).cols + 1; x++){
            for(int y = 0; y < rgbd.cols - kernel.at(i).cols + 1; y++){
                tempval = 0;
                int rgbdVal;
                for(int u = 0; u < kernel.at(i).rows; u++){
                    for(int v = 0; v < kernel.at(i).cols; v++){
                            for (int l = 0; l < rgbdlist.size(); l++){
                                double kernelVal = singleKernelList.at(l).at<double>(u,v);
                                if(epoch==1) rgbdVal = rgbd.at<cv::Vec4b>(x+u, y+v)[l];
                                if(epoch==2) rgbdVal = rgbdlist[l].at<cv::Vec4b>(x+u, y+v)[0];
                                //std::cout << kernelVal << " " << rgbdVal << std::endl;
                                tempval = tempval + kernelVal * rgbdVal;
                            }
                    }
                }
                tempval = tempval/(kernel.at(i).rows*kernel.at(i).cols)/4; // val/25
                res.at<double>(x, y) = tempval;
                //std::cout<< tempval<<std::endl;
            }
        }
/*
        if(epoch == 2){
            cv::namedWindow("PuRSE");
            cv::convertScaleAbs(res,res,1,0);
            cv::imshow("PuRSE", res);
            cv::waitKey(500);
        }
*/

/*
        //kernel print
        for(int u = 0; u < kernel.at(i).rows; u++){
            for(int v = 0; v < kernel.at(i).cols; v++){
                std::cout<< " "<< kernel.at(i).at<double>(u,v);
            }
            std::cout<< " "<<std::endl;
        }
        std::cout<< " "<<std::endl;
        std::cout<< " "<<std::endl;
*/
        resList.push_back(res);
    }


    cv::merge(resList, output);
    return output;
}

cv::Mat pooling(cv::Mat res, int size, int epoch){
    cv::Mat des = res;
    std::cout << "epoch: " << epoch << " old dimension before pooling: " << res.rows << std::endl;
    if (epoch == 1){
        pyrDown( res, des, cv::Size( res.cols/(2), res.rows/(2) ));
        pyrDown( des, des, cv::Size( des.cols/(2), des.rows/(2) ));
    }
    if (epoch == 2){
        pyrDown( res, des, cv::Size( res.cols/(2), res.rows/(2) ));
        pyrDown( des, des, cv::Size( des.cols/(2), des.rows/(2) ));
    }
    std::cout << "epoch: " << epoch << " new dimension after pooling: " << des.rows << std::endl;

/*
    cv::namedWindow("PuRSE");
    std::vector<cv::Mat> reslayer,rgblayer;
    cv::Mat rgb;
    cv::split(des, reslayer);
    rgblayer.push_back(reslayer[0]);
    rgblayer.push_back(reslayer[1]);
    rgblayer.push_back(reslayer[2]);
    cv::merge(rgblayer, rgb);
    cv::convertScaleAbs(rgb,rgb,1,0);
    cv::imshow("PuRSE", rgb);
    cv::waitKey(3000);
*/
    return des;
}

cv::Mat reshape(QVector<cv::Mat> res){
    cv::Mat output(51, 576, CV_64F, cv::Scalar(0));
    cv::Mat currRes;
    std::vector<cv::Mat> splitLayer;
    for(int i = 0; i < res.length(); i++){
        currRes = res.at(i);


        cv::split(currRes, splitLayer);
        for(int j = 0; j < splitLayer.size(); j++){
            for(int x = 0; x < 6; x++){
                for(int y = 0; y < 6; y++){
                    output.at<double>(i, j*36 + x*5 + y) = currRes.at<double>(x,y);
                }
            }
        }
    }
    return output;
}

cv::Mat reshape_testing(QVector<cv::Mat> res){
    cv::Mat output(21, 576, CV_64F, cv::Scalar(0));
    cv::Mat currRes;
    std::vector<cv::Mat> splitLayer;
    for(int i = 0; i < res.length(); i++){
        currRes = res.at(i);


        cv::split(currRes, splitLayer);
        for(int j = 0; j < splitLayer.size(); j++){
            for(int x = 0; x < 6; x++){
                for(int y = 0; y < 6; y++){
                    output.at<double>(i, j*36 + x*5 + y) = currRes.at<double>(x,y);
                }
            }
        }
    }
    return output;
}

std::vector<std::vector<double>> probMatInit(std::vector<int> groundTruth){
    std::vector<std::vector<double>> probabilityMat;
    std::vector<double> vector;
    qsrand(1); //choose seed
    int length = 576;//groundTruth.size();
    for(int i = 0; i < length; i++){
        vector.push_back((double)qrand()/RAND_MAX);
    }
    probabilityMat.push_back(vector);
    vector.clear();
    for(int i = 0; i < length; i++){
        vector.push_back((double)qrand()/RAND_MAX);
    }
    probabilityMat.push_back(vector);
    return probabilityMat;
}

std::vector<std::vector<double>> calculation(cv::Mat res3, std::vector<std::vector<double>> probabilityMat, std::vector<int> groundTruth){
    std::vector<std::vector<double>> errorList;
    std::vector<double> positiveErrorList, negativeErrorList;
    std::vector<double> guessList;
    double positiveVal, negativeVal, positiveError, negativeError;
    for(int j = 0; j < res3.rows; j++){
        positiveVal = 0;
        negativeVal = 0;
        for(int i = 0; i < probabilityMat.at(0).size(); i++){
            positiveVal += (double) probabilityMat.at(0).at(i) * res3.at<double>(j, i);
            negativeVal += (double) probabilityMat.at(1).at(i) * res3.at<double>(j, i);
            //std::cout << "res3.at<int>(j, i): " << res3.at<double>(j, i) << std::endl;
            //std::cout << "positive val: " << positiveVal << "  Negative val: " << negativeVal << std::endl;
        }
        //std::cout << "positive val: " << positiveVal << "  Negative val: " << negativeVal << std::endl;
        //NORMALIZE
        positiveError = groundTruth.at(j) - positiveVal / (positiveVal+negativeVal);
        negativeError = groundTruth.at(j) - negativeVal / (positiveVal+negativeVal);

        positiveErrorList.push_back(positiveError);
        negativeErrorList.push_back(negativeError);
        guessList.push_back((positiveVal > negativeVal)? 1:0);
    }
    errorList.push_back(positiveErrorList);
    errorList.push_back(negativeErrorList);
    errorList.push_back(guessList);
    return errorList;
}

double correctPercentageCalculation(std::vector<std::vector<double>> errorList, std::vector<int> groundTruth){
    int correctCount = 0;
    for(int i = 0; i < errorList.at(2).size(); i++){
        if((int)errorList.at(2).at(i) == groundTruth.at(i)){
            correctCount++;
        }
    }
    double correctPercentage = (double) correctCount / errorList.at(0).size();
    return correctPercentage;
}

std::vector<std::vector<double>> calculateDelta(cv::Mat res3, std::vector<std::vector<double>> errorlist,int learningrate){
    cv::Mat res3transpose(576, 51, CV_64F, cv::Scalar(0));                          //Calculate transpose
    for(int i=0;i<res3.rows;i++){
        for(int j=0;j<res3.cols;j++){
            res3transpose.at<double>(j,i)=res3.at<double>(i,j);
        }
    }

    cv::Mat errormat(51, 2, CV_64F, cv::Scalar(0));                      //Get matrix from error list
    for(int i=0;i<51;i++){
        errormat.at<double>(i,0)=errorlist.at(0).at(i);
        errormat.at<double>(i,1)=errorlist.at(1).at(i);
    }

    cv::Mat deltamat(576, 2, CV_64F, cv::Scalar(0));
    for(int i=0;i<res3transpose.rows;i++){            //Multiply error and transpose
        for(int j=0;j<errormat.cols;j++){
            for(int k=0;k<res3transpose.cols;k++){
                deltamat.at<double>(i,j)+=res3transpose.at<double>(i,k)*errormat.at<double>(k,j);
            }
        }
    }

    std::vector<std::vector<double>> delta; //576*2
    std::vector<double> col;
    for (int i = 0; i < deltamat.rows; i++){
        col.push_back(deltamat.at<double>(i,0));
    }
    delta.push_back(col);
    col.clear();
    for (int i = 0; i < deltamat.rows; i++){
        col.push_back(deltamat.at<double>(i,1));
    }
    delta.push_back(col);

    for (int i = 0; i < delta.at(0).size(); i++){
        delta.at(0).at(i)=delta.at(0).at(i)*learningrate;
        delta.at(1).at(i)=delta.at(1).at(i)*learningrate;
    }
    return delta;
}

std::vector<std::vector<double>> trainFCL(std::vector<std::vector<double>> probabilityMat,std::vector<std::vector<double>> delta){
    for (int i = 0; i < delta.at(0).size(); i++){
        probabilityMat.at(0).at(i) += delta.at(0).at(i);//+probabilityMat.at(0).at(i);
        probabilityMat.at(1).at(i) += delta.at(1).at(i);//+probabilityMat.at(1).at(i);
    }
    return probabilityMat;
}

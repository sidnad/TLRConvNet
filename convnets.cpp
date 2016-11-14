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


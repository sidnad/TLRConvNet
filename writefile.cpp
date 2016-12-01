#include "writefile.h"
/*
void writeConv1(std::vector<cv::Mat> conv1){

}
*/
void writeProbMat(std::vector<std::vector<double>> probabilityMat){
    // Declare what you need
    std::cout << "probabilityMat file writing start" << std::endl;

/*
    cv::FileStorage file("../TLRConvNet/probMat.xml", cv::FileStorage::WRITE);
    int length = probabilityMat.at(0).size();
    cv::Mat probMat(2, length, CV_64F, cv::Scalar(0));
    for(int i = 0; i < length; i++){
        probMat.at<double>(0,i) = probabilityMat.at(0).at(i);
        probMat.at<double>(1,i) = probabilityMat.at(1).at(i);
    }
    // Write to file!
    file << "AAAAA" << probMat;
*/
    cv::FileStorage fs("../TLRConvNet/probMat1.yml", cv::FileStorage::WRITE);
    cv::write(fs , "probMat1", probabilityMat.at(0));
    fs.release();
    cv::FileStorage fs2("../TLRConvNet/probMat2.yml", cv::FileStorage::WRITE);
    cv::write(fs2 , "probMat2", probabilityMat.at(1));
    fs2.release();

    //read files
    /*
    std::vector<double> newKeypoints;
    cv::FileStorage fs2("../TLRConvNet/probMat1.yml", cv::FileStorage::READ);
    cv::FileNode kptFileNode = fs2["probMat1"];
    cv::read(kptFileNode, newKeypoints);
    fs2.release();
    */

    std::cout << "probabilityMat file writing finished." << std::endl;
}

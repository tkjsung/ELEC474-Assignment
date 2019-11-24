// ELEC 474 Take Home Exam
// Author: Tom Sung

// C++ Standard Libraries
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>

// OpenCV Imports
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>

// Use OpenCV namespace
using namespace cv;
using namespace std;

// C++ Function Declaration
void importImages();
vector<String> getImages(String path);
vector<Mat> loadImages(vector<String> list);
void imgORBMatch(String identify);
void imgSIFTMatch(String identify);
void stitching(String identify);

// Global Variables
//vector<string> listofImages; // Original plan to do all the file names... ignore forever.
vector<string>listOfChurchImages = getImages("StJames/*.jpg");
vector<string>listOfOfficeImages = getImages("office2/*.jpg");
vector<string>listOfWLHImages = getImages("WLH/*.jpg");
//vector<Mat> matChurch = loadImages(listOfChurchImages);
//vector<Mat> matOffice = loadImages(listOfOfficeImages);
//vector<Mat> matwlh = loadImages(listOfWLHImages);
Mat pano;

int main()
{

//    vector<Mat> matChurch = loadImages(listOfChurchImages);
//    cout << matChurch[0] << endl;
//    imshow("test",matChurch[2]);
//    waitKey();
    
    //TODO:: Metrics to determine good matches and count them?
    //TODO:: Determine which image to take as the base image...?
    //TODO:: Make sure stitching tool works
    
    imgORBMatch("church"); // Choices: wlh, church, or office
//    imgSIFTMatch("church"); // Choices: wkh, church, or office
    
//    stitching("church");
    
    return 0;
}

// Reference: https://towardsdatascience.com/image-stitching-using-opencv-817779c86a83
void imgORBMatch(String identify)
{
    vector<Mat> matSource;
    
    // Load the necessary matrices into the code
    if(identify == "church")
    {
        matSource = loadImages(listOfChurchImages);
        cout << "Church selected." << endl;
    }
    else if(identify == "office")
    {
        matSource = loadImages(listOfOfficeImages);
        cout << "Office selected." << endl;
    }
    else if(identify == "wlh"){
        matSource = loadImages(listOfWLHImages);
        cout << "WLH selected." << endl;
    }
    else
    {
        cout << "Error occurred" << endl;
        return;
    }
    
    // All variable declarations
    Mat img1, img2, descriptors1, descriptors2, h, matchesMatrix, resultImg;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<Point2f> img1Points, img2Points;
    vector<DMatch> matches12, matches21, filteredMatches12, filteredMatches21;
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorMatcher> matcher;
    int nfeatures = 500;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    int patchSize = 31;
    int j = 1;
    int first = 0;
    
//    cout << matSource.size() << endl;
    
    for (j = 1; j < matSource.size(); j++)
    {
        if (first == 0)
        {
            img1 = matSource[0];
            img2 = matSource[1];
            first = 1;
        }
        else{
            img1 = pano;
            img2 = matSource[j];
        }
        
        
        // Feature Detector
        detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, ORB::HARRIS_SCORE, patchSize);
        detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
        detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);
        
        // Match features
        //vector<DMatch> matches12, matches21, filteredMatches12, filteredMatches21;
        matcher = DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(descriptors1, descriptors2, matches12, Mat());
        matcher->match(descriptors2, descriptors1, matches21, Mat());
        
        // Sort matches to delete the matches that aren't so great
//        sort(matches12.begin(), matches12.end());
        // Remove not good matches
//        const int numGoodMatches = matches12.size() * 0.05f;
//        matches12.erase(matches12.begin() + numGoodMatches, matches12.end());
        
        // From : https://answers.opencv.org/question/15/how-to-get-good-matches-from-the-orb-feature-detection-algorithm/
        for (size_t i = 0; i < matches12.size(); i++)
        {
            DMatch forward = matches12[i];
            DMatch backward = matches21[forward.trainIdx];
            if(backward.trainIdx == forward.queryIdx)
                filteredMatches12.push_back(forward);
        }
        
        // This for statement might not be needed...
        //    for (size_t i = 0; i < matches21.size(); i++)
        //    {
        //        DMatch forward = matches21[i];
        //        DMatch backward = matches21[forward.trainIdx];
        //        if(backward.trainIdx == forward.queryIdx)
        //            filteredMatches21.push_back(forward);
        //    }
        
        for (size_t i = 0; i < filteredMatches12.size(); i++)
        {
            img1Points.push_back(keypoints1[filteredMatches12[i].queryIdx].pt);
            img2Points.push_back(keypoints2[filteredMatches12[i].trainIdx].pt);
        }
        
        // Find homography (source pts, dst pts, algorithm)
        h = findHomography(img2Points, img1Points, RANSAC);
        
        
        // Draw Matches algorithm
        drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches12, matchesMatrix);
        imshow("Matches",matchesMatrix);
        waitKey();
        
        //    cout << img1.size[0] << endl; // height
        //    cout << img1.size[1] << endl; // width
        //    cout << img1.size << endl;
        
        // Warp image according to the homography
//        warpPerspective(img2, pano, h, Size((img1.rows + img2.rows),(img1.cols + img2.cols)));
//        Mat half(pano, Rect(0,0,img1.cols,img1.rows));
//        img1.copyTo(half);
        
        warpPerspective(img2, resultImg, h, Size((img1.rows + img2.rows),(img1.cols + img2.cols)));
        pano = Mat(resultImg, Rect(0,0,img1.cols,img1.rows));
        img1.copyTo(pano);
        
        imshow("Perspective change",resultImg);
        waitKey();
        imshow("Changed image",pano);
        waitKey();
        
        
        // Clear all variables
        keypoints1.clear();
        keypoints2.clear();
        img1Points.clear();
        img2Points.clear();
        descriptors1.release();
        descriptors2.release();
        h.release();
        matchesMatrix.release();
        resultImg.release();
        matches12.clear();
        matches21.clear();
        filteredMatches12.clear();
        filteredMatches21.clear();
//        half.release();
        detector->clear();
        matcher->clear();
    }
    
}


void imgSIFTMatch(String identify)
{
    vector<Mat> matSource;
    
    // Load the necessary matrices into the code
    if(identify == "church")
    {
        matSource = loadImages(listOfChurchImages);
        cout << "Church selected." << endl;
    }
    else if(identify == "office")
    {
        matSource = loadImages(listOfOfficeImages);
        cout << "Office selected." << endl;
    }
    else if(identify == "wlh"){
        matSource = loadImages(listOfWLHImages);
        cout << "WLH selected." << endl;
    }
    else
    {
        cout << "Error occurred" << endl;
        return;
    }
    Mat img1 = matSource[2];
    Mat img2 = matSource[3];
    
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    vector<Point2f> img1Points, img2Points;
    
    int nfeatures = 500;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 0.04;
    double sigma = 1.6;
    
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(nfeatures, nOctaveLayers,contrastThreshold, edgeThreshold, sigma); // Use all default values for SIFT feature detection
    f2d->detect(img1,keypoints1);
    f2d->detect(img2,keypoints2);
    f2d->compute(img1,keypoints1,descriptors1);
    f2d->compute(img2,keypoints2,descriptors2);
    
    cout << keypoints1.size() << endl;
    cout << descriptors1.size() << endl;
    
    vector<DMatch> matches12, matches21, filteredMatches12;
    BFMatcher matcher;
    matcher.match(descriptors1, descriptors2, matches12);
    matcher.match(descriptors2, descriptors1, matches21);
    
    
    // Filtering matches...
//    vector<KeyPoint> newKeypoints1, newKeypoints2;
    
    for (size_t i = 0; i < matches12.size(); i++)
    {
        DMatch forward = matches12[i];
        DMatch backward = matches21[forward.trainIdx];
        if(backward.trainIdx == forward.queryIdx)
        {
            filteredMatches12.push_back(forward);
//            newKeypoints1.push_back(keypoints1[i]);
//            newKeypoints2.push_back(keypoints2[i]);
        }
    }
    
    
    // Getting points for homography (can't use old keypoints... need filtering)
    // Reference: https://stackoverflow.com/questions/5937264/using-opencv-descriptor-matches-with-findfundamentalmat
//    KeyPoint::convert(newKeypoints1, img1Points);
//    KeyPoint::convert(newKeypoints2, img2Points);
//
//    cout << keypoints1.size() << endl;
//    cout << newKeypoints1.size() << endl;
//
    
    for (size_t i = 0; i < filteredMatches12.size(); i++)
    {
        img1Points.push_back(keypoints1[filteredMatches12[i].queryIdx].pt);
        img2Points.push_back(keypoints2[filteredMatches12[i].trainIdx].pt);
    }
    
    // Homography: Want second image to be place on first image, not the other way around (src, dst, method)
    Mat h = findHomography(img2Points, img1Points, RANSAC);
    
    // Draw Matches algorithm
    Mat matchesMatrix;
    drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches12, matchesMatrix);
    imshow("Matches",matchesMatrix);
    waitKey();
    
    // Warp image according to the homography
    warpPerspective(img2, matchesMatrix, h, img1.size());
    imshow("Perspective change",matchesMatrix);
    waitKey();
    
}

vector<String> getImages(String path)
{
    // This function retrieves all images from a folder and returns them in a vector of strings of the file names.
    
    String filePath = path;
    vector<String> listOfImages;
    glob(filePath, listOfImages, false);
    
    return listOfImages;
}

// Need new function to extract features and put them in variables
vector<Mat> loadImages(vector<String> list)
{
    vector<Mat> matImg;
    
    for (int i = 0; i < list.size(); i++)
    {
        Mat img = imread(list[i]);
        matImg.push_back(img);
    }
    
//    imshow("test",matImg[2]);
//    waitKey();
    
    return matImg;
}

void stitching(String identify)
{
    // Referenced from: https://docs.opencv.org/3.4/d8/d19/tutorial_stitcher.html
    
    vector<Mat> matSource;
    
    // Load the necessary matrices into the code
    if(identify == "church")
    {
        matSource = loadImages(listOfChurchImages);
        cout << "Church selected." << endl;
    }
    else if(identify == "office")
    {
        matSource = loadImages(listOfOfficeImages);
        cout << "Office selected." << endl;
    }
    else if(identify == "wlh"){
        matSource = loadImages(listOfWLHImages);
        cout << "WLH selected." << endl;
    }
    else
    {
        cout << "Error occurred" << endl;
        return;
    }
    
    Mat pano;
    Stitcher::Mode mode = Stitcher::PANORAMA;
    Ptr<Stitcher>stitcher = Stitcher::create(mode);
    Stitcher::Status status = stitcher->stitch(matSource, pano);
    
    if(status != Stitcher::OK)
    {
        cout << "Cannot stitch images" << endl;
        return;
    }
    
    imshow("Pano image",pano);
    waitKey();
    
    
    
}

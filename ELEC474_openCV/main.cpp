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

// Global Variables
//vector<string> listofImages; // Original plan to do all the file names... ignore forever.
vector<string>listOfChurchImages = getImages("StJames/*.jpg");
vector<string>listOfOfficeImages = getImages("office2/*.jpg");
vector<string>listOfWLHImages = getImages("WLH/*.jpg");
//vector<Mat> matChurch = loadImages(listOfChurchImages);
//vector<Mat> matOffice = loadImages(listOfOfficeImages);
//vector<Mat> matwlh = loadImages(listOfWLHImages);

int main()
{

//    vector<Mat> matChurch = loadImages(listOfChurchImages);
//    cout << matChurch[0] << endl;
//    imshow("test",matChurch[2]);
//    waitKey();
    
    cout << "TO DO LIST:" << endl;
    cout << "1. Metrics to determine good matches and count them?" << endl;
    cout << "2. Determine which image to take as the base image...?" << endl;
    cout << "3. Make sure stitching tool works" << endl;
    
//    imgORBMatch("church"); // Choices: wlh, church, or office
    imgSIFTMatch("church"); // Choices: wkh, church, or office
    
    return 0;
}


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
    Mat img1 = matSource[2];
    Mat img2 = matSource[3];
    
    
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    vector<Point2f> img1Points, img2Points;
    
    int nfeatures = 500;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    int patchSize = 31;
    
    // Feature Detector
    Ptr<FeatureDetector> detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, ORB::HARRIS_SCORE, patchSize);
    detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);
    
    // Match features
    vector<DMatch> matches12, matches21, filteredMatches12, filteredMatches21;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches12, Mat());
    matcher->match(descriptors2, descriptors1, matches21, Mat());

    
    
    // Sort matches to delete the matches that aren't so great
//    sort(matches12.begin(), matches12.end());
    // Remove not good matches
//    const int numGoodMatches = matches12.size() * 0.15f;
//    matches12.erase(matches.begin() + numGoodMatches, matches12.end());
    
    // From : https://answers.opencv.org/question/15/how-to-get-good-matches-from-the-orb-feature-detection-algorithm/
    for (size_t i = 0; i < matches12.size(); i++)
    {
        DMatch forward = matches12[i];
        DMatch backward = matches21[forward.trainIdx];
        if(backward.trainIdx == forward.queryIdx)
            filteredMatches12.push_back(forward);
    }
    for (size_t i = 0; i < matches21.size(); i++)
    {
        DMatch forward = matches21[i];
        DMatch backward = matches21[forward.trainIdx];
        if(backward.trainIdx == forward.queryIdx)
            filteredMatches21.push_back(forward);
    }

    for (size_t i = 0; i < matches12.size(); i++)
    {
        img1Points.push_back(keypoints1[matches12[i].queryIdx].pt);
        img2Points.push_back(keypoints2[matches12[i].trainIdx].pt);
    }
    
    
    // Find homography (source pts, dst pts, algorithm)
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
    
    // Put all the image matrices into a vector to use stitcher tool... potentially.
    vector<Mat> resultImg;
    resultImg.push_back(img1);
    resultImg.push_back(matchesMatrix);
    
    Mat pano;
    Stitcher::Mode mode = Stitcher::PANORAMA;
    Ptr<Stitcher>stitcher = Stitcher::create(mode);
    Stitcher::Status status = stitcher->stitch(resultImg, pano);
    
    if(status != Stitcher::OK)
    {
        cout << "Cannot stitch images" << endl;
        return;
    }
    
    imshow("Pano image",pano);
    waitKey();
    
    
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
    
    //TODO:: Transfer all code from SIFT back to ORB and continue development in ORB.
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

void importImages()
{
    string wlh1 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151846.jpg";
    string wlh2 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151850.jpg";
    string wlh3 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151853.jpg";
    string wlh4 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151855.jpg";
    string wlh5 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151857.jpg";
    string wlh6 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151905.jpg";
    string wlh7 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151907.jpg";
    string wlh8 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151910.jpg";
    string wlh9 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151912.jpg";
    string wlh10 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151914.jpg";
    string wlh11 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151915.jpg";
    string wlh12 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151918.jpg";
    string wlh13 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151920.jpg";
    string wlh14 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151922.jpg";
    string wlh15 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151923.jpg";
    string wlh16 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151925.jpg";
    string wlh17 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151927.jpg";
    string wlh18 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151929.jpg";
    string wlh19 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151930.jpg";
    string wlh20 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151932.jpg";
    string wlh21 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151933.jpg";
    string wlh22 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151935.jpg";
    string wlh23 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151936.jpg";
    string wlh24 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151938.jpg";
    string wlh25 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/WLH/20191119_151940.jpg";
    
    string church1 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152004.jpg";
    string church2 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152007.jpg";
    string church3 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152009.jpg";
    string church4 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152011.jpg";
    string church5 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152013.jpg";
    string church6 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152015.jpg";
    string church7 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152017.jpg";
    string church8 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152020.jpg";
    string church9 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152022.jpg";
    string church10 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152023.jpg";
    string church11 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152026.jpg";
    string church12 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152029.jpg";
    string church13 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152031.jpg";
    string church14 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152033.jpg";
    string church15 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152036.jpg";
    string church16 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/StJames/20191119_152040.jpg";
    
    string office1 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170646.jpg";
    string office2 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170650.jpg";
    string office3 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170652.jpg";
    string office4 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170653.jpg";
    string office5 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170655.jpg";
    string office6 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170657.jpg";
    string office7 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170658.jpg";
    string office8 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170700.jpg";
    string office9 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170702.jpg";
    string office10 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170704.jpg";
    string office11 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170706.jpg";
    string office12 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170707.jpg";
    string office13 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170709.jpg";
    string office14 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170711.jpg";
    string office15 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170712.jpg";
    string office16 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170714.jpg";
    string office17 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170715.jpg";
    string office18 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170717.jpg";
    string office19 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170719.jpg";
    string office20 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170721.jpg";
    string office21 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170723.jpg";
    string office22 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170725.jpg";
    string office23 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170727.jpg";
    string office24 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170729.jpg";
    string office25 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170730.jpg";
    string office26 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170732.jpg";
    string office27 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170733.jpg";
    string office28 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170735.jpg";
    string office29 = "/Users/tomsung/Desktop/ELEC474_openCV/ELEC474_openCV/office2/20191119_170737.jpg";
    
    
}

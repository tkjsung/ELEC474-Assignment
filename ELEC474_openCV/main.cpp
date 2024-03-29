/*
 Course: ELEC 474 - Machine Vision
 Term: Fall 2019
 Instructors: Prof. Michael Greenspan and others
 Title: Take Home Exam
 Date: November 28, 2019
 
 Copyright (c) 2019 by Tom (Ke Jun) Sung (Student #: 20001387).
 I verify that the code developed below is original and developed by myself. Any code that is re-used below by others will be a breach in copyright and Academic Integrity under Queen's University Student Code of Conduct.
 
 ELEC 474 is offered by the Department of Electrical and Computer Engineering at Queen's University in Kingston, ON, Canada.
 
 */

// C++ Standard Libraries
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>

// OpenCV Library Import
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>

// Use OpenCV namespace
using namespace cv;
using namespace std;

// C++ Function Declaration
vector<String> getImages(String path);
vector<Mat> loadImages(vector<String> list);
vector<Mat> loadMat(String identify);
void imgMatch(vector<Mat> matSource, int matchSensitivity);
Mat imgTransform(Mat img1, Mat img2, vector<DMatch> matches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2);
void imgMatchStitch(vector<Mat> & matSource);
void padding(vector<Point> pt_transform, int &desiredWidth, int &desiredHeight);
void stitching(vector<Mat> inputImg);

// Global Variables
vector<string>listOfChurchImages = getImages("StJames/*.jpg"); // Import all images in the folder into the program.
vector<string>listOfOfficeImages = getImages("office2/*.jpg");
vector<string>listOfWLHImages = getImages("WLH/*.jpg");
vector<int> goodMatch1, goodMatch2;

int main()
{
    vector<Mat> matSource;
    
    matSource = loadMat("office"); // Choices: church (StJames), office, and wlh.
    imgMatch(matSource, 75);
    matSource = loadMat("church"); // Choices: church (StJames), office, and wlh.
    imgMatch(matSource, 30);
    matSource = loadMat("wlh"); // Choices: church (StJames), office, and wlh.
    imgMatch(matSource, 25);
    imgMatchStitch(matSource);
    return 0;
}

// This function retrieves all images from a folder and returns them in a vector of strings of the file names.
vector<String> getImages(String path)
{
    String filePath = path;
    vector<String> listOfImages;
    glob(filePath, listOfImages, false);
    return listOfImages;
}

vector<Mat> loadImages(vector<String> list)
{
    vector<Mat> matImg;
    for (int i = 0; i < list.size(); i++)
    {
        Mat img = imread(list[i]);
        matImg.push_back(img);
    }
    return matImg;
}

// Load the necessary matrices into the code
vector<Mat> loadMat(String identify)
{
    vector<Mat> matSource;
    
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
    else if(identify == "wlh")
    {
        matSource = loadImages(listOfWLHImages);
        cout << "WLH selected." << endl;
    }
    
    return matSource;
}

void imgMatch(vector<Mat> matSource, int matchSensitivity)
{
    // Variable declaration
    Mat img1, img2, descriptors1, descriptors2, matchesMatrix;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches, filteredMatches;
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorMatcher> matcher;
    int counter;
    
    // Variables for Feature Detector
    int nfeatures = 2000;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    int patchSize = 31;
    
    cout << "Total images to match: " << matSource.size() << endl;
    
    for (int i = 0; i< matSource.size()-1; i++)
    {
        for (int j = i+1; j < matSource.size(); j++)
        {
            counter = 0;
            
            // Import images to compare
            img1 = matSource[i];
            img2 = matSource[j];
            
            // Feature Detector
            detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, ORB::HARRIS_SCORE, patchSize);
            detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
            detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);
            
            // Match features
            matcher = DescriptorMatcher::create("BruteForce-Hamming");
            matcher->match(descriptors1, descriptors2, matches, Mat());
            
            // Sort by distance
            filteredMatches = matches;
            for (int k = 0; k < filteredMatches.size(); k++)
            {
                for (int l = 0; l < filteredMatches.size() - 1; l++)
                {
                    if (filteredMatches[l].distance > filteredMatches[l + 1].distance)
                    {
                        auto temp = filteredMatches[l];
                        filteredMatches[l] = filteredMatches[l + 1];
                        filteredMatches[l + 1] = temp;
                    }
                }
            }
            
            // Set distance to 20 for a good match to occur
            for (int k = 0; k < filteredMatches.size(); k ++)
            {
                if(filteredMatches[k].distance < 21) counter++;
            }
            
            filteredMatches.erase(filteredMatches.begin() + counter, filteredMatches.end());
            
            if(counter > matchSensitivity)
            {
                goodMatch1.push_back(i); // Recording good matches
                goodMatch2.push_back(j); // Recording good matches
                cout << "Image " << i << " and Image " << j << " : GOOD MATCH (" << counter << " entries)\n";
                
                drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches, matchesMatrix);
                imshow("Matches", matchesMatrix);
                waitKey();
                
                // Call Transformation function so re-calculation of matches is not necessary.
                imgTransform(img1, img2, filteredMatches, keypoints1, keypoints2);
            }
            else
            {
                cout << "Image " << i << " and Image " << j << " : BAD MATCH (" << counter << " entries)\n";
                drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches, matchesMatrix);
                imshow("Matches", matchesMatrix);
                waitKey();
            }
        }
        cout << "Image " << i << " finished comparison." << endl;
    }
    cout << "MATCHING: DONE" << endl;
}

Mat imgTransform(Mat img1, Mat img2, vector<DMatch> matches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2)
{
    Mat h, resultImg;
    vector<Point2f> img1Points, img2Points;
    
    for (size_t i = 0; i < matches.size(); i++)
    {
        img1Points.push_back(keypoints1[matches[i].queryIdx].pt);
        img2Points.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    
    h = findHomography(img2Points, img1Points, RANSAC); // Outputs 64F... double matrix type
    
    int height_img1 = img1.rows;
    int height_img2 = img2.rows; // Unused but would be used if a more efficient algorithm is developed.
    int width_img1 = img1.cols;
    int width_img2 = img2.cols;
    int height_panorama = height_img1;// + height_img2;
    int width_panorama = width_img1 + width_img2;
    
    warpPerspective(img2, resultImg, h, Size(width_panorama, height_panorama), 1);
    imshow("Perspective change", resultImg);
    waitKey();
    
    return h;
}

// Image matching and stitching is done in this function (Step 3 only... see documentation)
void imgMatchStitch(vector<Mat> matSource)
{
    // Variables for Feature Detector
    int nfeatures = 5000;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    int patchSize = 31;
    
    // All other variable declarations
    Mat img1, img2, descriptors1, descriptors2, h, matchesMatrix, resultImg, pano;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<Point2f> img1Points, img2Points;
    vector<DMatch> matches12, matches21, filteredMatches12;
    vector<Mat> transformedImg, panoImg, hHistory;
    vector<Point> pt_orig, pt_transform;
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorMatcher> matcher;
    int first = 0;
    int j = 1;
    int img2_width = 0;
    int img2_height = 0;
    int cropwidth;
    
    cout << "Total images to match: " << matSource.size() << endl;
    
    for (j = 1; j < matSource.size(); j++)
    {
        if(first == 0)
        {
            img1 = matSource[j-1];
            img2 = matSource[j];
            first = 1;
            cout << "Feature detection for images " << j-1 << " and " << j << " (Image Index #)." << endl;
        }
        else
        {
            img1 = panoImg[j-2];
            img2 = matSource[j];
            cout << "Feature detection for previously stitched panorama image and image # (index): " << j << endl;
        }
        // Feature Detector
        detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, ORB::HARRIS_SCORE, patchSize);
        detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
        detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);
        
        // Match features
        matcher = DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(descriptors1, descriptors2, matches12, Mat());
        matcher->match(descriptors2, descriptors1, matches21, Mat());
        
        // Sort matches to get good matches (sort by distance)
        filteredMatches12 = matches12;
        for (int i = 0; i < filteredMatches12.size(); i++)
        {
            for (int j = 0; j < filteredMatches12.size() - 1; j++)
            {
                if (filteredMatches12[j].distance > filteredMatches12[j + 1].distance)
                {
                    auto temp = filteredMatches12[j];
                    filteredMatches12[j] = filteredMatches12[j + 1];
                    filteredMatches12[j + 1] = temp;
                }
            }
        }
        
        // Filter to take first 20 matches. Return error if less 4 matches are found
        if (filteredMatches12.size() > 20)
        {
            filteredMatches12.resize(20);
        }
        else if(filteredMatches12.size() < 4)
        {
            cout << "Not enough matches! Exit program." << endl;
        }
        
        // Printing all the filtered matches
        for (int i = 0; i < filteredMatches12.size(); i++)
        {
            cout << "Match #" << i+1 << " has distance: " << filteredMatches12[i].distance << endl;
        }
        
        
        for (size_t i = 0; i < filteredMatches12.size(); i++)
        {
            img1Points.push_back(keypoints1[filteredMatches12[i].queryIdx].pt);
            img2Points.push_back(keypoints2[filteredMatches12[i].trainIdx].pt);
        }
        
        // Find homography (source pts, dst pts, algorithm)
        h = findHomography(img2Points, img1Points, RANSAC); // Outputs 64F... double matrix type
        
        // Initializing vector of Point
        for (int i = 0; i < 4; i++)
        {
            Point temp;
            temp.x = 0;
            temp.y = 0;
            pt_orig.push_back(temp);
            pt_transform.push_back(temp);
        }
        
        // I am attempting to get the new coordinates of the homography-ed image 2.
        // Width is x, height is y
        pt_orig[0].x = 0; // This is upper left
        pt_orig[0].y = 0;
        pt_orig[1].x = img2.cols; // This is upper right
        pt_orig[1].y = 0;
        pt_orig[2].x = img2.cols; // This is lower right
        pt_orig[2].y = img2.rows;
        pt_orig[3].x = 0; // This is lower left
        pt_orig[3].y = img2.rows;
        
        for (int i = 0; i < 4; i++)
        {
            double a, b, c;
            a = pt_orig[i].x * h.at<double>(0,0);
            b = pt_orig[i].y * h.at<double>(0,1);
            c = h.at<double>(0,2);
            // cout << "a: " << a << " b: " << b << " c: " << c << endl;
            pt_transform[i].x = a + b + c;
            
            a = pt_orig[i].x * h.at<double>(1,0);
            b = pt_orig[i].y * h.at<double>(1,1);
            c = h.at<double>(1,2);
            // cout << "a: " << a << " b: " << b << " c: " << c << endl;
            pt_transform[i].y = a + b + c;
            
            a = pt_orig[i].x * h.at<double>(2,0);
            b = pt_orig[i].y * h.at<double>(2,1);
            c = h.at<double>(2,2);
            double result = a + b + c;
            
            pt_transform[i].x = pt_transform[i].x / result;
            pt_transform[i].y = pt_transform[i].y / result;
            
            cout << "Transformed point " << i+1 << ": " << pt_transform[i] << endl;
        }
        
        // Padding: This is meant to be used for WLH and StJames
        padding(pt_transform, img2_width, img2_height);
        
        // Getting the width where the image should be cropped.
        if(pt_transform[1].x > pt_transform[2].x)
        {
            cropwidth = pt_transform[2].x;
        }
        else{
            cropwidth = pt_transform[1].x;
        }
        
        // Trying to store history of homography and use it on the images after this one.
        // Homography history reading is not fully developed. This is left here to show what I have done.
        hHistory.push_back(h); cout << "Homography stored." << endl;
        
        // Panorama length needed.
        int height_img1 = img1.rows;
        int height_img2 = img2.rows; // Unused but would be used if a more efficient algorithm is developed.
        int width_img1 = img1.cols;
        int width_img2 = img2.cols;
        int height_panorama = height_img1;// + height_img2;
        int width_panorama = width_img1 + width_img2;
        
        // Draw Matches
        drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches12, matchesMatrix);
        imshow("Matches", matchesMatrix);
        waitKey();
 
        // Warp image according to the homography
        warpPerspective(img2, resultImg, h, Size(width_panorama, height_panorama), 1);
        imshow("Perspective change", resultImg);
        waitKey();
        
        // Origianlly, the transformed image was going to be used to stitch images future images together.
        // This would work with inverse homography. However, not enough time is given to finish development.
        transformedImg.push_back(resultImg);
        cout << "Image transformed and placed in matrix for images " <<  j << " and " << j+1 << "."  << endl;
        
        for (int r = 0; r < img1.rows; r++)
        {
            for (int c = 0; c < img1.cols; c++)
            {
                resultImg.at<Vec3b>(r,c)[0] = img1.at<Vec3b>(r,c)[0];
                resultImg.at<Vec3b>(r,c)[1] = img1.at<Vec3b>(r,c)[1];
                resultImg.at<Vec3b>(r,c)[2] = img1.at<Vec3b>(r,c)[2];
            }
        }
        imshow("Result Image with Image 1", resultImg);
        waitKey();
        
        // Matrix with the length of the cropped image.
        pano = Mat(height_panorama, cropwidth, CV_8UC3);
        for (int r = 0; r < height_panorama; r++)
        {
            for (int c = 0; c < cropwidth; c++)
            {
                pano.at<Vec3b>(r,c)[0] = resultImg.at<Vec3b>(r,c)[0];
                pano.at<Vec3b>(r,c)[1] = resultImg.at<Vec3b>(r,c)[1];
                pano.at<Vec3b>(r,c)[2] = resultImg.at<Vec3b>(r,c)[2];
            }
        }
        
        imshow("Panorama", pano);
        waitKey();
        panoImg.push_back(pano);
        
        // Clear all variables
        matches12.clear(); matches21.clear();
        keypoints1.clear(); keypoints2.clear();
        img1Points.clear(); img2Points.clear();
        filteredMatches12.clear();
        descriptors1.release(); descriptors2.release();
        h.release();
        matchesMatrix.release();
        resultImg.release();
        pano.release();
        detector->clear();
        matcher->clear();
        
    } // end for loop
    
    // Update matSource...
    matSource.clear();
    for (int i = 0; i < panoImg.size(); i++) matSource.push_back(panoImg[i]);
    cout << "matSource #: " <<  matSource.size() << endl;
    cout << "panoImg #: " << panoImg.size() << endl;
}

// Black padding around images to give placeholders for panorama image.
void padding(vector<Point> pt_transform, int &desiredWidth, int &desiredHeight)
{
    // reminder: x is width, y is height
    int w1, w2, w3, w4;
    int h1, h2, h3, h4;
    vector<int> width, height;
    
    w1 = pt_transform[0].x;
    w2 = pt_transform[1].x;
    w3 = pt_transform[2].x;
    w4 = pt_transform[3].x;
    h1 = pt_transform[0].y;
    h2 = pt_transform[1].y;
    h3 = pt_transform[2].y;
    h4 = pt_transform[3].y;
    
    width.push_back(abs(w1-w2)); width.push_back(abs(w1-w3));
    width.push_back(abs(w1-w4)); width.push_back(abs(w2-w3));
    width.push_back(abs(w2-w4)); width.push_back(abs(w3-w4));
    
    height.push_back(abs(h1-h2)); height.push_back(abs(h1-h3));
    height.push_back(abs(h1-h4)); height.push_back(abs(h2-h3));
    height.push_back(abs(h2-h4)); height.push_back(abs(h3-h4));

    sort(width.begin(), width.end());
    sort(height.begin(), height.end());
    
    desiredWidth = width[5];
    desiredHeight = height[5];
}

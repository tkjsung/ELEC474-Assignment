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
vector<String> getImages(String path);
vector<Mat> loadImages(vector<String> list);
void imgORBMatch(String identify);
void stitching(vector<Mat> inputImg);

// Global Variables
//vector<string> listofImages; // Original plan to do all the file names... ignore forever.
vector<string>listOfChurchImages = getImages("StJames/*.jpg");
vector<string>listOfOfficeImages = getImages("office2/*.jpg");
vector<string>listOfWLHImages = getImages("WLH/*.jpg"); // I may want to manually order the images...
//vector<Mat> matChurch = loadImages(listOfChurchImages); // loading matrices.
//vector<Mat> matOffice = loadImages(listOfOfficeImages); // loading matrices.
//vector<Mat> matwlh = loadImages(listOfWLHImages); // loading matrices
Mat pano;

int main()
{

//    vector<Mat> matChurch = loadImages(listOfChurchImages);
//    cout << matChurch[0] << endl;
//    imshow("test",matChurch[2]);
//    waitKey();
    
    //TODO:: Determine which image to take as the base image
    //TODO:: Make sure stitching tool works
    //TODO:: Keep track of image indices so when one image is placed on another image we know where it goes
    
    imgORBMatch("office"); // Choices: wlh, church, or office
//    stitching(matOffice);
    
    return 0;
}

// Reference: https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/Image_Stitching.py
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
    
    // Variables for Feature Detector
    int nfeatures = 5000;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    int patchSize = 31;
    
    // All other variable declarations
    Mat img1, img2, descriptors1, descriptors2, h, matchesMatrix, resultImg;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<Point2f> img1Points, img2Points;
    vector<DMatch> matches12, matches21, filteredMatches12, filteredMatches21;
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorMatcher> matcher;
    int first = 0;;
    int j = 1; // for loop counter
    vector<Mat> transformedImg;
    
//    cout << matSource.size() << endl;
    transformedImg.push_back(matSource[0]); // Put one image in the vector<Mat>, acts as base image
    
    cout << "Total images to match: " << matSource.size() << endl;
    
//    for (j = 0; j < matSource.size(); j++)
    for (j = 0; j < 3; j++)
    {

//        if (matSource.size() < j + 2) break;
        if(first == 0){
            img1 = matSource[j];
            img2 = matSource[j+1];
            cout << "Feature detection for images " << j << " and " << j+1 << " (Image Index #)." << endl;
            j++;
            first = 1;
        }
        else
        {
            img1 = pano;
            img2 = matSource[j];
            cout << "Feature detection for panorama image and image " << j << " (Image Index #)." << endl;
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
        
        // Sort matches to get good matches
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
        
        if (filteredMatches12.size() > 20)
        {
            filteredMatches12.resize(20);
        }
        
        
        // same thing as above...?
//        sort(matches12.begin(), matches12.end());
//        // Remove not good matches
//        const int numGoodMatches = matches12.size() * 0.15f;
//        matches12.erase(matches12.begin() + numGoodMatches, matches12.end());

    
        for (size_t i = 0; i < filteredMatches12.size(); i++)
        {
            img1Points.push_back(keypoints1[filteredMatches12[i].queryIdx].pt);
            img2Points.push_back(keypoints2[filteredMatches12[i].trainIdx].pt);
        }
        
        // Find homography (source pts, dst pts, algorithm)
        h = findHomography(img2Points, img1Points, RANSAC); // Outputs 64F... double matrix type

        // Trying to do warpPerspective manipulation but didn't really work so it's getting commented out
        
        vector<Point> pt_orig, pt_transform;
        for (int i = 0; i < 4; i++)
        {
            Point temp;
            temp.x = 0;
            temp.y = 0;
            pt_orig.push_back(temp);
            pt_transform.push_back(temp);
        }
        
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
//            cout << "a: " << a << " b: " << b << " c: " << c << endl;
            pt_transform[i].x = a + b + c;
            
            a = pt_orig[i].x * h.at<double>(1,0);
            b = pt_orig[i].y * h.at<double>(1,1);
            c = h.at<double>(1,2);
//            cout << "a: " << a << " b: " << b << " c: " << c << endl;
            pt_transform[i].y = a + b + c;
            
            cout << "Transformed point " << i+1 << ": " << pt_transform[i] << endl;
        }
        
        int cropwidth;
        if(pt_transform[1].x > pt_transform[2].x)
        {
            cropwidth = pt_transform[2].x;
        }
        else{
            cropwidth = pt_transform[1].x;
        }
        
        
        // Draw Matches algorithm
        drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches12, matchesMatrix);
        imshow("Matches",matchesMatrix);
        waitKey();
        
        // Panorama length needed...
        int height_img1 = img1.rows;
        int width_img1 = img1.cols;
        int width_img2 = img2.cols;
        int height_panorama = height_img1;
        int width_panorama = width_img1 + width_img2;
        
        
        // Warp image according to the homography
        warpPerspective(img2, resultImg, h, Size(width_panorama, height_panorama), 1);
        
        imshow("Perspective change",resultImg);
//        imwrite("test.jpg", pano);
        waitKey();
        transformedImg.push_back(resultImg);
        cout << "Image transformed and placed in matrix for images " <<  j << " and " << j+1 << "."  << endl;
        img1.copyTo(resultImg);
        
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
        
//        pano = resultImg(height_panorama, cropwidth, CV_8UC3);
        
        imshow("Panorama", pano);
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
//        pano.release();
        matches12.clear();
        matches21.clear();
        filteredMatches12.clear();
        filteredMatches21.clear();
//        half.release();
        detector->clear();
        matcher->clear();
        
        
    } // end for loop
    cout << "Feature detection complete. Now STITCHING." << endl;
    cout << "Stitching code not developed yet. Exit program." << endl;
    
//    stitching(transformedImg);
    
//    for (int i = 0; i < transformedImg.size(); i++)
//    {
//        imshow("Transformed Images", transformedImg[i]);
//        waitKey(2000);
//        cout << "Image #: " << i+1 << endl;
//    }
    
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

// This stitching function uses the high level Stitcher class...
void stitching(vector<Mat> inputImg)
{
    // Referenced from: https://docs.opencv.org/3.4/d8/d19/tutorial_stitcher.html
    
    vector<Mat> matSource = inputImg;
    Mat panorama;
    Stitcher::Mode mode = Stitcher::PANORAMA;
    Ptr<Stitcher>stitcher = Stitcher::create(mode);
    Stitcher::Status status = stitcher->stitch(matSource, panorama);
    
    if(status != Stitcher::OK)
    {
        cout << "Cannot stitch images" << endl;
        return;
    }
    
    imshow("Pano image",panorama);
    waitKey();
    
}

/*
Mat mask(Mat img1, Mat img2, String version, int smoothing_window_size)
{
    int height_img1 = img1.rows;
    int width_img1 = img1.cols;
    int width_img2 = img2.cols;
    int height_panorama = height_img1;
    int width_panorama = width_img1 + width_img2;
    int offset = smoothing_window_size/2;
    int barrier = img1.cols - smoothing_window_size/2;
    Mat mask = Mat::zeros(height_panorama, width_panorama, CV_8UC3);
    
    if (version == "left_image")
    {
        for(int r = 0; r < mask.rows; r++)
        {
            for (int c = (barrier-offset); c < (barrier+offset); c++)
            {
                mask.at<unsigned char>(r,c) =
            }
        }
    }
    int spaces = (2*offset)-1;
    

     height_img1 = img1.shape[0]
     width_img1 = img1.shape[1]
     width_img2 = img2.shape[1]
     height_panorama = height_img1
     width_panorama = width_img1 +width_img2
     offset = int(self.smoothing_window_size / 2)
     barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
     mask = np.zeros((height_panorama, width_panorama))
     if version== 'left_image':
         mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
         mask[:, :barrier - offset] = 1
     else:
         mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
         mask[:, barrier + offset:] = 1
     return cv2.merge([mask, mask, mask])
     
    
    
    return;
}
*/

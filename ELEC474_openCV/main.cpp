// This is the ELEC 474 final course project/take home exam. Steven Crosby (#20011059)
 
// C++ Standard Libraries
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
 
// OpenCV Import
#include <opencv2/opencv.hpp>
 
// Use OpenCV namespace
using namespace cv;
using namespace std;
 
// Create a struct to hold indices of two well-matched (overlapping) images, & another to hold the overlapped file names
struct Pair
{
    int id1, id2;
};
 
// Function declarations
vector<String> getImages(String path);
vector<Pair> pickOverlap(vector<String> listOfImages, int multiplier, int thresh, int inlierThresh, int maxMatches);
int pickBase(vector<String> listOfImages, vector<Pair> overlapping);
void panorama(vector<String> listOfImages, vector<Pair> overlapping, int base, int padding_top, int padding_side, int n);
void test(String image);
 
int main()
{
    vector<String> listOfImages;
    vector<Pair> overlapped;
    int baseIdx;
 
    listOfImages = getImages("office2/*.jpg");
    overlapped = pickOverlap(listOfImages, 3, 130, 40, 10);
    baseIdx = pickBase(listOfImages, overlapped);
    panorama(listOfImages, overlapped, baseIdx, 500, 3000, 2000);
 
    //test("office2/a.jpg");
 
    //overlapped = pickOverlap_fast(listOfImages, 2, 130, 40, 3, 1400);
    //baseIdx = pickBase_fast(listOfImages, overlapped);
    //panorama_fast(listOfImages, overlapped, baseIdx, 1400, 5, 2000);
}
 
// This function retrieves all images from a folder and returns them in a vector of strings of the file names.
vector<String> getImages(String path)
{
    String filePath = path;
    vector<String> listOfImages;
    glob(filePath, listOfImages, false);
 
    //for (int i = 0; i < listOfImages.size(); i++) cout << "index " << i << ", file " << listOfImages[i] << endl;
   
    return listOfImages;
}
 
// This function selects the images that overlap & returns a vector of pairs that contain the overlapping indices
vector<Pair> pickOverlap(vector<String> listOfImages, int multiplier, int thresh, int inlierThresh, int maxMatches)
{
    int numRand = (listOfImages.size()) * multiplier;
    RNG rng((uint64)-1);
 
    vector<Pair> goodPairs;
    vector<int> hisList1, hisList2, matchedList; // History Lists (of all checked)
 
    for (int j = 0; j < listOfImages.size(); j++) matchedList.push_back(0);
 
    for (int i = 0; i < numRand; i++)
    {
        // Randomly select 2 images from the folder
        int idx1 = (int)rng.uniform(0, (int)listOfImages.size());
        int idx2 = (int)rng.uniform(0, (int)listOfImages.size());
        if (idx1 == idx2) continue;
 
        // Check if this set of 2 images has already been checked, and if so, skip
        int flag = 0;
        int flag1 = 0;
        int flag2 = 0;
        for (int j = 0; j < hisList1.size(); j++)
        {
            if (((hisList1[j] == idx1) && (hisList2[j] == idx2)) | ((hisList2[j] == idx1) && (hisList1[j] == idx2)))
            {
                flag = 1;
                //cout << "History repeated " << idx1 << ", " << idx2 << endl;
                break;
            }
        }
 
        if (flag == 1) continue;
        if ((flag1 == 1) & (flag2 == 1))
        {
            cout << listOfImages[idx1] << " & " << listOfImages[idx2] << " already matched " << maxMatches << " times. Moving on..." << endl;
            continue;
        }
        if (matchedList[idx1] == maxMatches) flag1 = 1;
        if (matchedList[idx2] == maxMatches) flag2 = 1;
 
        // Put the indices of the current images onto the lists to be checked for repetition in further loops
        hisList1.push_back(idx1);
        hisList2.push_back(idx2);
 
        // Retrieve and resize the images
        Mat image1 = imread(listOfImages[idx1]);
        Mat image2 = imread(listOfImages[idx2]);
        resize(image1, image1, Size(), 0.25, 0.25, INTER_NEAREST);
        resize(image2, image2, Size(), 0.25, 0.25, INTER_NEAREST);
 
        // Detect matches in the two images
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
 
        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        int patchSize = 31;
 
        // Feature detector
        Ptr<FeatureDetector> detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, ORB::HARRIS_SCORE, patchSize);
        detector->detectAndCompute(image1, Mat(), keypoints1, descriptors1);
        detector->detectAndCompute(image2, Mat(), keypoints2, descriptors2);
 
        // Match features
        vector<DMatch> matches;
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(descriptors1, descriptors2, matches, Mat());
 
        // Sort matches by score
        sort(matches.begin(), matches.end());
 
        // Remove not good matches
        const int numGoodMatches = matches.size() * 0.1f;
        matches.erase(matches.begin() + numGoodMatches, matches.end());
 
        vector<Point2f> kp1List, kp2List, kp2Trans;
        for (int q = 0; q < matches.size(); q++)
        {
            kp1List.push_back(keypoints1[matches[q].queryIdx].pt);
            kp2List.push_back(keypoints2[matches[q].trainIdx].pt);
        }
 
        Mat imgTransed, h;
 
        // Find homography
        h = findHomography(kp2List, kp1List, RANSAC);
        perspectiveTransform(kp2List, kp2Trans, h);
 
        int inliers = 0;
 
        for (int q = 0; q < kp1List.size(); q++)
        {
            float piecex, piecey, diff;
            int kp1x = kp1List[q].x;
            int kp2x = kp2Trans[q].x;
            int kp1y = kp1List[q].y;
            int kp2y = kp2Trans[q].y;
           
            piecex = pow((abs(kp1x - kp2x)), 2);
            piecey = pow((abs(kp1y - kp2y)), 2);
            diff = sqrt(piecex + piecey);
 
            if (diff <= thresh) inliers++;
        }
 
        if (inliers >= inlierThresh)
        {
            cout << listOfImages[idx1] << " & " << listOfImages[idx2] << " OVERLAP w/ inliers: " << inliers << endl;
            Pair current;
            current.id1 = idx1;
            current.id2 = idx2;
            goodPairs.push_back(current);
            matchedList.at(idx1) = matchedList[idx1] + 1;
            matchedList.at(idx2) = matchedList[idx2] + 1;
        }
        else cout << listOfImages[idx1] << " & " << listOfImages[idx2] << " **DO NOT** overlap. Inliers: " << inliers << endl;
 
        // Make a vector, goodPairs, of pairs of corresponding well-matched images
    }
 
    return goodPairs;
}
 
// This looks at the overlapping pairs and finds the most common image, and returns its index as a 'base' in the panorama
int pickBase(vector<String> listOfImages, vector<Pair> overlapping)
{
    int countMax = 0;
    int base = -1;
    for (int i = 0; i < listOfImages.size(); i++)
    {
        int count = 0;
        for (int j = 0; j < overlapping.size(); j++)
        {
            if ((overlapping[j].id1 == i) | (overlapping[j].id2 == i)) count++;
        }
        //cout << i << " overlapped w/ " << count << " other images" << endl;
        if (count > countMax)
        {
            base = i;
            countMax = count;
        }
    }
    cout << "Base is image w/ index " << base << endl;
    return base;
}
 
// This stitches images previously deemed 'good matches' together to create the panorama
void panorama(vector<String> listOfImages, vector<Pair> overlapping, int base, int padding_top, int padding_side, int n)
{
    Mat image1 = imread(listOfImages[base]); //image1 is the CURRENT BASE; "base" to start, and then the current pano for future iterations
    Mat image2;
    vector<int> alreadyDone;
    alreadyDone.push_back(base);
    int lastAdded = base;
    resize(image1, image1, Size(), 0.25, 0.25, INTER_NEAREST);
    copyMakeBorder(image1, image1, padding_top, padding_top, padding_side, padding_side, BORDER_CONSTANT, Scalar(0));
    Mat next_base = image1.clone();
    int out_of_options = 1;
    int nowAdd;
    int keepGoing = 0;
    //int hisIdx[20];
    array<int, 30> hisIdx;
    vector<Mat> history;
    for (int j = 0; j < listOfImages.size(); j++)
    {
        Mat temp;
        history.push_back(temp);
        hisIdx[j] = 0;
    }
    int q = 0;
    int ecount = 0;
 
    hisIdx[base] = 1;
    history.at(base) = image1;
 
    //for (int q = 0; q < listOfImages.size(); q++)
    while (q < (listOfImages.size() - 1))
    //for (int q = 0; q < 3; q++)
    {
        Mat img2Transed;
        keepGoing = 0;
        cout << "Looking for image to stitch to " << lastAdded << endl;
        for (int r = 0; r < overlapping.size(); r++)
        {
            if (overlapping[r].id1 == lastAdded)
            {
                nowAdd = overlapping[r].id2;
                keepGoing = 1;
                cout << "Found image to stitch: " << nowAdd << endl;
                //alreadyDone.push_back(nowAdd);
                overlapping.erase(overlapping.begin() + r);
                ecount = 0;
                break;
            }
            else if (overlapping[r].id2 == lastAdded)
            {
                nowAdd = overlapping[r].id1;
                keepGoing = 1;
                cout << "Found image to stitch: " << nowAdd << endl;
                //alreadyDone.push_back(nowAdd);
                overlapping.erase(overlapping.begin() + r);
                ecount = 0;
                break;
            }
        }
        if (keepGoing == 0)
        {
            if (ecount == 0)
            {
                cout << "Found no image to stitch! Try with base, " << base << endl;
                nowAdd = base;
                ecount++;
            }
            else
            {
                cout << "Still found no image to stitch! Ending program." << endl;
                break;
            }
            /*if (ecount == max_ecount)
            {
                cout << "Too many failed tries. Ending program" << endl;
                break;
            }*/
        }
        if (hisIdx[nowAdd] != 0)
        {
            cout << "Image " << nowAdd << " already stitched" << endl;
            img2Transed = history[nowAdd];
        }
        else
        {
            image2 = imread(listOfImages[nowAdd]);
            resize(image2, image2, Size(), 0.25, 0.25, INTER_NEAREST);
            copyMakeBorder(image2, image2, padding_top, padding_top, padding_side, padding_side, BORDER_CONSTANT, Scalar(0));
 
            // Detect matches in the two images
            vector<KeyPoint> keypoints1, keypoints2;
            Mat descriptors1, descriptors2;
 
            int nfeatures = n;
            float scaleFactor = 1.2f;
            int nlevels = 8;
            int edgeThreshold = 31;
            int firstLevel = 0;
            int WTA_K = 2;
            int patchSize = 31;
 
            // Feature detector
            Ptr<FeatureDetector> detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, ORB::HARRIS_SCORE, patchSize);
            detector->detectAndCompute(next_base, Mat(), keypoints1, descriptors1);
            detector->detectAndCompute(image2, Mat(), keypoints2, descriptors2);
 
            // Match features
            vector<DMatch> matches;
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
            matcher->match(descriptors1, descriptors2, matches, Mat());
 
            // Sort matches by score
            sort(matches.begin(), matches.end());
 
            // Remove not good matches
            const int numGoodMatches = matches.size() * 0.15f;
            matches.erase(matches.begin() + numGoodMatches, matches.end());
 
            vector<Point2f> pointsBase, pointsTrans;
 
            for (size_t i = 0; i < matches.size(); i++)
            {
                pointsBase.push_back(keypoints1[matches[i].queryIdx].pt);
                pointsTrans.push_back(keypoints2[matches[i].trainIdx].pt);
            }
 
 
            Mat img1Transed, h;
 
            // Find homography
            h = findHomography(pointsTrans, pointsBase, RANSAC);
            // Use homography to warp image
            //warpPerspective(image2, img2Transed, h, img2Transed.size(), 1, 0, 0.1);
 
            if (h.at<double>(2,1) < 0)
            {
                h = estimateAffine2D(pointsTrans, pointsBase);
                warpAffine(image2, img2Transed, h, img2Transed.size(), 1, 0, 0.1);
            }
            else
                warpPerspective(image2, img2Transed, h, img2Transed.size(), 1, 0, 0.1);
                //cout << h << endl;
 
            Mat imgPan;
 
            for (int r = 0; r < image1.rows; r++)
            {
                for (int c = 0; c < image1.cols; c++)
                {
                    if ((image1.at<Vec3b>(r, c)[0] == 0) & (image1.at<Vec3b>(r, c)[1] == 0) & (image1.at<Vec3b>(r, c)[2] == 0))
                    {
                        image1.at<Vec3b>(r, c)[0] = img2Transed.at<Vec3b>(r, c)[0];
                        image1.at<Vec3b>(r, c)[1] = img2Transed.at<Vec3b>(r, c)[1];
                        image1.at<Vec3b>(r, c)[2] = img2Transed.at<Vec3b>(r, c)[2];
                    }
                    /*else
                    {
                        image1.at<Vec3b>(r, c)[0] = (image1.at<Vec3b>(r, c)[0] + img2Transed.at<Vec3b>(r, c)[0]) / 2;
                        image1.at<Vec3b>(r, c)[1] = (image1.at<Vec3b>(r, c)[1] + img2Transed.at<Vec3b>(r, c)[1]) / 2;
                        image1.at<Vec3b>(r, c)[2] = (image1.at<Vec3b>(r, c)[2] + img2Transed.at<Vec3b>(r, c)[2]) / 2;
                    }*/
                }
            }
            hisIdx[nowAdd] = 1;
            history.at(nowAdd) = img2Transed;
            q++;
        }
        lastAdded = nowAdd;
        next_base = img2Transed.clone();
 
        /*namedWindow("Stitching", WINDOW_KEEPRATIO);
        imshow("Stitching", image1);
        cout << "Loaded new image" << endl;
        waitKey();*/
    }
 
    namedWindow("Panorama", WINDOW_KEEPRATIO);
    imshow("Panorama", image1);
    waitKey();
 
    history.clear();
    listOfImages.clear();
    overlapping.clear();
    alreadyDone.clear();
}
 
void test(String image)
{
    Mat image1 = imread(image);
    resize(image1, image1, Size(), 0.25, 0.25, INTER_NEAREST);
    Mat gray;
    cvtColor(image1, gray, COLOR_BGR2GRAY);
    Mat corners = gray.clone();
    cornerHarris(gray, corners, 5, 5, 0.05, BORDER_DEFAULT);
    imshow("Corners", corners);
    waitKey();
}

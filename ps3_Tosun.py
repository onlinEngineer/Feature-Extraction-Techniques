#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:03:13 2018

@author: bahadir
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2 

raw_h1to2p=np.array([[8.7976964e-01 ,  3.1245438e-01,  -3.9430589e+01,],[-1.8389418e-01 ,  9.3847198e-01 ,  1.5315784e+02],[1.9641425e-04 , -1.6015275e-05,   1.0000000e+00]])

print("Given Homography Matrix",raw_h1to2p)

feature_methods=["SIFT","SURF","ORB"]


### 3. FIND homography ####
def findHomography(keyp1,keyp2,match):

        image_1_points = np.zeros((len(match), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(match), 1, 2), dtype=np.float32)

        for i in range(0,len(match)):
            image_1_points[i] = keyp1[match[i].queryIdx].pt
            image_2_points[i] = keyp2[match[i].trainIdx].pt


        homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

        return homography


#### 4. Warp Perspective ####
def warpImage(image1,image2,homography):
    destination_image = cv2.warpPerspective(image1, homography, (image2.shape[0], image2.shape[1]))
    plt.imshow(destination_image)
    plt.show()
    cv2.imwrite("WarpedImage.png",destination_image)


def stitch_images(img1, img2, H):
    # Transform the second image according to the homography matrix
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    img2_warped = cv2.warpPerspective(img2, H, (cols1 + cols2, rows1))

    # Create a mask for the transformed image
    mask = np.zeros((rows1, cols1 + cols2, 3), dtype=np.uint8)
    mask[:, cols1:cols1+cols2, :] = 255

    # Make sure that the input arrays have the same shape and type
    img1 = cv2.resize(img1, (cols1 + cols2, rows1))
    img2_warped = img2_warped.astype(np.float32)

    # Blend the two images together using the mask
    blended = cv2.addWeighted(img1, 0.5, img2_warped, 0.5, 0, dtype=cv2.CV_32F)
    blended = blended.astype(np.uint8)
    blended = cv2.bitwise_and(blended, mask)



    return blended



for i in range(len(feature_methods)):

    
    # Read the images
    img1 = cv2.imread('PS3_data/input/img1.ppm')
    img2 = cv2.imread('PS3_data/input/img2.ppm')
  

    # Convert the images to grayscale
    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    

    if  feature_methods[i]=="SIFT":
        print('Calculating SIFT features...')
        
        # Create a SIFT object
        sift = cv2.xfeatures2d.SIFT_create()

        # Get the keypoints and the descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1_gray,None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2_gray,None)
        # keypoints object includes position, size, angle, etc.
        # descriptors is an array. For sift, each row is a 128-length feature vector

    elif feature_methods[i]=="SURF":
        print('Calculating SURF features...')
        surf = cv2.xfeatures2d.SURF_create(4000)
        keypoints1, descriptors1 = surf.detectAndCompute(img1_gray,None)
        keypoints2, descriptors2 = surf.detectAndCompute(img2_gray,None)

    elif feature_methods[i]=="ORB":
        print('Calculating ORB features...')
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(img1_gray,None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2_gray,None)     
        # Note: Try cv2.NORM_HAMMING for this feature
        

    # Draw the keypoints
    img1 = cv2.drawKeypoints(image=img1, outImage=img1, keypoints=keypoints1, 
                            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                            color = (0, 0, 255))

    img2 = cv2.drawKeypoints(image=img2, outImage=img2, keypoints=keypoints2, 
                            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                            color = (0, 0, 255))


    # Display the images
    cv2.imshow(f'Keypoints 1 for  {feature_methods[i]}', img1)
    cv2.imwrite(f"Keypoints 1 {feature_methods[i]}.png",img1)
    cv2.imshow(f'Keypoints 2 for  {feature_methods[i]}', img2)
    cv2.imwrite(f"Keypoints 2 {feature_methods[i]}.png",img2)
    
    if feature_methods[i]=="ORB":
    # Create a brute-force descriptor matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) 
    # Different distances can be used.
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) 

    # Match keypoints
    matches1to2 = bf.match(descriptors1,descriptors2)
    # matches1to2 is a DMatch object
    # LOOK AT OPENcv2DOCUMENTATION AND 
    #   LEARN ABOUT THE DMatch OBJECT AND ITS FIELDS, SPECIFICALLY THE STRENGTH OF MATCH
    #   matches1to2[0].distance

    # Sort according to distance and display the first 40 matches
    matches1to2 = sorted(matches1to2, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches1to2[:40],img2,flags=2)
    plt.imshow(img3)
    cv2.imwrite(feature_methods[i]+".png",img3)
    plt.title(f"Keypoint Matches for {feature_methods[i]}")
    plt.show()


    ##### 1.FIND BEST MATCHES #####
    matches = bf.knnMatch(descriptors1,descriptors2,k=2)
        # Apply ratio test
    ground_truth = []
    for m,n in matches:
            if m.distance < 0.75*n.distance:
                ground_truth.append([m])
        # cv2drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,keypoints1,img2,keypoints2,ground_truth,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.title(f"Best Matches for {feature_methods[i]}")
    plt.imshow(img3),plt.show()
    cv2.imwrite(f"Best Matches {img3}.png",img3)


# Calculate the repeatability rate
    repeatability = len(ground_truth) / (len(keypoints2) + len(keypoints2)) / 2
   

    # Print the repeatability rate
    print("Repeatability: {:.2f}%".format(repeatability * 100))


    ##### 3.FIND HOMOGRAPHY #####
    homography=findHomography(keypoints1,keypoints2,matches1to2)
    print(f"Homography Matrix for {feature_methods[i]}",homography)





#### 4. WRAP IMAGE ONTO OTHER IMAGE ####
warpImage(img1,img2,raw_h1to2p)





# Load the images and the homography matrix
img1 = cv2.imread('PS3_data/input/img1.ppm')
img2 = cv2.imread('PS3_data/input/img2.ppm')


# Stitch the images together
stitched_img = stitch_images(img2, img1, raw_h1to2p)

stitcher = cv2.createStitcher()
status, result = stitcher.stitch([img1,img2])

# Display the stitched image
cv2.imshow('Stitched Image1', stitched_img)
cv2.imshow('Stitched Image2', stitched_img)

cv2.waitKey(0)

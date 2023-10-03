# Feature-Extraction-Techniques
Feature Extraction Techniques such as ORB, SIFT, SURF

# Find the best matches.
## Best matches for SIFT
### Keypoints for First Image

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/f953e35b-d609-406c-a8ed-d1daf1c1131a" width="50%" height="50%">

### Keypoints for Second Image

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/ec4e4fe9-aa8b-46ca-84ea-bb0cdd017066" width="50%" height="50%">

### Keypoint Matches for SIFT

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/27ce415e-e0f7-4587-b094-af1c0b6d6d7b" width="50%" height="50%">

### Best Matches for SIFT

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/42cae679-d8f0-444d-8c4e-3acdeeed2e1d" width="50%" height="50%">


## Best matches for SURF
### Keypoints for First Image

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/19fff1f6-047c-43f2-a199-e5ec9eaae589" width="50%" height="50%">

### Keypoints for Second Image

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/824b4c9a-80ab-46f7-96ab-0363dda81df5" width="50%" height="50%">


### Keypoint Matches for SURF

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/b2525f43-2f60-4f61-b7e6-684b0ed5961f" width="50%" height="50%">


### Best Matches for SURF

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/58a2a9d3-94aa-42ac-bd7e-93f1632129b6" width="50%" height="50%">


## Best matches for ORB
### Keypoints for First Image

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/fe11fe76-2f09-4f7b-aa66-fa20b02f2806" width="50%" height="50%">


### Keypoints for Second Image

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/4da57260-32b9-45e3-8955-dde1023d02d4" width="50%" height="50%">



### Keypoint Matches for ORB


<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/1f789290-1fd2-4d82-b27a-e327023028f2" width="50%" height="50%">



### Best Matches for ORB

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/4bf164ce-5678-48f5-9ff3-63a4f8433860" width="50%" height="50%">


<br>

# The repeatability rates of the feature detectors

| Detector | Repeatability Rates              |
| :-------- | :------- | 
| `SIFT` | 9.61% |
| `SURF` | 8.47% | 
| `ORB` | 10.20% | 

<br>

## Find Homography by Using RANSAC
For each algorithm, we are using cv2.findHomography function to calculate the homography matrix of given image. The function takes at least 3 parameters that are keypoints of two image and the cv2.RANSAC algorithm.
The matrices are close to each other with the given matrix.

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/d1fc6b07-ca26-487b-a051-715ed2783de9" width="50%" height="50%">

### Wrap Image
We are using cv2.wrapPerspective algorithm to find out the wrapped image. This function takes 3 parameters which are image, homography that we calculate, and width and height of image.

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/52152fd2-a2aa-4aad-998b-98ebd2d55c29" width="50%" height="50%">


### Stitches Image
We are using cv2.createStitcher class to find out the stitched image. After creating an instance from Stitcher class, we are using stitch method. This method takes two parameters that are images.

<img src="https://github.com/onlinEngineer/Feature-Extraction-Techniques/assets/70773825/96c438c5-7881-4d3a-b9c1-ab470ef27e78" width="50%" height="50%">

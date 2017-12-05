# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


<<<<<<< HEAD
<p align="center">
 <a href="https://youtu.be/Rz8r4_qigSY"><img src="https://img.youtube.com/vi/Rz8r4_qigSY/0.jpg" alt="Overview" width="50%" height="50%"></a>
 <br>Click for full video
</p>

---

**Vehicle Detection Project**

This project is currently ongoing. Above is current result using a simple ConvNet as the classifier. And the code is available in this notebook [here](https://github.com/Xfan1025/SDCND-Vehicle-Detection/blob/master/convnet.ipynb)

---

Table of Contents (updating)
=================

   * [Goal](#goal)
   * [Code & Files](#code-and-files)
   * [Pipeline using Machine Learning](#pipeline-using-machine-learning)
      * [Feature Extraction](#feature-extraction)
          * [Histogram of Oriented Gradients](#histogram-of-oriented-gradients)
          * [Spatial Binning of Color](#spatial-binning-of-color)
          * [Histogram of Color](#histogram-of-color)
          * [Combined Features](#combined-features)
      * [Training Classifier](#training-classifier)
          * [Linear SVM](#linear-SVM)
          * [Random Forest](#random-forest)
      * [Car Searching](#car-searching)
          * [Sliding Window Search](#sliding-window-search)
          * [Heatmap Thresholding](#heat-map-thresholding)
      * [Output](#Output)
          * [Test Images](#test-images)
          * [Video](#video)
   * [Pipeline using Deep Learning](#pipeline-using-deep-learning)
      * [Image Preprocessing](#image-preprocessing)
      * [Training Classifier](#training-classifier)
          * [ConvNet](#convnet)
          * [Prediction Visualisation](#prediction-visualisation)
      * [Pipeline](#pipeline)
          * Sliding window
          * Heatmap Thresholding
          * Output
      * [Final Video Output](#final-video-output)
   * [Discussion](#discussion)
---

# Goal

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


---

# Code and Files

You're reading this project! This Project repo contains the following code files:

* [utils.py](./utils.py) The script containing functions to extract features from images.
* [ML.ipynb](./ML.ipynb) The notebook containing exploration on feature extraction and pipeline using Random Forest as Classifier.
* [ConvNet.ipynb](./) The notebook containing pipeline using ConvNet as Classifier.
* [model.h5](./model.h5) The ConvNet model trained for the pipeline.
* [data_extraction.ipynb](./data_extraction.ipynb) The notebook containing exploration and extraction dataset from Udacity.



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
||||||| merged common ancestors
=======
<p align="center">
 <a href="https://youtu.be/Rz8r4_qigSY"><img src="https://img.youtube.com/vi/Rz8r4_qigSY/0.jpg" alt="Overview" width="50%" height="50%"></a>
 <br>Click for full video
</p>

---

**Advanced Lane Finding Project**

This project is currently ongoing. Above is current result using a simple ConvNet as the classifier.
>>>>>>> 2370493bdd91df19e261f1744ec49d5f0662dbf1

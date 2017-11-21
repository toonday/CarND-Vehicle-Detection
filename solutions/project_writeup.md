##Project WriteUp

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup/car_not_car.png
[image3]: ./writeup/pipeline-sliding.jpg
[image4]: ./writeup/pipeline-examples.jpg
[image5]: ./writeup/pipeline-heatmap.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the python file `./solutions/vehicle_detection.py` (through lines 128 - 141 and lines 88 - 100).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I was more into getting a solution that works well, so I spent less time in data exploration than I would have liked, but I spent a lot of time tunning the parameters, exploring different color spaces to find out which configuration gave me better accuracy on my validation data set. 

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters such as orientation, color spaces, hog_channels, etc.
Like I mentioned above my focus was to get a good classification accuracy value.
I ran tests on test examples to have an idea how my configuration would work in the actual video.
I noticed some things I would talk about in the discussion section, but the trade-offs being done when making these decisions are quite interesting.
Definitely documenting your approach to testing and tunning would be very valuable.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the `LinearSVC()` object.
This can be seen in the first code cell of my IPython notebook `./solutions/Testbed_1.ipynb`
I used a combination of parameters as discussed earlier but I noticed I got the best training accuracy using "ALL" channels for the hog features and using either the "YUV" or "YCrCb" color spaces for the color histogram and spatial features. I had memory issues when attempting to use all of these features at once, so I had to resize the image before extracting hog features which made my pipeline a bit easier on memory but a lot harder for me to play around with things (especially when it came to using the `find_cars` function). Also, I could not help but think to myself that a CNN would bhave done a better job here and life would have been a lot easier! [Note: After a long night, I finally thought about reducing the data size used for training and this worked like a charm!, Shame I did not think of this earlier.]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this [NO, I am not kidding! :)]
Yes, I did some random offset (between 1 and 10) from a start y position I set to have some variation in how the windows are selecting to have a broader set of potiental outcomes. I spent so so so so much time experimenting with differnet values and noticed that even with a relatively high accuracy score on my training set of about 98.5% the performance of the model compared to a dumb human like me was still poor. I almost used a CNN, but I felt it would be overkill for this project, but I certainly got an idea of how labour intensive it could be to generate these perfect models which can only work if massaged the right type of way. Definitely more room for improvement in ML/DL workflow in the future! :)
[Sorry, I went a bit off track :)].
Yes, I experimented with some values noticed that the model identified cars based on different scales and starting y position, so I decided to use a minor random offset for starting point and a for loop and a step for using different scales for the search window.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Yeah this was a rough one.
So, since I decided to resize my images before extracting the HOG features, this kind of slowed me down a bit.
But I also could not help but want to normalize and use a zero-mean image for my pipeline. I felt this made sense if you think of how the data we are used for classification are. Basically we could zero out the mean of the patch from the image and be left with the features that would actually aid our model in learning. This idea actually worked well for my `search window` function. Since I was searching and extracting hog features from each window patch by path I could also zero out the mean of the patch in the test set. When I did this, the Linear SVM was actually able to better detect cars in the scene. However like the instructor mentioned in the video extracting features this way was very slow when I actually tried to use it on both the test_video and the project video. So I had to do the next best thing and zero out from a general mean `(x -128) / 255` rather than the cooler `(x - x.mean()) / (x.max() - x.min())`. Still works decently, but I for sure would not be using a self driving car with this pipeline :)

Ultimately I searched on multiple scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video.
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Yeah, I ran out of time here [a long time ago! :)], but I filtered out the image based on heatmaps, duration of consistent detection over several frames and the slope and size of the bounding box detected. I created a `FrameHeatMap` class which had the value of the global heatmap and I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap of the current frame. I accepted or rejected some blobs based on the slope and I also used some counters to detect how long a blob region has been detected. Simple stuff, kind of works [Dear Aesome Reviewer: please feel free to recommend any papers, blogs or articles that adresses segmentation and knowing this is the same car over several frames :)]

The code that I was blurbing about is in `./solutions/vehicle_detection.py` lines 354 - 446.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are the test images and their corresponding heatmaps:

![alt text][image5]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

I kind of mentioned some things along the way in my answers to previous questions, but I am racing to complete this before my deadline today and I need to get some things of my chest! :)

> More complex Model:
    I am generally a guy that favours the simple life, ways and hacks, but in this case a more complex model would have done a better job at classifying the image (a simple CNN should do, which is more complex than the Linear SVM).
    
> Shadows! Shadows! Shadows!:
    My model would definitely struggle in dark areas. This was one of the reasons I attempted to use a zero mean image to reduce the effect of shadows and other noise on the patch, but there was a trade-off between accuracy of model and usefulness/performance of model.
    
> Data Gathering:
    Working on this project also got me thinking about how we gather data and how a biased data set can either affect positively or negatively the accuracy and performance of a model. For example in this case, I kind of wonder if it would be better to have a lot more not car data sets for training so our model does a better job and predicting when there are no cars in the image. But oh well, I think an argument can be made for the other case that letting the model know bout more cars can be better than the former. My point here though is that understanding the problem you are trying to solve can and should influence how you gather data for training. Training a model to identify cars so you can estimate traffic is different for requiring a model to identify cars so you can help train a robot to assist people with disability in crossing the road, one is a bit more life dependent [Once again reviewer I am terrible with words so I am writting programmer English].

> Parameter Tunning:
    This is the part that is either fun or not fun depending on the day. It is quite daunting to know how much time can be spent in tunning models/algorithms for better performance.
    
> Accuracy Score vs Test Score vs User Experience Score:
    I kind of noticed something weird I did not fully understand. I actually got a slightly better accuracy score using the YCrCb color space on the training data, but when it came to the actual task of identifying cars and non-cars in test images the YUV color space seemed to perform a lot better to me as a human. So this got me thinking how do we validate and test ML/DL models in the wild. Is have a high accuracy score good enough or should we have more User experience scoring and testing of ML models and algorithms?
    
> There really is a difference between 98.5% and 99%
    Naively, I always assumed that 98% was good enough, but after re-doing my algorithm to ignore resizing but use less data for training, but use all the hog features, this solved all my problems! I am back on the simplicity train! It really though is amazing how a slight difference in percentages can result in huge performance results. Lesson learnt :)
    
[PHEW! that was a bit much thanks so much for your time. Hopefully I made the dealine fingers-crossed :&. AI and ML is definitely fun]
    
On a side note, I also looked into using some feature extraction algorithms to help reduce the size of my data when I was dealing with my memory issues. They did not help much, I was loosing a lot in accuracy score and resizing was a lot quicker and faster than using PCA or other feature reduction algos I found. Would be interesting to know if you agree or not.

### Reference:
- `./solutions/*` - contatins all project related work done by me
- `./solutions/Testbed_1.ipynb` - starting point, exploration and experimentation
- `./solutions/Testbed_2.ipynb` - heatmap exploration and experimentation
- `./solutions/Testbed_3.ipynb` - pipeline exploration and experimentation with find_cars
- `./solutions/Testbed_4.ipynb` - video pipeline  with search_windows
- `./solutions/Testbed_5.ipynb` - video pipeline with find_cars
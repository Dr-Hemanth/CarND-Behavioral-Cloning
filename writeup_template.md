# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_architecture.PNG "Model Visualization"
[image2]: ./examples/left.jpg "Left Image"
[image3]: ./examples/centre.jpg "Centre Image"
[image4]: ./examples/right.jpg "Right Image"
[image5]: ./examples/left_gray.jpg "Left Grayscaling Image"
[image6]: ./examples/centre_gray.jpg "Centre Grayscaling Image"
[image7]: ./examples/right_gray.jpg "Right Grayscaling Image"
[image8]: ./examples/left_flip.jpg "Left Flipped Image"
[image9]: ./examples/centre_flip.jpg "Centre Flipped Image"
[image10]: ./examples/right_flip.jpg "Right Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 containing the video recorded for one lap.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5*5 and 3*3 filter sizes. (model.py lines 72-88) 

The model includes RELU layers to introduce nonlinearity (code line 76-86), and the data is normalized in the model using a Keras lambda layer (code line 74). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 81). 

The model was trained and validated on different data sets which are provided by Udacity. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 88).

#### 4. Appropriate training data

For the training data I ended up using the dataset provided to me by Udacity. In addition to the center, left and right images from the simulator I used the flipped version of these images to increase the training data. A correction factor of +0.2 was applied for the left steering measurement and a correction factor of -0.2 was applied for the right steering measurement.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy
#### 1. Solution Design Approach
The overall strategy for deriving a model architecture was described below:

My first step was to use a convolution neural network model similar to the  one created by Nvidia. I thought this model might be appropriate because it was applied on real life autonomous driving conditions.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.To combat the overfitting, I modified the model by adding a dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. This was resolved by augmenting the left and right image data, and also the steering angles. This improved my model but there were shakiness when the car was moving. To resolve it, I have adjusted the steering correction value from 0.4 to 0.2 for both left and right.

#### 2. Final Model Architecture

The final model architecture (model.py lines 71-86) consisted of a convolution neural network with the following layers and layer sizes.

* Image normalization & Cropping
* Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
* Dropout : 0.25 probability
* Fully connected: neurons: 1164, activation: RELU
* Fully connected: neurons: 100, activation: RELU
* Fully connected: neurons: 50, activation: RELU
* Fully connected: neurons: 10, activation: RELU
* Fully connected: neurons: 1 (output)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process
The training data was obtained from Udacity provided images (/opt/carnd_p3/data) in the workspace. This data was shuffled and used to 
train the model. As the size of the data increased due to data augmentation I switched to using generators.

To augment the data sat, I also flipped images and angles,
- Randomly shuffled the data set.
- Use OpenCV to convert to RGB for drive.py.
- For steering angle associated with three images, I use correction factor for left and right images with correction factor of 0.2: increase the steering angle by 0.2 for left image and for the right one  decrease the steering angle by 0.2.


i have used the sample driving data from the workspace /opt/carnd_p3/data/.The images of the different positions of the camera ,mounted on the vehicle.Say left camera, centre camera and right camera

![alt text][image2]
![alt text][image3]
![alt text][image4]

Grayscaling of the above images of the left camera, right camera and centre camera.

![alt text][image5]
![alt text][image6]
![alt text][image7]


To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image8]
![alt text][image9]
![alt text][image10]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as validation loss and loss made constant... I used an adam optimizer so that manually training the learning rate wasn't necessary.

* Also refer "README.md" for full detail working of the project.
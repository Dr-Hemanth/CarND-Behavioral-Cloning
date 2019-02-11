# P3: Behavioural Cloning
The objective of this project is to train a model that teaches a car to drive around a track in Udacity's simulator.

This README contains info on

1. Files in this repo
2. Dataset characteristics
    * How data was captured
    * What data was recorded
3. Solution Design
    * Problem objective
    * Pre-processing of input data
    * Data Augumentation
    * Cropping Image
    * Generators
    * Approach taken for designing model architecture
4. Model architecture

## 1. Files in this repo
* `model.py`: Python script used to create, train and save the model.
    * [Video of the model driving](https://www.youtube.com/watch?v=WCat1CpKQc4) 
* `model.h5`: The saved model. It contains the weights of the model.
* `drive.py`: Python script to tel how to drive the car in simulator.
* `data/`: file with training data (no longer in repo)
    * IMG folder - this folder contains all the frames of our driving.
    * driving_log.csv - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of your car. You'll       mainly be using the steering angle..
* `run1/`: The directory in which images are saved while recording the vedio seen by the agent.
* `run1.mp4`: A video recording of our vehicle driving autonomously at least one lap around the track..

## 2. Dataset Characteristics

### Data Generation: Udacity's Car-Driving Simulator
The model was trained by driving a car in the left-hand-side track of Udacity's car-driving simulator. The simulator recorded 9-10 datapoints per second. Each datapoint comprised 7 attributes:
* Image taken from the center camera on the car (320 x 160 pixel image with 3 colour channels)
* Image taken from the left camera on the car
* Image taken from the right camera on the car
* Throttle
* Speed
* Brake
* Steering angle (variable to predict, a float ranging from -1.0 to 1.0 inclusive)

### What data was recorded
In order to start collecting training data, you'll need to do the following:

* Enter Training Mode in the simulator.
* Start driving the car to get a feel for the controls.
* When you are ready, hit the record button in the top right to start recording.
* Continue driving for a few laps or till you feel like you have enough data.
* Hit the record button in the top right again to stop recording.

#### Sample Data
Here are three examples of images and attributes from the dataset.
<br>
<br>
Example One: Left turn

![Center Image](IMG/left_2016_12_01_13_30_48_287.jpg)

<table>
<th>Steering Angle</th><th>Throttle</th><th>Brake</th><th>Speed</th>
<tr><td>0</td><td>0</td><td>0</td><td>22.14829</td></tr>
</table>

Example Two: Straight road

![Center Image](IMG/center_2016_12_01_13_30_48_287.jpg)

<table>
<th>Steering Angle</th><th>Throttle</th><th>Brake</th><th>Speed</th>
<tr><td>0</td><td>0</td><td>0</td><td>22.14829</td></tr>
</table>


Example Three: Right turn

![Center Image](IMG/right_2016_12_01_13_30_48_287.jpg)

<table>
<th>Steering Angle</th><th>Throttle</th><th>Brake</th><th>Speed</th>
<tr><td>0</td><td>0</td><td>0</td><td>22.14829</td></tr>
</table>


### How the model was trained
Now that we have training data, it’s time to build and train your network!

Use Keras to train a network to do the following:
* Take in an image from the center camera of the car. This is the input to your neural network.
* Output a new steering angle for the car.

Around 9000 datapoints were used to train the model. I divided these examples into training and validation sets using `sklearn.model_selection.train_test_split` to reduce overfitting.You don’t have to worry about the throttle for this project, that will be set for you.

Due to limitations in GPU memory, I fed training examples in batches of 32 using a generator. (See *Model Architecture* for details on the model.)

Save your trained model architecture as model.h5 using model.save('model.h5').

## 3. Solution Design

### Objective
The target is for the car to drive within the lane lines, so the main features the model needs to recognise from the center image are the lane lines.

### Pre-processing of input data
The input image fed into the model are the original unprocessed images taken by the camera. They are of size 160x320 pixels with 3 channels (RGB).In order to improve model pre-processing the data is nescessary.
Two pre-processing steps:
1) Normalizing Data : Here Lambda layer is added to model, i will normalize the image by dividing each element by 255, which is maximum value of image pixel.Normalized image is in the reange of 0 to 1.
2) Mean centering Data : After image is normalized, i will mean center the image by substracting 0.5 from each eement, which will shift the element mean down from 0.5 to 0.

All these preprocessing steps were taken inside the model so that these steps could:
1. be easily applied to new input images as they came in, and
2. make the most of parallelised computing if available to reduce computation time.

### Data Augmentation
Since the training track is in loop and the car drives in counter-clockwise.Most of times model is learning to turn left. The best solution is Data Augumentation. In Data augmentation images are flipped horizontally like a mirror. By doing this car will learn to steer clockwise and counter-clockwise.

### Cropping Images
Camera captures 160x320 pixel images, it contains top pixels like mountains, hills and the bottom portion of the image captures the hood of the car. Our model might train faster if you crop each image to focus on only the portion of the image that is useful for predicting a steering angle.Keras provides the Cropping2D layer for image cropping within the model. This is relatively fast, because the model is parallelized on the GPU, so many images are cropped simultaneously. 

### Generators
The images captured in the car simulator are much larger than the images encountered in the Traffic Sign Classifier Project, a size of 160 x 320 x 3 compared to 32 x 32 x 3. Storing 10,000 traffic sign images would take about 30 MB but storing 10,000 simulator images would take over 1.5 GB. Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator you can pull pieces of the data and process them on the fly only when you need them, which is much more memory-efficient.

### Approach taken for designing model architecture

#### Convolutions
* Convolutions generally seem to perform well on images, so I tried adding a few convolution layers to my model. 

#### Activations and Dropout
* Activation to introduce non-linearities into the model: I chose ReLU as my activation. 
* I added dropout of 0.25 to prevent the network from overfitting.

#### Fully connected layer
* I added a fully connected layer after the convolutions to allow the model to perform high-level reasoning on the features taken from the convolutions.

#### Final layer
* For the network constructed i will compile the model. For the loss function i will use 'mean squared error' or MSE because this is reggression network(at output node steering angle is predicted) not classification network.

## 4. Model architecture

My first step was to use a convolution neural network model similar to the one created by Nvidia. I thought this model might be appropriate because it was applied on real life autonomous driving conditions and various combinations of conv layers combined with fully connected layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.To combat the overfitting, I modified the model by adding a dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. This was resolved by augmenting the left and right image data, and also the steering angles. This improved my model but there were shakiness when the car was moving. To resolve it, I have adjusted the steering correction value from 0.4 to 0.2 for both left and right.

The model is a Sequential model comprising five convolution layers and five fully-connected layers. The model weights used were those obtained after training for **5 epochs**.

The model code and specifications are below:
```
model = Sequential()
# Normalise and Mean centering the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Crop 70 pixels from the top of the image and 25 from the bottom
model.add(Cropping2D(cropping=((70, 25), (0, 0))

# Conv layer 1
model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))

# Conv layer 2
model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))

# Conv layer 3
model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))

# Conv layer 4
model.add(Conv2D(64,(3,3), activation='relu'))

# Conv layer 5
model.add(Conv2D(64,(3,3), activation='relu'))

#Adding a Dropout Layer
model.add(Dropout(0.25))
model.add(Flatten())

# Fully connected layer 1
model.add(Dense(1164, activation='relu'))

# Fully connected layer 2
model.add(Dense(100, activation='relu'))

# Fully connected layer 3
model.add(Dense(50, activation='relu'))

# Fully connected layer 4
model.add(Dense(10, activation='relu'))

# Fully connected layer 5
model.add(Dense(1))

#Compiling the Model
model.compile(optimizer='adam', loss='mse')


```

Specs in a table:

<table>
	<th>Layer</th><th>Details</th>
	<tr>
		<td>Convolution Layer 1</td>
		<td>
			<ul>
				<li>Filters: 24</li>
				<li>Kernel: 5 x 5</li>
				<li>Stride: 2 x 2</li>
				<li>Activation: ReLU</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 2</td>
		<td>
			<ul>
				<li>Filters: 36</li>
				<li>Kernel: 5 x 5</li>
				<li>Stride: 2 x 2</li>
				<li>Activation: ReLU</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 3</td>
		<td>
			<ul>
				<li>Filters: 48</li>
				<li>Kernel: 5 x 5</li>
				<li>Stride: 2 x 2</li>
				<li>Activation: ReLU</li>
			</ul>
		</td>
	</tr>
<tr>
		<td>Convolution Layer 4</td>
		<td>
			<ul>
				<li>Filters: 64</li>
				<li>Kernel: 3 x 3</li>
				<li>Activation: ReLU</li>
			</ul>
		</td>
	</tr>
  <tr>
		<td>Convolution Layer 5</td>
		<td>
			<ul>
				<li>Filters: 64</li>
				<li>Kernel: 3 x 3</li>
				<li>Activation: ReLU</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Dropout = 0.25</td>
	</tr>
	<tr>
		<td>Flatten layer</td>
		<td>
			<ul>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 1</td>
		<td>
			<ul>
              <li>Neurons : 1164</li>
				<li>Activation: ReLU</li>
			</ul>
		</td>
	</tr>
   	<tr>
		<td>Fully Connected Layer 2</td>
		<td>
			<ul>
				<li>Neurons: 100</li>
				<li>Activation: ReLU</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 3</td>
		<td>
			<ul>
				<li>Neurons: 50</li>
				<li>Activation: Relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 4</td>
		<td>
			<ul>
              <li>Neurons : 10</li>
				<li>Activation: ReLU</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 5</td>
		<td>
			<ul>
              <li>Neurons : 1</li>
				<li>Activation: ReLU</li>
			</ul>
		</td>
	</tr>
  
</table>

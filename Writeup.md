# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./md_images/summary.png "Model Summary"
[image2]: ./md_images/pipeline.png "Image Augmentation Pipeline"
[image3]: ./md_images/loss.png "Validation Loss"
[image4]: ./md_images/dataset.png "Data Set"
[image5]: ./md_images/centerlane.png "Center Lane"
[image6]: ./md_images/leftlane.png "Left Lane"
[image7]: ./md_images/rightlane.png "Right Lane"
[image8]: ./md_images/shift.png "Shift Axes"
[image9]: ./md_images/flip.png "Flip Horizontal"
[image10]: ./md_images/brightness.png "Adjust Brightness"
[image11]: ./md_images/crop.png "Crop Image"
[image12]: ./md_images/resize.png "Resize Image"

### Files Submitted & Code Quality

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* model.json containing a trained convolution neural network in json

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```
The model.py file contains the code for training and saving the convolution neural network. The next section describes the pipeline for training and validating the model.

### Model Architecture and Training Strategy

The model consists of a convolution neural network with 3 layers of 5x5 filters and 2 layers of 3x3 filters and depths between 24 and 64 (`buildModel()`).

The model includes RELU layer in each convolutional layer to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on Udacity-provided data set to ensure that the model was not overfitting. It used an Adam optimizer with a learning rate of 0.0001. The model summary is shown below:

![alt text][image1]

Training data was chosen to keep the vehicle driving on the road using a combination of 
* reducing the data samples with steering angle of less than ±0.1
* center lane driving
* recovering from the left and right sides of the road
* flipping (horizontal) the images randomly
* shifting the image along the x- and y- axes 
* adjusting the image brightness randomly
* cropping the image from 320x160 to 320x65
* resizing the image from 320x65 to 80x40

I used a convolution neural network model similar to the [Nvidia End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) because it is similar to the project problem.

In order to gauge how well the model was working, the Udacity-provided data set was shuffled and split into a training set and validation set, in the ratios of 9:1. The model is trained using Keras with Tensorflow backend. To obtain new training samples, image augmentation is applied and the image augmentation pipeline is shown below:

![alt text][image2]

`fit_generator()` is used to fit the training model. Generators are used to generate samples per batch to fit into `fit_generator()`. In each batch, random images are selected, shuffled and image augmentation applied (`dataGenerator()`). This ensures that the training and validation data sets fed into the model are always different. The model was trained over 8 epochs of ~7,200 samples each. The figure below depicts the training and validation loss as the model trains. 

![alt text][image3]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the first test track.The car fell off the track at sharp corners and the number of data samples with steering angles less than ±0.1 was reduced to improve the driving behavior. This helped the car to drive autonomously around the track without leaving the road.

### Data Visualization
* Udacity-provided data set distribution

  ![alt text][image4]

* Centre Lane Driving

  ![alt text][image5]

* Recovering from the left side of the road by adding a steer correction angle of 0.22 to the original steering angle
  - Original steering angle of 0.1191711
  - After recovering, steering angle of 0.3191711

  ![alt text][image6]

* Recovering from the right side of the road by adding a steer correction angle of 0.22 to the original steering angle
  - Original steering angle of 0.1191711
  - After recovering, steering angle of -0.0808289

  ![alt text][image7]

* Shifting the image along the x- and y- axes

  ![alt text][image8]

* Flipping the image horizontally
  - Original steering angle of 0.1191711
  - After flipping, steering angle of -0.1191711

  ![alt text][image9]

* Adjusting the image brightness

  ![alt text][image10]

* Cropping the image to 320x65

  ![alt text][image11]

* Resizing the image to 80x40

  ![alt text][image12]


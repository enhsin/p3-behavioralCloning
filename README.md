# **Behavioral Cloning** 

My project includes the following files:
* model.py - the script to create and train the model
* drive.py - for driving the car in autonomous mode (from the project's [repo](https://github.com/udacity/CarND-Behavioral-Cloning-P3))
* model.h5 - Keras convolution neural network model
* README.md - summarizing the results

To drive the car autonomously, run 
```
python drive.py model.h5
```
after the [simulator](https://github.com/udacity/self-driving-car-sim) is started.

To train the model, run
```
python model.py
```
after the [sample](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) training data and [additional](http://web.ics.purdue.edu/~epeng/test2.tgz) training data are unpacked. 

### Model Architecture and Training Strategy

I follow Paul Heraty's [advice](https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb) to use Nvidia’s CNN [model](https://arxiv.org/pdf/1604.07316v1.pdf), which consists of 5 convolutional layers (three 5x5 and two 3x3, depth 24, 36, 48, 64, 64) and 3 fully-connected layers (size 100, 50, 10).

| Layer (type)                    | Output Shape         | Param #    | Connected to            |       
|:-------------------------------:|:--------------------:|:----------:|:-----------------------:|
| cropping2d_1 (Cropping2D)       | (None, 80, 320, 3)   | 0          | cropping2d_input_2[0][0]|
| lambda_1 (Lambda)               | (None, 80, 320, 3)   | 0          | cropping2d_1[0][0]      |
| convolution2d_1 (Convolution2D) | (None, 38, 158, 24)  | 1824       | lambda_1[0][0]          |
| convolution2d_2 (Convolution2D) | (None, 17, 77, 36)   | 21636      | convolution2d_1[0][0]   |   
| convolution2d_3 (Convolution2D) | (None, 7, 37, 48)    | 43248      | convolution2d_2[0][0]   |   
| convolution2d_4 (Convolution2D) | (None, 3, 18, 64)    | 27712      | convolution2d_3[0][0]   |   
| convolution2d_5 (Convolution2D) | (None, 1, 8, 64)     | 36928      | convolution2d_4[0][0]   |   
| flatten_1 (Flatten)             | (None, 512)          | 0          | convolution2d_5[0][0]   |
| dense_1 (Dense)                 | (None, 100)          | 51300      | flatten_1[0][0]         |   
| dropout_1 (Dropout)             | (None, 100)          | 0          | dense_1[0][0]           |   
| activation_1 (Activation)       | (None, 100)          | 0          | dropout_1[0][0]         |   
| dense_2 (Dense)                 | (None, 50)           | 5050       | activation_1[0][0]      |   
| dropout_2 (Dropout)             | (None, 50)           | 0          | dense_2[0][0]           |   
| activation_2 (Activation)       | (None, 50)           | 0          | dropout_2[0][0]         |   
| dense_3 (Dense)                 | (None, 10)           | 510        | activation_2[0][0]      |   
| dropout_3 (Dropout)             | (None, 10)           | 0          | dense_3[0][0]           |   
| activation_3 (Activation)       | (None, 10)           | 0          | dropout_3[0][0]         |   
| dense_4 (Dense)                 | (None, 1)            | 11         | activation_3[0][0]      |   

Total params: 188,219

Images are cropped and normalized (see Sections 9 and 13 of the course material) within the model. I did not add pooling after the convolution (like what we used to do in LeNet) because max pooling was quite time consuming.  I simply use 2x2 stride to reduce the number of parameters. Three dropout layers are added after the three connected layers to reduce overfitting. I started from one dropout layer, but it was not effective. After trying different values by hands, I’m set with dropout rate 0.5, 0.3, and 0.3. The loss function is the mean squared error of the steering angle, which penalizes outlier harshly, so that the car will stay on the track. The loss function is minimized by Adam optimizer. The model will train for a few epochs until the validation loss stops improving using _EarlyStopping_ callback function.

The training data is mainly from the provided Track 1 sample data. I later added two laps of keyboard controlled Track 1 data.  20% of the data are reserved for the validation. Center, left and right cameras are all used to teach the car to drive (details in the next section). 


### Architecture and Training Documentation
I first tried training the central camera data from the sample data. There are 8036 images. Flipping the images (driving clockwise) doubles the amount of the data. The car was able to drive smoothly initially, but it went to a dirt road and crashed.

<table>
  <tr>
    <td><br>version 1</br><br>training loss: 0.0100</br>validation loss: 0.0094</td>
    <td><a href="http://www.youtube.com/watch?feature=player_embedded&v=j5pJNoAC-38" target="_blank"><img src="http://img.youtube.com/vi/j5pJNoAC-38/0.jpg" alt="v1" width="240" height="160" border="10" /></a></td>
  </tr>
</table>


####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.




version 2, correction 0.15, epoch 7
<a href="http://www.youtube.com/watch?feature=player_embedded&v=7eGYfVB5oB8" target="_blank"><img src="http://img.youtube.com/vi/7eGYfVB5oB8/0.jpg" alt="v2_c0.15_7" width="240" height="160" border="10" /></a>


version 2, correction 0.20, epoch 7
<a href="http://www.youtube.com/watch?feature=player_embedded&v=Kd4DzaxKaPA" target="_blank"><img src="http://img.youtube.com/vi/Kd4DzaxKaPA/0.jpg" alt="v2_c0.20_7" width="240" height="160" border="10" /></a>


version 2, correction 0.20, epoch 3
<a href="http://www.youtube.com/watch?feature=player_embedded&v=tY1NqwFFQC0" target="_blank"><img src="http://img.youtube.com/vi/tY1NqwFFQC0/0.jpg" alt="v2_c0.20_3" width="240" height="160" border="10" /></a>


version 3, correction 0.15, epoch 7
<a href="http://www.youtube.com/watch?feature=player_embedded&v=in-UAMa_PQQ" target="_blank"><img src="http://img.youtube.com/vi/in-UAMa_PQQ/0.jpg" alt="v3_c0.15_7" width="240" height="160" border="10" /></a>


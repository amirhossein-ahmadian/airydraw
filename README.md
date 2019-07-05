# Introduction

*AiryDraw!* is a simple augmented reality program which gives you the feeling that you can draw in the air just using your hand. You move your hand in front of webcam, and the program paints the path in which the hand moves in the image (in 2D coordinates), and shows the augmented image in real time. It therefore looks as you were using a virtual pen in your hand to draw something on the air! You have also some additional control, and particularly, you can make gestures (by opening and closing hand) in order to start/stop drawing. 

*AiryDraw!* has been implemented with Python, TensorFlow and OpenCV, and is based on a deep object detector model (Single Shot MultiBox Detector network) and Bayesian tracking techniques. I have made use of a pretrained TensorFlow model of SSD that has been specifically trained for detecting hands, from this [Github repository](https://github.com/victordibia/handtracking) by Victor Dibia. Thanks to that project, a large part of what I needed had been already done. However, after some experiments, I found out that this hand detector was not robust enough to be used in my tracking application. Apparently, the main problem was the threshold value, which was not easy to tune (the model detects a hand when one of the outputs of the network exceeds the threshold value). Therefore, I combined the detector neural network with a Bayesian-like tracking algorithm, which significantly improved the results. This method may be described as an ‘adaptive threshold’ technique. In addition, in order to detect gestures (signals indicating start or stop) I wrote a convolutional network based classifier in TensorFlow from scratch, and trained it on my own labeled dataset of hand images.

I suppose that this software can be used mostly for entertainment now, but it may be extended to other human-computer interaction and educational applications. It is also a good project for programmers and researchers to learn and practice some aspects of deep learning, TensorFlow, and tracking, which was actually the main motivation for me to initiate this!
This program still needs much improvement in the algorithm and interface. Any contribution is welcomed!

# Prerequisites
You can run the program using a Python interpreter (by running *main.py* file). The following software and libraries are required to do this. The numbers indicate the versions which the program has been tested with.
- Python (3.6)
- TensorFlow (1.12)
- Numpy (1.15)

A webcam (or another realtime image capturing device) is also required.

# How to use
After running *main.py*, you will see the image captured by webcam on the screen (if multiple webcams are installed, you may choose the device by changing the *CAM_DEVICE_INDEX* variable in *main.py*). Do the following to test the program:
1. Move your hand slowly in front of the webcam. A bounding box should appear around your hand. If it does not, try decreasing the *DETECT_THRESH* variable (direct detection threshold) in *main.py*. Please also note that currently this program is not able to detect/track more than one hand simultaneously, so just a single hand should be present in the scene.
2. In order to start drawing, you must make a ‘signal’, which means closing your hand (making a fist) and reopening it. Repeat this several times with a moderate speed until the signal is detected. The color of the bounding box will change when the signal is detected.
3. Move your hand gently to draw! The hand should be open while drawing.
4. When you want to finish drawing, make a signal (similar to step 2) again. The color of the bounding box will change to what was before.
5. You can add to your drawing by repeating the above steps.

You can press the following keys on the keyboard at any time:

- *q*: exit
- *c*: clear the drawing
- *h*: halt (stop tracking the hand)

Note: based on my experience, currently the light conditions have a considerable effect on the performance of the system. The best condition is a room with a good lighting, and no direct light (from lamps, etc.) going through the webcam. 

# The Algorithm
The hand detector neural network is a SSD network, which outputs a set of bounding boxes with their associated confidence scores (in [0,1] range). The bounding box which has the highest score will be called the best output of the network.
When the confidence score of the best output of hand detector network exceeds a fixed threshold value (direct detection threshold), we say that a ‘direct detection’ has happened. The algorithm enters the main loop after the first direct detection. When this happens, a variable holding the coordinates of the bounding box around the hand is initialized with the best output of the network. Then, this variable is updated on every iteration in the following way:

If there is a direct detection, the variable is updated to the best output of the hand detector network. Otherwise, a Bayesian tracking algorithm is used to estimate the coordinates of the bounding box and update the variable, based on its value in the previous iteration and the current outputs of the hand detector network.
When the user presses the halt button, the main loop is terminated.

Signal (gesture) detection is handled by a separate module. Its main part is a binary classifier, which classifies images of hands into ‘normal’ (open) and ‘fist’ (closed) classes. The classifier is a feedforward neural network consisting of convolutional and fully connected layers. When feeding image to this network, the webcam image is cropped to the area inside the current bounding box. The number of ‘fist’ detections (following a ‘normal’ one) are counted, and when this count reaches the threshold, a signal has been detected. Besides, the outputs of the classifier network over multiple iterations are smoothed with a quite simple technique inspired by Markov chain.

## Hand Detector Network
This network is an instance of Single Shot MultiBox Detector ([Liu et al, 2015],introduced in [this paper](https://arxiv.org/abs/1512.02325)). SSD is a general real-time object detector model, but Victor Dibia has trained it in TensorFlow to be adapted to the specific task of detecting hands (see References). For training, he has used *ssd_mobilenet_v1_coco* (from TensorFlow zoo) as the starting point, and fine-tuned it on the *EgoHands* dataset, which is an annotated dataset of hands from Indiana University. Please see the GitHub page of [Victordibia’s hand tracking](https://github.com/victordibia/handtracking) project for more details. The model files and source codes used from that project have been placed under *handdetector* directory.
## Bayesian Tracker
This module combines the current outputs of the hand detector network with some information from the past to estimate the position (bounding box) of the hand at the current frame. In essence, the algorithm updates the following two variables on each iteration:
  
- *current_state*: the current estimate that we have for the bounding box (consisting of its position and size)
- *current_variance*: represents the amount of uncertainty about the current state. In other words, the less current_variance is, the more sure we are that the current state is a true estimate of the current bounding box.

In order to update *current_state*, a probability value is calculated for each of the outputs of the detector network, based on the score of the output (given by the network itself) and a likelihood. The likelihood is obtained from a Gaussian with a mean and variance equal to *current_state* and *current _variance* variables. Finally, *current_state* is updated to the output of the network which has the highest probability, if this probability is higher than a constant threshold value.
In the next step, *current_variance* is updated through the following rule:

*current_variance* <- (1-*S*)* *current_variance* + *Motion_Variance*

Where *S* is the score of the output of the detector network with the highest probability (which was selected to update current_state). This means that we first shrink the variance (uncertainty) in such a way that a better (more confident) output of the detector network results in more shrinkage. Then, the constant *Motion_Variance* is added to the value. This constant is there to consider the uncertainty which is caused by movement of the hand, i.e. the variance which corresponds to the movement of the hand from the previous frame to the current frame. Its value can be chosen based on the maximum reasonable speed that we assume hand can move at.

The tracking algorithm has been implemented as a class in the Python module *bayestracker.py*.

## Signal Detector
In this program, ‘signal’ refers to several consecutive hand openings and closings (making a fist by hand and then returning it to normal form). To detect these gestures, on each iteration, first a neural network classifies the image of the hand into one of the ‘fist’ or ‘normal’ classes. This network consists of two convolutional layers followed by two fully connected layers, and a softmax output layer. Input images are resized to a fixed dimension. The network was trained on images of fist (closed) and normal (open) hands (about 120 samples from each class) which were retrieved by Google image search and manually selected. Adam optimizer and Dropout were used in training the network. Moreover, to improve generalization, the training data was augmented by flipping images and adding noise to them.
The detection algorithm watches the outputs of the neural network over several iterations to trigger detections. A counter is incremented whenever the output of the classifier network (the class that has the higher probability) changes. A signal is detected when this counter reaches a fixed threshold. However, there is a timeout for incrementing the counter. That means if the output of the classifier does not change for more than a time specified by a constant, the counter is reset to zero.

In practice, it was observed that the outputs of the classifier network were quite noisy. For example, when the hand remained in the ‘fist’ state, the classifier would output ‘fist’ most of the times, but sometimes it also resulted in ‘normal’ at one or two frames (iterations) and then quickly returned to the true state. To address this, a type of smoothing was used. The basic idea is that if the hand is currently in normal/fist state, it is more likely to remain in the same state at the next frame. In other words, we can consider a hidden Markov model with two states, where the transition probability from each state to itself is higher than its transition probability to the other state. This idea has been currently implemented by the following simple rule: in order to increment the counter, it is not enough that the winner class (normal/fist) changes, but the probability of the new winner class should be also higher than the other class by a fixed threshold value.

The *fistdetector.py* file is the implementation of the neural network classifier. The trained network has been saved in *fist_network**** files. The complete signal detector has been implemented as the class *SignalDetector* in *detector.py*.

# References
This project uses some material (including a trained TensorFlow model) and code form the following work:
Victor Dibia, HandTrack: A Library For Prototyping Real-time Hand TrackingInterfaces using Convolutional Neural Networks, https://github.com/victordibia/handtracking

# License
MIT License - see the LICENSE file for details  


# Chapter 2 

### Purpose

Chapter 2 of Deep Learning with Python brings me up to speed with the math behind machine, and more specifically deep, learning. 
It teaches me about the Dense layers - output = activation(dot(W, input) + b) - about the Sequential model, about forward pass, gradients, backpropogation, training the model, and everything else that is required to do deep learning. 

---
### What is this code?

At the end of chapter 2 there was a section called "build-it-from-scratch", where we were encouraged to use the basic Tensorflow (without the Keras API on top of it) to build a deep learning model "from scratch", using what we just learned. 
Of course, the author told us exactly how to do it if you kept reading the textbook. 
I tried to do most of it myself, referring to earlier pages of the textbook to learn what to do and where. 
However there were parts where I had to look at how the author did it. 

All in all, this is a deep learning model built entirely in basic Tensorflow with no Keras API support (except for the loss function). I am quite proud of it. 

---
### Model Performance?

This model achieved a performance of 81% on the standard MNIST dataset.
A model built using the Keras API, and exactly the same calls, has a performance of 98%. 

I think the main differences in performance are due to optimizations in the backpropogation part of the algorithm. 
I built my own optimizer (called update_weights) that told my model how to change the weights after a gradient computation, and with what size to take this step. 
This probably led to me getting stuck in a gradient local minimum, where no matter which way the weights were changed the loss function went up, and so the model did not apply these changes - because it would lead to a higher loss function, which we were trying to minimize. 
There was no concept of momentum in my optimizer, which is probably why we got stuck in a gradient local minimum. 
Had I used an industry-standard optimizer such as Adam or RMSProp, I believe that I would not have gotten stuck in the gradient local minimum and would have achieved results closer to the Keras model. 

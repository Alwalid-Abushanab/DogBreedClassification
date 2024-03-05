[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
# DogBreedClassification

## Project Overview

Welcome to my Convolutional Neural Networks (CNN) project. In this project, I have built a pipeline that can be used within a web app to process real-world, user-supplied images.  Given an image of a dog, my algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.

![Sample Output][image1]

## Required Libraries
Python version 3.* is needed.

And the following libraries need to be installed

(1) flask

(2) tensorflow

(3) opencv-python

(4) matplotlib

If any of them are unavailable, use pip install (library_name)

## Motivation 
I was really interested in how Convolutional Neural Networks work and wanted to learn more. Furthermore, I am really bad at identifying dog breeds, I only know about 5 of them. So, i thought this project would be a great apportionity to
build a model that can identify dog breeds better than me.

## File Describtion
This Repository contains several files and folders.

The run.py file is a web app program that predicts a dog breed from an image supplied by the user. static and templates folders are needed to run this file.

The dog_app.ipynb is a notbook that was used to explore the data and train some models. It saves the best model for each approach in the saved_models folder.

The extract_bottleneck_features.py contains some helper functions to extract bottleneck features.

## Results
I was able to get an accuracy of 83.01% on the testing data with the InceptionV3 pre-trained model

## Licensing, Authors, Acknowledgements
Udacity Data sets: [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). 
[human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). 
[VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz).


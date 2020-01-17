# Modak Detection

This project deals predicting whether the given image contains a Modak or not(Image Classification). Modak is an Indian sweet popular in many parts of India.

![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSFl0r0k0fR3Uqle3VKUS3LI8K91eL_ouTolRkvaQZigLLYfcM2)

For this project I have used Residual Neural Network for the Binary Classification because :-  

* ResNets are easy to optimize.
* ResNets improve feature extraction.
* ResNets overcome the vanishing gradient problem.
* ResNets can easily gain accuracy from greatly increased depth, producing results which are better than previous networks.

In this project, I have used ResNet-50 which contains 50 parameter layers.

## Data Collection

I downloaded the images of Modak from google. I downloaded 217 distinct images of Modak. For, Non-Modak i used images of Egg, Dairy product, Dessert, Meat, Rice and Fried Food. I got the Non-Modak dataset from Kaggle. So there was a total of 434 images in the dataset 217 for each class(Modak and Non-Modak).

## Data Augmentation

Since i am working on keras framework I used the "ImageDataGenerator" for augmentation. The implementation is shown in "Augmentation.py" and the Augmented image sample is present in "Augmentation Sample". 


# MNIST_M-Digit_Recognizer

## Problem Statement: I have to Create a web Application that can predict the number in the range from 0-9 when user write input number.

MNIST-M dataset is created by combining MNIST digits with the patches randomly extracted from color photos of BSDS500 as their background. and It contains 60,000 training and 10,000 test images.

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.
The database is also widely used for training and testing in the field of machine learning.

I Loaded this MNIST_M Dataset Using Torchvision dataset. You can see the source code in my repo through wich you can download the MNIST_M dataset.

After loading the data I convert the images into numpy array format to proceed further.
After that i reshape the array into 28×28 and normalized it.
and converted the target column into categorical.

After that i build a model using CNN(Convolutional Neural Network) Below you can see the parameters.






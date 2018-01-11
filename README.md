# CNN-for-image-rotation-measurement

Objective is to develop convolutional neural network for image rotation measurment.

Input data for CNN is rotated image, output data - rotation angle.

Size of train set is 10000 samples and validation set 1000 samples, each sample contain rotated image and value of rotation angle.

## Image descripton:
  * size 128x128 pixels;
  * white background, containing from 1 to 5 black rectangles;
  * height and width of each rectangle from 1 to 128 pixels;
  * line width from 12 to ??? pixels;
  * center of each rectangle is within the image;
  * angle of rotation between -10 and 10 degrees.

## Solution description
Solution consists of Jupiter notebook files. First file is notebook with image generator code and samples of generated images, second is notebook containing code for training and perform CNN with some visualization (training plot for test and validation set, error histogram, samples of reconstructed images etc.)

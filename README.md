# Digital-Image-Processing-Assignment-1

Comprehensive Image Processing Tool

## Introduction 
<br>
The comprehensive image processing tool presented in this report is designed to provide users with a wide range of features for image manipulation. Developed as a group project, this tool leverages the knowledge acquired from lectures, tutorials, and practical sessions to empower users to perform both basic and advanced image processing, as well as explore deep learning techniques for image enhancement and transformation.<br>

## Background
<br>
In today's digital age, image processing is a fundamental aspect of various industries, from photography and graphic design to medical imaging and computer vision. With the proliferation of digital media, the demand for tools that enable image manipulation has grown substantially. The project aims to address this need by implementing a versatile image processing tool.<br>

## Algorithms Used
<br>
To achieve the desired functionality, the tool utilizes a combination of traditional image processing techniques and deep learning approaches. Here is a brief overview of the algorithms and methods incorporated:<br>

## Basic Requirements<br>
Image Upload<br>

The tool allows users to upload images of their choice, making it a user-friendly platform for image processing.
<br>
Basic Manipulations
<br>
Color Change:
<br> Users can transform images into color, black and white (BW), and grayscale formats.<br>
Transformations:
<br>
Rotation:
<br> Images can be rotated to any desired angle.<br>
Cropping:
<br> The tool enables users to crop images to their desired size and shape.<br>
Flipping:
 <br>Users can invert the colors of the image.<br>

##  Advanced Requirements
<br>
Filters<br>

The tool offers a variety of filters to enhance image quality, reduce noise, and experiment with artistic effects. Examples include sharpening, smoothing, edge detection, embossing, and more.<br>

Intensity Manipulation using Color Transformation<br>

Users can adjust the intensity levels of an image through color transformation techniques, such as tonal transformations and color balancing.<br>

Image Segmentation<br>

The tool supports image segmentation, allowing users to divide images into multiple segments based on specific criteria, including region-based segmentation.<br>

## Deep Learning Requirements<br>

Image Segmentation <br>

The code performs image segmentation using OpenCV. It converts the image to the HSV
color space and then defines lower and upper bounds for color ranges. A mask is created
using the inRange function, which is used to segment the image by selecting pixels that fall
within the specified color range.<br>

Style Transfer<br>
In addition to image enhancement, the tool also offers a style transfer feature, allowing users to apply artistic styles from one image to another. This creative transformation can turn ordinary photos into works of art and offers a wide range of artistic possibilities.<br>

## Model Architectures
<br>
The tool relies on the following model architectures and techniques:
<br>

VGG-19 Model: This deep learning model is used for style transfer. It is pre-trained on a large dataset and enables the extraction of content and style features from images.<br>

Gram Matrix: Gram matrices are used to capture the style information from the style image. They represent the correlations between features in different layers of the VGG-19 model.<br>

Neural Style Transfer: This technique combines content and style images to create a new image with the content of one and the artistic style of another.<br>

## Conclusion<br>
The comprehensive image processing tool presented in this report provides users with a wide range of image manipulation capabilities. By combining traditional image processing techniques with deep learning algorithms, the tool empowers users to enhance and transform their images in creative and practical ways. Whether for basic adjustments or advanced artistic transformations, this tool offers a valuable resource for a wide range of applications.<br>

With a user-friendly interface and a rich set of features, this tool is well-equipped to meet the diverse needs of users seeking to process and enhance their images.

## Note

<br> make sure to run in jupitor note book
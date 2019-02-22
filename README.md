# Neural Style Transfer

Implementation of Gaty's et. al. approach to image style transfer using Convolutional Neural Networks. The project
aims to transfer the style of one image onto the content of another image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.7
- Pip (Python Package Manager)

### Installing

1. Download source code
2. cd neuralstyletransfer
3. activate your virtual environment (optional)
4. pip install -r requirements.txt

### Sample Usage
`python style_transfer.py -cp "content_images/content_1.jpg" -sp "style_images/picasso.png" -i 1000 -cw 10 -sw 0.0001 -w 1024`

## Usage
The program accepts the following command line arguments:

`-cp`
`--contentpath`
- The (relative) path to the image that's supposed to be used for the content recreation. 
- When no value is provided, program raises an error.
- Example "content_images/content_1.jpg"
- or "/home/user/images/image1.png"

`-sp`
`--stylepath`
- The (relative) path to the image that's supposed to be used for the style recreation.
- When no value is provided, program raises an error.
- Example "style_images/picasso.png"
- or "/home/user/images/image2.png"

`-i`
`--iterations`
- The number of iterations that are going to be computed while recreating content and style in the target image.
- Default: 1000
- Set this to i > 750 for appropriate results.

`-cw`
`--contentweight`
- The weight of the content when transfering the style. Usually it is better to give the content a higher weight than the style. 
- Default: 100
- The range of appropriate values is approximately [100, 0.01]

`-sw`
`--styleweight`
- The weight of the style when transfering. Accordingly to the content weight it is advised to weight the style less than the content in order to be able to recognize the content in the result.
- Default: 0.01

`-w`
`--imagewidth`
- The width of the target image.
- Default: 1024 px.

## Background
Gatys et. al. approach to image style transfer aims to do the following: Given two images, extract the content (what is seen) of one image and the style (how is it visualised) of the other image and merge both extractions in another image. Thus a style transfer from one image to another is achieved.

### Approach
To extract content and style from images, we can use a pre-trained model. The VGG-19 is a neural network that is successfully trained for object localization and classification. Once can feed an image into the network and use it's layers' outputs to extract style and content.

This is just a rough sketch of what's happening. For more information, please have a look at the original paper.

#### Content Extraction
The content of the first reference image is extracted by considering the outputs of each layer in the VGG-19 net. When the reference image is fed into the network, it's layers compute their outcome. If we now optimize (thus repeatedly change) a target image until it's activations match those of the reference image when passed into the VGG-19, we are able to visualize the contents in an image.

#### Style Extraction
The style if the second reference image can be extracted by considering the correlation of the activations of the VGG-19' layers. The correlation of activations is modelled as their' Gramian matrix. 
- https://en.wikipedia.org/wiki/Gramian_matrix
If we now change a target image until the correlations of it's activations match the correlation of the activations of the style reference image, we are able to visualize the style of the reference image.

#### Actual Style Transfer
The actual style transfer is a combination of content and style extraction. A target image is repeatedly changed using an optimization method until both it's content activations and it's style correlation match those of the two reference image.
Thus we minimize a combined loss indicating the "distance" to content and style represenations, respectively.




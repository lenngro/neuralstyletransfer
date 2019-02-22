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
The (relative) path to the image that's supposed to be used for the style recreation.
When no value is provided, program raises an error.
- Example "style_images/picasso.png"
- or "/home/user/images/image2.png"

`-i`
`--iterations`
The number of iterations that are going to be computed while recreating content and style in the target image.
Default: 1000
Set this to i > 750 for appropriate results.

`-cw`
`--contentweight`
The weight of the content when transfering the style. Usually it is better to give the content a higher weight than the style. 
Default: 100
The range of appropriate values is approximately [100, 0.01]

`-sw`
`--styleweight`
The weight of the style when transfering. Accordingly to the content weight it is advised to weight the style less than the content in order to be able to recognize the content in the result.
Default: 0.01

`-w`
`--imagewidth`
The width of the target image.
Default: 1024 px.




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

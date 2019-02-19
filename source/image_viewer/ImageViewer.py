from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.preprocessing import image as kp_image

"""

basewidth = 300
img = Image.open('somepic.jpg')
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('sompic.jpg') 

"""


class ImageViewer(object):

    def __init__(self, image_width):
        self.image_width = image_width

    def load_img(self, path_to_img):
        """
        Load image from disk and resize to a width of 512 and an according height, keeping the ratio.
        :param path_to_img: path to image
        :return: resized image
        """
        img = Image.open(path_to_img)
        wpercent = (self.image_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((self.image_width, hsize), Image.ANTIALIAS)
        img = kp_image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def imshow(self, img, title=None):
        out = np.squeeze(img, axis=0)
        out = out.astype('uint8')
        plt.imshow(out)
        if title is not None:
            plt.title(title)
        plt.imshow(out)

    def save(self, img, directory, name):
        img = Image.fromarray(img, 'RGB')
        img.save(directory + '/' + name + '.png')

    def deprocess_img(self, processed_img):
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                   "dimension [1, height, width, channel] or [height, width, channel]")
        if len(x.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")

        # perform the inverse of the preprocessiing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x

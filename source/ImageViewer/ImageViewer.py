import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from PIL import Image

class ImageViewer(object):

    def __init__(self, image_width):
        """
        Initialize the image viewer.
        :param image_width: target image width
        """
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

    def show(self, img, title=None):
        """
        Display image.
        :param img: image to be displayed
        :param title: name that will be shown above the figure
        :return:
        """
        out = np.squeeze(img, axis=0)
        out = out.astype('uint8')
        plt.imshow(out)
        if title is not None:
            plt.title(title)
        plt.imshow(out)

    def save(self, img, directory, name):
        """
        Saves an image to disk.
        First, ensure the image has the correct shape and revert the zero-centering.
        :param img: image
        :param directory: directory where the image will be stored
        :param name: name of the file (without file format)
        :return:
        """
        if len(img.shape) == 4:
            img = np.squeeze(img, 0)
            assert len(img.shape) == 3

        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype('uint8')
        img = Image.fromarray(img, 'RGB')
        img.save(directory + '/' + name + '.png')

    def process_img(self, img):
        """
        The VGG-19 net expects images to be zero-centered.
        Therefore a transformation is needed.
        :param img: image to be transformed
        :return: transformed image
        """
        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means
        clipped = tf.clip_by_value(img, min_vals, max_vals)
        return img.assign(clipped)
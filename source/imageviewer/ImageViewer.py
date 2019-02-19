from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.preprocessing import image as kp_image

class ImageViewer(object):

    def load_img(self, path_to_img):
        max_dim = 512
        img = Image.open(path_to_img)
        long = max(img.size)
        scale = max_dim / long
        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
        img = kp_image.img_to_array(img)
        # We need to broadcast the image array such that it has a batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def imshow(self, img, title=None):
        # Remove the batch dimension
        out = np.squeeze(img, axis=0)
        # Normalize for display
        out = out.astype('uint8')
        plt.imshow(out)
        if title is not None:
            plt.title(title)
        plt.imshow(out)

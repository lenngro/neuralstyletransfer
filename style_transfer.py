"""
-----------------------------
## Neural Style Transfer ##
@author: Lennart Grosser
-----------------------------
Sample Use:
python style_transfer.py -cp "content_images/content_1.jpg" -sp "style_images/picasso.png" -i 100 -cw 0.001 -sw 0.0001 -w 1024
"""

import tensorflow as tf
import numpy as np
import os, errno
import datetime
from source.model.Model import Model
from source.command_line_parser.CommandLineParser import CommandLineParser
tf.enable_eager_execution()


def ensure_dir(directory):
    """
    Ensure the directory exists or is creatable.
    :return:
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def start_transfer(content_path, style_path, iterations, content_weight, style_weight, target_image_width):
    """
    Executing the style transfer.
    Steps:
    1. Building the model and freezing the layers in order to prevent weight changing.
    2. Receiving the style- and content feature representations that the model outputs for
    the given style- and content image that are being used for the style transfer.
    3. Initializing the target image. As stated in the original paper: the content image will be used as a
    starting point. This image is going to be transformed until it encorporates both style and content of the reference
    images.
    4. Creating the optimizer.
    5. Creating a configuration dictionary that will be used by the model to transform the target image.
    6. For n iterations:
        a. Compute the gradients that indicate how to change the target image in order to minimize the loss functions.
        b. Extract loss (for printing)
        c. Check if the current loss is smaller than the previously best loss.
        If so: make current image the best image (best image = result)
    7. Save the result to disk.
    :param content_path: (relative) Path to content image
    :param style_path: (relative) Path to style image
    :param iterations: number of optimization iterations (>1000 for best results)
    :param content_weight: Weight of the content image
    :param style_weight: Weight of the style image
    :param target_image_width: image width of result image
    :return:
    """
    directory = 'result_images/' + 'transfer_' + str(datetime.datetime.utcnow())
    ensure_dir(directory)

    model = Model(target_image_width)
    for layer in model.model.layers:
        layer.trainable = False

    style_features, content_features = model.get_feature_representations(content_path, style_path)
    gram_style_features = [model.gram_matrix(style_feature) for style_feature in style_features]

    target_image = model.load_and_process_img(content_path)
    target_image = tf.Variable(target_image, dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(learning_rate=5, epsilon=1e-1)
    iteration_count = 1

    best_loss, best_img = np.inf, target_image.numpy()

    loss_weights = (style_weight, content_weight)
    configuration = {
        'loss_weights': loss_weights,
        'target_image': target_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    for i in range(iterations):

        gradients, all_loss = model.compute_gradients(configuration)
        loss, style_score, content_score = all_loss

        optimizer.apply_gradients([(gradients, target_image)])
        target_image = model.viewer.process_img(target_image)

        if loss < best_loss:
            best_loss = loss
            best_img = target_image.numpy()

        if iteration_count % 10 == 0:
            print("Iteration: " + str(iteration_count))
            print("Loss: " + str(loss))

        iteration_count += 1

    model.viewer.save(best_img, directory, "result_image")


if __name__ == '__main__':
    """
    Parse the arguments and start the style transfer.
    """
    parser = CommandLineParser()
    content_path, style_path, iterations, content_weight, style_weight, target_image_width = parser.parse()
    start_transfer(content_path, style_path, iterations, content_weight, style_weight, target_image_width)

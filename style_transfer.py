"""
@title: Neural Style Transfer
@author: Lennart Grosser
Sample Use:
python style_transfer.py -cp "content_images/content_1.jpg" -sp "style_images/picasso.png" -i 100 -cw 0.001 -sw 0.0001 -w 1024
"""
import tensorflow as tf
from source.model.Model import Model
from source.command_line_parser.CommandLineParser import CommandLineParser
import os, errno
import numpy as np
import datetime

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
    parser = CommandLineParser()
    content_path, style_path, iterations, content_weight, style_weight, target_image_width = parser.parse()
    start_transfer(content_path, style_path, iterations, content_weight, style_weight, target_image_width)

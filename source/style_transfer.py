import time
from PIL import Image
import numpy as np
import tensorflow as tf
from model.Model import Model
import os, errno
import datetime
from command_line_parser.CommandLineParser import CommandLineParser
tf.enable_eager_execution()

"""
NEURAL STYLE TRANSFER IMPLEMENTATION

Sample Use:
python style_transfer.py -cp "content_images/content_1.jpg" -sp "style_images/picasso.png" -i 100 -cw 0.001 -sw 0.0001 -w 1024

"""

def ensure_dir():
    """
    Ensure the directory exists or is creatable.
    :return:
    """
    directory = 'result/' + 'transfer_' + str(datetime.datetime.utcnow())

    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

parser = CommandLineParser()
content_path, style_path, iterations, content_weight, style_weight, target_image_width = parser.parse()

"""
Create the model.
"""

model = Model(target_image_width)
for layer in model.model.layers:
    layer.trainable = False

style_features, content_features = model.get_feature_representations(content_path, style_path)
gram_style_features = [model.gram_matrix(style_feature) for style_feature in style_features]

content_init_image = model.load_and_process_img(content_path)
content_init_image = tf.Variable(content_init_image, dtype=tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate=5, epsilon=1e-1)
iteration_count = 1

best_loss, best_img = None, None

loss_weights = (style_weight, content_weight)
cfg = {
    'model': model.model,
    'loss_weights': loss_weights,
    'init_image': content_init_image,
    'gram_style_features': gram_style_features,
    'content_features': content_features
}

norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means

imgs = []

for i in range(iterations):
    grads, all_loss = model.compute_grads(cfg)
    loss, style_score, content_score = all_loss

    optimizer.apply_gradients([(grads, content_init_image)])
    clipped = tf.clip_by_value(content_init_image, min_vals, max_vals)
    content_init_image.assign(clipped)

    if best_loss is None:
        best_loss = loss
        continue

    if loss < best_loss:
        best_loss = loss
        best_img = model.viewer.deprocess_img(content_init_image.numpy())

    if iteration_count % 10 == 0:
        print("Iteration: " + str(iteration_count))
        print("Loss: " + str(loss))

    iteration_count += 1

model.viewer.save(best_img, directory)

import time
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from model.Model import Model
import os, errno
import datetime

"""
NEURAL STYLE TRANSFER IMPLEMENTATION
"""

"""
Enable Tensorflow's eager execution mode.
"""
tf.enable_eager_execution()

"""
If no or falsy command-line arguments were given, use defaults.
"""
content_path = "content_images/content_1.jpg"
style_path = "style_images/picasso.png"
num_iterations = 750
content_weight = 1e3
style_weight = 1e-2


def save(img, directory):
    img = Image.fromarray(img, 'RGB')
    img.save(directory + '/transfered.png')


def deprocess_img(processed_img):
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


"""
Ensure directory exists or can be created before starting the transfer.
"""

directory = 'result/' + 'transfer_' + str(datetime.datetime.utcnow())

try:
    os.makedirs(directory)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

"""
Create the model.
"""

model = Model()
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

num_rows = 2
num_cols = 5
display_interval = num_iterations / (num_rows * num_cols)
start_time = time.time()
global_start = time.time()

norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means

imgs = []

for i in range(num_iterations):
    grads, all_loss = model.compute_grads(cfg)
    loss, style_score, content_score = all_loss

    optimizer.apply_gradients([(grads, content_init_image)])
    clipped = tf.clip_by_value(content_init_image, min_vals, max_vals)
    content_init_image.assign(clipped)
    end_time = time.time()

    if best_loss is None:
        best_loss = loss
        continue

    if loss < best_loss:
        best_loss = loss
        best_img = deprocess_img(content_init_image.numpy())

    if iteration_count % 10 == 0:
        print("Iteration: " + str(iteration_count))
        print("Loss: " + str(loss))

    iteration_count += 1

save(best_img, directory)

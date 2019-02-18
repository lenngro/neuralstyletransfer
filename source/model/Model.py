from tensorflow.python.keras import models

from imageviewer.ImageViewer import ImageViewer
import tensorflow as tf

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

class Model(object):

    def __init__(self):
        self.viewer = ImageViewer()
        self.model = self.create()

    def create(self):
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_outputs = [vgg.get_layer(name).output for name in content_layers]
        model_outputs = style_outputs + content_outputs
        return models.Model(vgg.input, model_outputs)

    def load_and_process_img(self, path_to_img):
        img = self.viewer.load_img(path_to_img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def get_content_loss(self, base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))

    def gram_matrix(self, input_tensor):
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def get_style_loss(self, base_style, gram_target):
        height, width, channels = base_style.get_shape().as_list()
        gram_style = self.gram_matrix(base_style)

        return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)

    def get_feature_representations(self, content_path, style_path):
        content_image = self.load_and_process_img(content_path)
        style_image = self.load_and_process_img(style_path)
        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)
        style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
        content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
        return style_features, content_features

    def compute_loss(self, model, loss_weights, init_image, gram_style_features, content_features):
        style_weight, content_weight = loss_weights
        model_outputs = self.model(init_image)

        style_output_features = model_outputs[:num_style_layers]
        content_output_features = model_outputs[num_style_layers:]

        style_score = 0
        content_score = 0
        weight_per_style_layer = 1.0 / float(num_style_layers)

        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * self.get_style_loss(comb_style[0], target_style)

        weight_per_content_layer = 1.0 / float(num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer * self.get_content_loss(comb_content[0], target_content)

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        return loss, style_score, content_score

    def compute_grads(self, cfg):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(**cfg)
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss







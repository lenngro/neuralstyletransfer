import tensorflow as tf
from tensorflow.python.keras import models
from source.image_viewer.ImageViewer import ImageViewer

class Model(object):

    def __init__(self, image_width):
        """
        The general net structure. Determines which exact layers of the VGG-19 are going to be used.
        """
        self.viewer = ImageViewer(image_width)
        self.content_layers = [
            'block5_conv2'
        ]
        self.style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'
        ]
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.model = self.create()

    def create(self):
        """
        Load the VGG-19 net from the pretrained models section in Keras.
        Freeze it's layers by setting trainable to False. This prevents the layer weights
        from changing while computation.
        Extract the layer's outputs from the VGG-19 which we need to create the model that we're
        going to use for the actual computation.
        :return: model
        """
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs
        return models.Model(vgg.input, model_outputs)

    def load_and_process_img(self, path_to_img):
        """
        Load and format an image to the format that is required by the VGG-19.
        :param path_to_img: Path to target image.
        :return: transformed image
        """
        img = self.viewer.load_img(path_to_img)
        self.viewer.show(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def get_content_loss(self, base_content, target):
        """
        Compute the loss (~distance) from the content of the content image.
        We compute the loss as l(x) = (content_reference -x)².
        :param base_content: content reference (=> content image)
        :param target: target reference (=> image to be transformed)
        :return: loss: Float
        """
        return tf.reduce_mean(tf.square(base_content - target))

    def gram_matrix(self, input_tensor):
        """
        Gaty's approach to neural style transfer computes the style representation of an image as the correlation
        of the outputs of the net's layers. The correlation is defined as the Gramian Matrix.
        Literature:
        https://en.wikipedia.org/wiki/Gramian_matrix
        We need to reshape the matrix in order to multiply it with itself.
        :param input_tensor: Input matrix.
        :return: Gramian matrix as tensor.
        """
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def get_style_loss(self, base_style, gram_target):
        """
        To compute the loss (how far is the target image from the style representation) we need to compute the Gramian
        matrix of our reference image before. We then compute the quadratic loss.
        :param base_style: Style reference.
        :param gram_target: Target reference
        :return: loss: Float
        """
        # height, width, channels = base_style.get_shape().as_list()
        gram_style = self.gram_matrix(base_style)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def get_feature_representations(self, content_path, style_path):
        """
        In order to recreate both content and style from the reference images, we need to retain their
        particular represenation as given by the VGG-19. This is simply achieved by feeding both images into our model
        and capturing the outputs of the layers.
        :param content_path: Path to content image.
        :param style_path: Path to style image.
        :return: style features, content features
        """
        content_image = self.load_and_process_img(content_path)
        style_image = self.load_and_process_img(style_path)
        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)
        style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
        content_features = [content_layer[0] for content_layer in content_outputs[self.num_style_layers:]]
        return style_features, content_features

    def compute_loss(self, loss_weights, target_image, gram_style_features, content_features):
        """
        To compute the overall loss (how far the model is from recreating both content and style in the target image),
        we compute the style- and content loss for each layer separately and simply add them up.
        Layers are weighted equally. The overall loss is simply content loss + style loss.
        :param model: the VGG-19 net (deprecated due to OOP)
        :param loss_weights: weights for both content- and style
        :param init_image: target image to be transformed
        :param gram_style_features: style feature representations
        :param content_features: content feature representations
        :return: overall loss
        """
        style_weight, content_weight = loss_weights
        model_outputs = self.model(target_image)

        style_output_features = model_outputs[:self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers:]

        style_score = 0
        content_score = 0
        weight_per_style_layer = 1.0 / float(self.num_style_layers)

        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * self.get_style_loss(comb_style[0], target_style)

        weight_per_content_layer = 1.0 / float(self.num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer * self.get_content_loss(comb_content[0], target_content)

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        return loss, style_score, content_score

    def compute_gradients(self, configuration):
        """
        Compute the gradients.
        compute_loss(**configuration) passes the "content" of configuration instead of the entire object.
        :param cfg: config of the style transfer
        :return: gradients
        """
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(**configuration)
        total_loss = all_loss[0]
        return tape.gradient(total_loss, configuration['target_image']), all_loss

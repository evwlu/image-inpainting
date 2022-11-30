import tensorflow as tf

class LocalDiscriminator(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # initialize layers
        pass

    def call(self, window):
        """
        takes as input a window (completed or ground truth)
        """
        pass

class GlobalDiscriminator(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # initialize layers
        pass

    def call(self, full_image):
        """
        takes as input the entire image (completed or ground truth)
        """
        pass
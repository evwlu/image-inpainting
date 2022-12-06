import tensorflow as tf

class LocalDiscriminator(tf.keras.Model):
    def __init__(self, shape=(32,32,3), **kwargs):
        super(LocalDiscriminator, self).__init__(**kwargs)
        # initialize layers
        # note: the shape for cifar-100 is (32,32,3); leave this parameter if changing the training data
        self.conv1 = tf.keras.layers.Conv2D(2, 5, strides=2, padding='SAME')
        self.bnorm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        # input: 16 x 16
        self.conv2 = tf.keras.layers.Conv2D(4, 5, strides=2, padding='SAME')
        self.bnorm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        # input: 8 x 8
        self.conv3 = tf.keras.layers.Conv2D(8, 5, strides=2, padding='SAME')
        self.bnorm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        # input: 4 x 4
        self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(units=1024)

    @tf.function
    def call(self, window, training=False):
        """
        takes as input a window (completed or ground truth);
        mask should be applied before
        """
        x = self.conv1(window)
        x = self.bnorm1(x, training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bnorm2(x, training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bnorm3(x, training)
        x = self.relu3(x)

        x = self.flatten(x)
        return self.linear(x)

class GlobalDiscriminator(tf.keras.Model):
    def __init__(self, shape=(32,32,3), **kwargs):
        super(GlobalDiscriminator, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2D(2, 5, strides=2, padding='SAME')
        self.bnorm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        # 16 x 16
        self.conv2 = tf.keras.layers.Conv2D(4, 5, strides=2, padding='SAME')
        self.bnorm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        # 8 x 8
        self.conv3 = tf.keras.layers.Conv2D(8, 5, strides=2, padding='SAME')
        self.bnorm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        # 4 x 4
        self.conv4 = tf.keras.layers.Conv2D(16, 5, strides=2, padding='SAME')
        self.bnorm4 = tf.keras.layers.BatchNormalization()
        self.relu4 = tf.keras.layers.ReLU()

        # 2 x 2
        self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(units=1024)

    @tf.function
    def call(self, full_image, training=False):
        """
        takes as input the entire image (completed or ground truth);
        mask should be applied before
        """
        x = self.conv1(full_image)
        x = self.bnorm1(x, training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bnorm2(x, training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bnorm3(x, training)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bnorm4(x, training)
        x = self.relu4(x)

        x = self.flatten(x)
        return self.linear(x)
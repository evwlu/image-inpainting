import tensorflow as tf
import completion_net as cn
import gan

class ImageInpaint(tf.keras.Model):
    def __init__(self, batch_size, **kwargs):
        """
        completion network, local discriminator, global discriminator, and fully connected layer
        that makes a prediction after concatenating the local + global disc output
        """
        pass
    
    def compile(self, optimizer, loss, metrics):
        """
        initialize optimizer (Adam), initialize loss functions for completion network (l2 loss),
        overall discriminator (binary cross-entropy), initialize acc metric (ie. dice coeff)
        """
        pass

    def train(self, images, M_D, M_C, epoch):
        """
        M_D represents masks used for training the discriminators on real inputs; M_C is the 
        masks used for training it on fake inputs; epoch is the current epoch number (recall that training
        is split up into three phases depending on the epoch number)
        """
        pass

    def test(self, images, masks, epoch):
        """
        apply masks and assess completion network loss + accuracy, discriminator loss
        (note: not concerned with the accuracy of the discriminator)
        """
        pass

import tensorflow as tf
import numpy as np
from completion_net import CompletionNetwork
from gan import LocalDiscriminator, GlobalDiscriminator
import random
from utils.masks import initialize_masks

def get_mini_batch(images, batch_size):
    batch = []
    indices = set()

    # collects a batch of unique indices from images (the entire training dataset)
    while (len(indices) < batch_size):
        indices.add(random.randint(0, len(images)-1))
    
    for idx in indices:
        batch.append(images[idx])
    
    return tf.convert_to_tensor(batch)

def get_windows(images, masks, locations):
    output = []
    for i in range(len(images)):
        img = images[i] * masks[i]

        x, y, w, h = locations[i][0], locations[i][1], locations[i][2], locations[i][3]
        output += [img[x: x+w, y: y+h]]
    
    return np.array(output)

class ImageInpaint(tf.keras.Model):
    def __init__(self, shape=(32,32,3), **kwargs):
        """
        completion network, local discriminator, global discriminator, and fully connected layer
        that makes a prediction after concatenating the local + global disc output
        """
        super().__init__(kwargs)
        self.completion = CompletionNetwork(shape=shape)

        self.local_disc = LocalDiscriminator()
        self.global_disc = GlobalDiscriminator()

        self.concat = tf.keras.layers.Concatenate()
        self.fc = tf.keras.layers.Dense(units=1, activation='sigmoid')
    
    def compile(self, optimizer, losses, metrics):
        """
        initialize optimizer (Adam), initialize loss functions for completion network (l2 loss),
        overall discriminator (binary cross-entropy), initialize acc metric (ie. dice coeff)
        """
        self.optimizer = optimizer

        self.comp_loss = losses[0]
        self.disc_loss = losses[1]
        self.joint_loss = losses[2]

        self.acc = metrics[0]

    def train(self, images, batch_size, T_C, T_D, T):
        """
        M_D represents masks used for training the discriminators on real inputs; M_C is the 
        masks used for training it on fake inputs; epoch is the current epoch number (recall that training
        is split up into three phases depending on the epoch number)
        """
        
        total_comp_loss, total_disc_loss, total_joint_loss = 0, 0, 0
        total_seen = 0

        # note: training is split up into three phases: training completion using 
        # reconstruction (phase 1), training discriminator (phase 2), training completion using joint loss (phase 3)
        for i in range(T):
            total_seen += 1
            batch = get_mini_batch(images, batch_size)

            M_C, locations_C = initialize_masks(batch_size, images.shape[1], int(images.shape[1]/2), int(images.shape[1]/2), int(images.shape[1]/2))
            if (i < T_C):
                
                with tf.GradientTape() as tape:
                    incomplete_images = tf.cast(batch * (1 - M_C), dtype=tf.float32)
                    completed_images = self.completion(incomplete_images, training=True)

                    comp_loss = self.comp_loss(tf.cast(batch, dtype=tf.float64), tf.cast(completed_images, dtype=tf.float64))
                
                    self.update_variables(tape, self.completion, comp_loss)

                total_comp_loss += comp_loss
                avg_loss = total_comp_loss / total_seen

                print(f"\r[Training {i}/{T_C}]\t completion loss={avg_loss:.3f}", end='')

            else:

                fake_windows = get_windows(batch, M_C, locations_C)
                fake_images = None # note: run completion network in the scope of the tape

                M_D, locations_D = initialize_masks(batch_size, images.shape[1], int(images.shape[1]/2), int(images.shape[1]/2), int(images.shape[1]/2))
                real_windows = get_windows(batch, M_D, locations_D)
                real_images = batch

                # train discriminator
                with tf.GradientTape(persistent=True) as tape:
                    local_disc_out_fake = self.local_disc(fake_windows, training=True)

                    fake_images = self.completion(batch * (1 - M_C), training=True)
                    global_disc_out_fake = self.global_disc(fake_images, training=True)

                    fake_concat = self.concat([local_disc_out_fake, global_disc_out_fake])
                    fake_pred = self.fc(fake_concat)
                    
                    local_disc_out_real = self.local_disc(real_windows, training=True)
                    global_disc_out_real = self.global_disc(real_images, training=True)

                    real_concat = self.concat([local_disc_out_real, global_disc_out_real])
                    real_pred = self.fc(real_concat)

                    # train discriminators
                    disc_loss = self.disc_loss(real_pred, fake_pred)

                    self.update_variables(tape, self.fc, disc_loss)
                    self.update_variables(tape, self.concat, disc_loss)
                    self.update_variables(tape, self.local_disc, disc_loss)
                    self.update_variables(tape, self.global_disc, disc_loss)

                    total_disc_loss += disc_loss
                    avg_disc_loss = total_disc_loss / total_seen

                    if (i > T_C + T_D):
                        # train completion net using joint loss
                        joint_loss = self.joint_loss(batch, completed_images, real_pred, fake_pred, 0.0004)
                        self.update_variables(tape, self.completion, joint_loss)

                        total_joint_loss += joint_loss
                        avg_joint_loss = total_joint_loss / total_seen

                        print(f"\r[Training {i - T_C - T_D}/{(T - T_C - T_D)}]\t discriminator loss={avg_disc_loss:.3f}\t joint loss={avg_joint_loss:.3f}", end='')
                    else:
                        print(f"\r[Training {i - T_C}/{T_D}]\t discriminator loss={avg_disc_loss:.3f}", end='')

    def test(self, images):
        """
        apply masks and assess completion network loss + accuracy, discriminator loss
        (note: not concerned with the accuracy of the discriminator)
        """
        pass

    def update_variables(self, tape, layer, loss):
        grads = tape.gradient(loss, layer.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, layer.trainable_variables))

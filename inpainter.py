import tensorflow as tf
import numpy as np
from completion_net import CompletionNetwork
from gan import LocalDiscriminator, GlobalDiscriminator
import random
import os
from utils.masks import initialize_masks
from utils.metrics import dice_coefficient

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
    
    def build(self, input_shape):
        super().build(input_shape)
    
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
    
    @tf.function
    def call(self, incomplete_images):
        return self.completion(incomplete_images)

    def train(self, images, batch_size, T_C, T_D, T, augment_fn, restore=False):
        """
        M_D represents masks used for training the discriminators on real inputs; M_C is the 
        masks used for training it on fake inputs; epoch is the current epoch number (recall that training
        is split up into three phases depending on the epoch number)
        """
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(model=self)

        total_comp_loss, total_disc_loss, total_joint_loss = 0, 0, 0
        total_seen = 0

        # note: training is split up into three phases: training completion using 
        # reconstruction (phase 1), training discriminator (phase 2), training completion using joint loss (phase 3)
        for i in range(T):
            if (i == 0 and restore):
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        
            if ((i + 1) % 100 == 0):
                checkpoint.save(file_prefix=checkpoint_prefix)

            total_seen += 1
            batch = get_mini_batch(images, batch_size)
            batch = augment_fn(batch)

            M_C, locations_C = initialize_masks(batch_size, images.shape[1], int(images.shape[1]/2), int(images.shape[1]/2), int(images.shape[1]/2))
            if (i < T_C):
                
                with tf.GradientTape() as tape:
                    incomplete_images = tf.cast(batch * (1 - M_C), dtype=tf.float64)
                    completed_images = self.completion(incomplete_images, training=True)

                    comp_loss = self.comp_loss(tf.cast(batch, dtype=tf.float64), tf.cast(completed_images, dtype=tf.float64))
                
                    self.update_variables(tape, self.completion, comp_loss)

                total_comp_loss += comp_loss
                avg_loss = total_comp_loss / total_seen

                print(f"\r[Training {i}/{T_C}]\t completion loss={avg_loss:.3f}", end='')

            else:

                fake_windows = tf.cast(get_windows(batch, M_C, locations_C), dtype=tf.float64)
                fake_images = None # note: run completion network in the scope of the tape

                M_D, locations_D = initialize_masks(batch_size, images.shape[1], int(images.shape[1]/2), int(images.shape[1]/2), int(images.shape[1]/2))
                real_windows = tf.cast(get_windows(batch, M_D, locations_D), dtype=tf.float64)
                real_images = tf.cast(batch, dtype=tf.float64)

                # train discriminator
                with tf.GradientTape(persistent=True) as tape:
                    local_disc_out_fake = self.local_disc(fake_windows, training=True)

                    fake_images = self.completion(tf.cast(batch * (1 - M_C), dtype=tf.float64), training=True)
                    global_disc_out_fake = self.global_disc(fake_images, training=True)

                    fake_concat = self.concat([local_disc_out_fake, global_disc_out_fake])
                    fake_pred = self.fc(fake_concat)
                    
                    local_disc_out_real = self.local_disc(real_windows, training=True)
                    global_disc_out_real = self.global_disc(real_images, training=True)

                    real_concat = self.concat([local_disc_out_real, global_disc_out_real])
                    real_pred = self.fc(real_concat)

                    # train discriminators
                    disc_loss = self.disc_loss(real_pred, fake_pred)

                    if (i > T_C + T_D):
                        # train completion net using joint loss
                        incomplete_images = tf.cast(batch * (1 - M_C), dtype=tf.float64)
                        completed_images = tf.cast(self.completion(incomplete_images, training=True), dtype=tf.float64)

                        joint_loss = self.joint_loss(tf.cast(batch, dtype=tf.float64), completed_images, tf.cast(fake_pred, dtype=tf.float64), 0.0004)

                self.update_variables(tape, self.fc, disc_loss)
                self.update_variables(tape, self.concat, disc_loss)
                self.update_variables(tape, self.local_disc, disc_loss)
                self.update_variables(tape, self.global_disc, disc_loss)

                total_disc_loss += disc_loss
                avg_disc_loss = total_disc_loss / total_seen

                if (i > T_C + T_D):
                    # train completion net using joint loss
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
        total_comp_loss = 0
        total_accuracies = 0

        num_images = images.shape[0]

        # note: training is split up into three phases: training completion using 
        # reconstruction (phase 1), training discriminator (phase 2), training completion using joint loss (phase 3)
        for i in range(num_images):
            image = images[i]

            M_C, _ = initialize_masks(1, image.shape[1], int(image.shape[1]/2), int(image.shape[1]/2), int(image.shape[1]/2))
                

            incomplete_images = tf.cast(image * (1 - M_C), dtype=tf.float64)
            completed_images = self.completion(incomplete_images, training=True)

            comp_loss = self.comp_loss(tf.cast(image, dtype=tf.float64), tf.cast(completed_images, dtype=tf.float64))
            dice_coeff = dice_coefficient(tf.cast(image, dtype=tf.float64), tf.cast(completed_images, dtype=tf.float64))

            total_comp_loss += comp_loss
            total_accuracies += dice_coeff

            print(f"\r Testing {i + 1}/{num_images} // avg_loss: {total_comp_loss/(i+1)} dice_coeff (acc) : {total_accuracies/(i+1)}", end='')
        
        avg_loss = total_comp_loss / num_images
        avg_acc = total_accuracies / num_images
        print(f"\nTraining Summary: average completion loss={avg_loss:.3f} // average accuracy = {avg_acc:.3f}", end='')

    def update_variables(self, tape, layer, loss):
        grads = tape.gradient(loss, layer.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, layer.trainable_variables))
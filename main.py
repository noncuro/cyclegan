import time

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import data
import eval
import hparams
import train
import wandb

AUTOTUNE = tf.data.AUTOTUNE

if __name__ == "__main__":
    wandb.init(project="cyclegan-horse-zebra")
    print("Num GPUs available: ", len(tf.config.list_physical_devices("GPU")))
    # tf.debugging.set_log_device_placement(True)

    dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                                  with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    print(f"Size of horse training set: {len(train_horses)} instances")
    print(f"Size of zebra training set: {len(train_zebras)} instances")

    train_horses = train_horses.cache().map(
        data.preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        hparams.BUFFER_SIZE).batch(hparams.BATCH_SIZE)

    train_zebras = train_zebras.cache().map(
        data.preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        hparams.BUFFER_SIZE).batch(hparams.BATCH_SIZE)

    sample_horse = next(iter(train_horses))
    # Currently unused.
    sample_zebra = next(iter(train_zebras))

    generator_g = pix2pix.unet_generator(hparams.OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(hparams.OUTPUT_CHANNELS, norm_type='instancenorm')

    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # TODO: check if we have permissions to store under /tmp/
    checkpoint_path = "/tmp/checkpoints/train"

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_f=generator_f,
                               discriminator_x=discriminator_x,
                               discriminator_y=discriminator_y,
                               generator_g_optimizer=generator_g_optimizer,
                               generator_f_optimizer=generator_f_optimizer,
                               discriminator_x_optimizer=discriminator_x_optimizer,
                               discriminator_y_optimizer=discriminator_y_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_step = train.make_training_step(generator_g, generator_f, discriminator_x, discriminator_y,
                                          generator_g_optimizer, generator_f_optimizer,
                                          discriminator_x_optimizer,
                                          discriminator_y_optimizer)

    print(f"Starting training for {hparams.EPOCHS} epochs")
    global_step = -1
    for epoch in range(hparams.EPOCHS):
        print(f"Starting epoch {epoch}")
        start = time.time()

        writer = tf.summary.create_file_writer("./tmp/cyclegans/train")
        n = 0
        with writer.as_default():
            for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
                global_step += 1
                loss_dict = train_step(image_x, image_y)
                wandb.log(loss_dict, step=global_step)
                for loss_key, loss_val in loss_dict.items():
                    tf.summary.scalar(loss_key, loss_val, step=global_step)
                if n % 1 == 0:
                    print('.', end='')
                n += 1

        # Using a consistent image (sample_horse) so that the progress of the model
        # is clearly visible.
        fake_zebra = eval.generate_images(generator_g, sample_horse)
        fake_horse = eval.generate_images(generator_f, sample_zebra)

        wandb.log({"fake_zebra":wandb.Image(fake_zebra),
                   "fake_horse":wandb.Image(fake_horse)})

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))

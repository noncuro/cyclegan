import time
import hparams
import tensorflow as tf

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return hparams.LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return hparams.LAMBDA * 0.5 * loss


def make_training_step(generator_g,
                       generator_f,
                       discriminator_x,
                       discriminator_y,
                       generator_g_optimizer,
                       generator_f_optimizer,
                       discriminator_x_optimizer,
                       discriminator_y_optimizer):
    @tf.function
    def train_step(real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = generator_g(real_x, training=True)
            cycled_x = generator_f(fake_y, training=True)

            fake_x = generator_f(real_y, training=True)
            cycled_y = generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = generator_f(real_x, training=True)
            same_y = generator_g(real_y, training=True)

            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)

            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                  generator_g.trainable_variables))

        generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                  generator_f.trainable_variables))

        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                      discriminator_x.trainable_variables))

        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                      discriminator_y.trainable_variables))

    return train_step

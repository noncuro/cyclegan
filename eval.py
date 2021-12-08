import matplotlib.pyplot as plt


def preprocess_image_test(image, label):
    image = normalize(image)
    return image


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

if False:
    test_horses = test_horses.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_zebras = test_zebras.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    # Run the trained model on the test dataset
    for inp in test_horses.take(5):
        generate_images(generator_g, inp)

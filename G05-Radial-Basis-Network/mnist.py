import gzip
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 28
NUMBER_OF_IMAGES = 60000


def MNIST_images():
    with gzip.open('train-images-idx3-ubyte.gz', 'r') as f:
        f.read(16)
        buf = f.read(IMAGE_SIZE * IMAGE_SIZE * NUMBER_OF_IMAGES)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(NUMBER_OF_IMAGES, IMAGE_SIZE * IMAGE_SIZE)
        return data


def MNIST_labes():
    with gzip.open('train-labels-idx1-ubyte.gz', 'r') as f:
        f.read(8)
        buf = f.read(NUMBER_OF_IMAGES)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return data


def print_image(data):
    data = data.reshape(IMAGE_SIZE, IMAGE_SIZE)
    image = np.asarray(data).squeeze()
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    images = MNIST_images()
    labels = MNIST_labes()
    print_image(images[105])
    print(labels[105])

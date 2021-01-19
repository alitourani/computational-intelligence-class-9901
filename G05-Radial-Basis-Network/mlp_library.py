import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

from mnist import MNIST_images, MNIST_labes
from block_print import block_print, enable_print

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def do_mlp(X_train, y_train, X_test, y_test, hidden_layer_size):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size, max_iter=10, alpha=1e-4,
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=.1, )
    # block_print()
    mlp.fit(X_train, y_train)
    # enable_print()
    print(f'hidden layer size: {mlp.hidden_layer_sizes}')
    print(f'Training set score: {mlp.score(X_train, y_train)}')
    print(f'Test set score: {mlp.score(X_test, y_test)}')


hidden_layers = [
    (5,),
    (20,),
    (50,),
    (100,),
    (200,),
    (30, 30),
    (50, 50),
    (100, 100),
    (150, 150),
    (200, 200),
]

if __name__ == '__main__':
    X = MNIST_images()
    y = MNIST_labes()
    X = X / 255.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for hidden_layer in hidden_layers:
        do_mlp(X_train, y_train, X_test, y_test, hidden_layer)

import numpy as np
import mnist


def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)


def to_one_hot(x, num_of_classes):
    arr = np.zeros((len(x), num_of_classes))
    for i in range(len(x)):
        c = int(x[i])
        arr[i][c] = 1
    return arr


def kmeans(X, k, max_iters):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    converged = False
    current_iter = 0

    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for i in range(len(centroids))]

        for x in X:
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)

        cluster_list = list((filter(None, cluster_list)))
        prev_centroids = centroids.copy()
        centroids = []

        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0))

        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))
        converged = (pattern == 0)
        current_iter += 1

    return np.array(centroids), [np.std(x) for x in cluster_list]


class RBF:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.centroids = []
        self.std_list = []
        self.w = 0

        self.k = 10

    def rbf(self, x, c, s):
        distance = get_distance(x, c)
        return 1 / np.exp(-distance / s ** 2)

    def rbf_list(self, data):
        rbf_list = []
        for x in self.X:
            rbf_list.append([self.rbf(x, c, s) for (c, s) in zip(self.centroids, self.std_list)])
        return np.array(rbf_list)

    def fit(self):
        self.centroids, self.std_list = kmeans(self.X, self.k, max_iters=1000)
        d_max = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
        self.std_list = np.repeat(d_max / np.sqrt(2 * self.k), self.k)
        rbf_x = self.rbf_list(self.X)
        self.w = np.linalg.pinv(rbf_x.T @ rbf_x) @ rbf_x.T @ to_one_hot(self.y, 10)

    def test(self, data, labels):
        rbf_list_tst = self.rbf_list(data)
        pred_ty = rbf_list_tst @ self.w
        pred_ty = np.array([np.argmax(x) for x in pred_ty])
        diff = pred_ty - labels
        print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))


if __name__ == '__main__':
    images = mnist.train_images().astype(float).reshape(60000, -1)
    labels = mnist.train_labels().astype(float)

    train_labels = labels[0:500]
    train_images = images[0:500]

    test_labels = labels[1000:1500]
    test_images = images[1000:1500]

    rbf = RBF(train_images, train_labels)
    rbf.fit()
    rbf.test(test_images, test_labels)

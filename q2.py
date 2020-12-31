'''
Question 2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import numpy as np
import matplotlib.pyplot as plt
import os

PREFIX = "digit_"

TEST_STEM = "test_"
TRAIN_STEM = "train_"


def load_data(data_dir, stem):
    """
    Loads data from either the training set or the test set and returns the pixel values and
    class labels
    """
    data = []
    labels = []
    for i in range(0, 10):
        path = os.path.join(data_dir, PREFIX + stem + str(i) + ".txt")
        digits = np.loadtxt(path, delimiter=',')
        digit_count = digits.shape[0]
        data.append(digits)
        labels.append(np.ones(digit_count) * i)
    data, labels = np.array(data), np.array(labels)
    data = np.reshape(data, (-1, 64))
    labels = np.reshape(labels, (-1))
    return data, labels


def logsumexp_stable(a, axis=None):
    '''
    Compute the logsumexp of the numpy array x along an axis.
    '''
    m = np.max(a, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(a - m), axis=axis)) + np.squeeze(m, axis=axis)


class GaussianDiscriminantAnalysis:

    def compute_mean_mles(self, X, y):
        """
        X = Train data
        y = Train labels
        """
        self.Mu = np.array([np.mean(X[y == k], axis=0) for k in range(10)])
        return self.Mu

    def compute_sigma_mles(self, X, y):
        """
        X = Train data
        y = Train labels
        """

        def _cov(X):
            A = X.T
            A -= np.average(A, axis=1, returned=True)[0][:, None]
            c = np.dot(A, X.conj())
            c *= np.true_divide(1, 699)
            return c.squeeze()

        self.Cov = np.array([_cov(X[y == k]) for k in range(10)])

    def avg_conditional_likelihood(self, X, y):
        '''
        Compute the conditional likelihood:

            log p(y|x, mu, Sigma)

        This should be a numpy array of shape (n, 10)
        Where n is the number of datapoints and 10 corresponds to each digit class
        formula used (from slides): log p(t|x) ∝ log p(x|t)p(t) =
        -1/2(x - u)^T(sigma)^-1(x-u) -1/2log(det(sigma^-1)) - d/2(log(2pi)) + log p(t)
        '''
        if not self.predicted: self.predict(X)
        s = 0
        for i in range(10):
            pxy = self.likc[y == i]
            s += logsumexp_stable(pxy)
            -1 / 2 * np.linalg.det(np.linalg.inv(self.Cov[i])) * len(pxy)  # log-sum p(x|y)
        return s / X.shape[0] + np.log(0.1) - 32 * np.log(6.28)  # p(t) = 0.1

    def fit(self, X, y):
        """
        X = Train data
        y = Train labels
        """
        self.predicted = False
        self.compute_mean_mles(X, y)
        self.compute_sigma_mles(X, y)

    def generative_likelihood(self, X):
        '''
        Compute the generative log-likelihood:
            log p(x|y,mu,Sigma)

        Should return an n x 10 numpy array
        '''
        if not self.predicted: self.predict(X)
        add = 1 / 2 * np.array([np.log(np.linalg.det(np.linalg.inv(self.Cov[k])))
                                for k in range(10)])
        add = np.array([add[k] * np.ones(int(len(self.likc) / 10))
                        for k in range(10)]).reshape(len(self.likc))
        gl = add + self.likc + np.ones(self.likc.shape) * 32 * np.log(2 * 3.14)
        return gl.reshape((int(len(X) / 10), 10))

    def conditional_likelihood(self, X):
        '''
        Compute the conditional likelihood:

            log p(y|x, mu, Sigma)

        This should be a numpy array of shape (n, 10)
        Where n is the number of datapoints and 10 corresponds to each digit class
        '''
        pxy = self.generative_likelihood(X)
        return pxy + np.log(0.1) * np.ones(pxy.shape)  # 0.1 is the prior

    def predict(self, X):
        """X = Test data
        """
        self.predicted = True
        self.labels = []
        self.likc = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            self.labels.append(self.predict_i(X[i], i))

        return np.array(self.labels)

    def predict_i(self, X, i):
        """
        i = index of sample
        X = Test_data[i]

        """
        max_label = 0
        max_likelihood = 0

        for k in range(10):
            likelihood = np.exp(-1 / 2 * (X - self.Mu[k]).T @ np.linalg.inv(self.Cov[k]) @ (X - self.Mu[k]))
            if likelihood > max_likelihood:
                max_label = k
                max_likelihood = likelihood
                self.likc[i] = -1 / 2 * (X - self.Mu[k]).T @ np.linalg.inv(self.Cov[k]) @ (X - self.Mu[k])
        return max_label


def main():
    train_data, train_labels = load_data("", TRAIN_STEM)
    test_data, test_labels = load_data("", TEST_STEM)

    gda = GaussianDiscriminantAnalysis()
    gda.fit(train_data, train_labels)

    # cond_lik = gda.conditional_likelihood(test_data)      conditional likelihood p(y|x)
    # gen_lik = gda.generative_likelihood(test_data)        generative likelihood p(x|y)

    print("average log likelihood= {} - log p(x)".format(gda.avg_conditional_likelihood(test_data, test_labels)))
    print("test accuracy= {}".format(sum(gda.labels == test_labels) / len(test_labels)))
    print("train accuracy= {}".format(sum(gda.predict(train_data) == train_labels) / len(train_labels)))
    print("average log likelihood - Train = {} - log p(x)".format(gda.avg_conditional_likelihood(train_data, train_labels)))

    def plot_cov(cov):
        """
        q 2c) Plot the leading eigenvectors of each covariance k
        """
        fig = plt.figure(figsize=(100, 100))
        for i in range(1, 11):
            e1, e2 = np.linalg.eigh(cov[i - 1])
            fig.add_subplot(1, 10, i)
            plt.imshow(e2[np.argmax(e1)].reshape(8, 8))

    plot_cov(gda.Cov)

if __name__ == '__main__':
    main()

from Coursera.Exercise4.Sigmoid import sigmoid


def sigmoid_gradient(z):
    ans = sigmoid(z) * (1 - sigmoid(z));
    return ans

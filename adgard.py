import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split

# Mean Squared Error (MSE) 함수 정의
def mse(w, xy):
    (x, y) = xy
    o = np.sum(x * w, axis=1)
    return np.mean((o - y) ** 2) / 2

# grad_mse 함수 정의
def grad_mse(w, xy):
    (x, y) = xy
    (rows, cols) = x.shape
    
    # Compute the output
    o = np.sum(x * w, axis=1)
    diff = y - o
    diff = diff.reshape((rows, 1))
    diff = np.tile(diff, (1, cols))
    grad = -np.sum(diff * x, axis=0)
    return grad

# SGD를 추가한 그래디언트 디센트 함수
def stochastic_gradient_descent(max_epochs, threshold, w_init, obj_func, grad_func, xy,
                                learning_rate=0.05, momentum=0.8):
    (x_train, y_train) = xy
    w = w_init
    w_history = w
    f_history = obj_func(w, xy)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    rows = x_train.shape[0]
    error_history = []

    # Run epochs
    while i < max_epochs and diff > threshold:
        # Shuffle rows using a fixed seed to reproduce the results
        np.random.seed(i)
        p = np.random.permutation(rows)
        
        # Run for each instance/example in training set
        for x, y in zip(x_train[p, :], y_train[p]):
            delta_w = -learning_rate * grad_func(w, (np.array([x]), y)) + momentum * delta_w
            w = w + delta_w
            
        i += 1
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, xy)))
        diff = np.absolute(f_history[-1] - f_history[-2])

        # Compute error rate and append to error_history
        error_rate = np.mean(f_history[-1])
        error_history.append(error_rate)
        
    return w_history, error_history

# Add Adagrad to the gradient descent function
def adagrad(max_iterations, threshold, w_init, obj_func, grad_func, extra_param=[], initial_learning_rate=0.05, epsilon=1e-8):
    w = w_init
    w_history = w
    f_history = obj_func(w, extra_param)
    grad_squared = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    error_history = []

    while i < max_iterations and diff > threshold:
        grad = grad_func(w, extra_param)
        grad_squared += grad ** 2
        adjusted_learning_rate = initial_learning_rate / (np.sqrt(grad_squared) + epsilon)
        w -= adjusted_learning_rate * grad

        # Store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, extra_param)))

        # Compute error rate and append to error_history
        error_rate = np.mean(f_history[-1])
        error_history.append(error_rate)

        # Update iteration number and diff between successive values
        # of objective function
        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, error_history

# Returns error rate of classifier
def error(w, xy):
    (x, y) = xy
    o = np.sum(x * w, axis=1)
    
    # Map the output values to 0/1 class labels
    o = np.where(o > 0.5, 1, 0)
    return np.mean(o != y) * 100

# Load dataset
digits, target = dt.load_digits(n_class=2, return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(digits, target, test_size=0.2, random_state=10)
x_train = np.hstack((np.ones((y_train.size, 1)), x_train))
x_test = np.hstack((np.ones((y_test.size, 1)), x_test))

# Initial weights
rand = np.random.RandomState(19)
w_init = rand.uniform(-1, 1, x_train.shape[1]) * .000001

# Run SGD
w_history_sgd, error_history_sgd = stochastic_gradient_descent(100, 0.1, w_init, mse, grad_mse, (x_train, y_train), learning_rate=1e-6, momentum=0.7)

# Run Adagrad
w_history_adagrad, error_history_adagrad = adagrad(100, 0.1, w_init, mse, grad_mse, (x_train, y_train), initial_learning_rate=0.05)

# Plot the error rate
plt.plot(np.arange(len(error_history_sgd)), error_history_sgd, label='SGD')
plt.plot(np.arange(len(error_history_adagrad)), error_history_adagrad, label='Adagrad')
plt.xlabel('Iteration')
plt.ylabel('Error Rate')
plt.title('Error Rate Comparison')
plt.legend()
plt.grid(True)
plt.show()

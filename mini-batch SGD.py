import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split

# Make threshold a negative value if you want to run exactly
# max_iterations.
def gradient_descent(max_iterations, threshold, w_init,
                     obj_func, grad_func, extra_param=[],
                     learning_rate=0.05, momentum=0.8):
    
    w = w_init
    w_history = [w]
    f_history = [obj_func(w, extra_param)]
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    
    while i < max_iterations and diff > threshold:
        delta_w = -learning_rate * grad_func(w, extra_param) + momentum * delta_w
        w = w + delta_w
        
        # store the history of w and f
        w_history.append(w)
        f_history.append(obj_func(w, extra_param))
        
        # update iteration number and diff between successive values
        # of objective function
        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])
    
    return w_history, f_history

# Input argument is weight and a tuple (train_data, target)
def grad_mse(w, xy):
    (x, y) = xy
    (rows, cols) = x.shape
    
    # Compute the output
    o = np.sum(x * w, axis=1)
    diff = y - o
    diff = diff.reshape((rows, 1))    
    diff = np.tile(diff, (1, cols))
    grad = diff * x
    grad = -np.sum(grad, axis=0)
    return grad

# Input argument is weight and a tuple (train_data, target)
def mse(w, xy):
    (x, y) = xy
    
    # Compute output
    # keep in mind that we're using mse and not mse/m
    # because it would be relevant to the end result
    o = np.sum(x * w, axis=1)
    mse = np.sum((y - o) * (y - o))
    mse = mse / 2
    return mse

# Returns error rate of classifier
# total misclassifications/total*100
def error(w, xy):
    (x, y) = xy
    o = np.sum(x * w, axis=1)
    
    # map the output values to 0/1 class labels
    ind_1 = np.where(o > 0.5)
    ind_0 = np.where(o <= 0.5)
    o[ind_1] = 1
    o[ind_0] = 0
    return np.sum((o - y) * (o - y)) / y.size * 100

def visualize_fw():
    xcoord = np.linspace(-10.0, 10.0, 50)
    ycoord = np.linspace(-10.0, 10.0, 50)
    w1, w2 = np.meshgrid(xcoord, ycoord)
    pts = np.vstack((w1.flatten(), w2.flatten()))
    
    # All 2D points on the grid
    pts = pts.transpose()
    
    # Function value at each point
    f_vals = np.sum(pts * pts, axis=1)
    function_plot(pts, f_vals)
    plt.title('Objective Function Shown in Color')
    return pts, f_vals

# Helper function to annotate a single point
def annotate_pt(text, xy, xytext, color):
    plt.plot(xy[0], xy[1], marker='P', markersize=10, c=color)
    plt.annotate(text, xy=xy, xytext=xytext,
                 arrowprops=dict(arrowstyle="->",
                                 color=color,
                                 connectionstyle='arc3'))

# Plot the function
# Pts are 2D points and f_val is the corresponding function value
def function_plot(pts, f_val):
    f_plot = plt.scatter(pts[:, 0], pts[:, 1],
                         c=f_val, vmin=min(f_val), vmax=max(f_val),
                         cmap='RdBu_r')
    plt.colorbar(f_plot)
    # Show the optimal point
    annotate_pt('global minimum', (0, 0), (-5, -7), 'yellow')

# (xy) is the (training_set, target) pair
def stochastic_gradient_descent(max_epochs, threshold, w_init,
                                obj_func, grad_func, xy,
                                learning_rate=0.05, momentum=0.8):
    (x_train, y_train) = xy
    w = w_init
    w_history = [w]
    f_history = [obj_func(w, xy)]
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    rows = x_train.shape[0]
    
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
        w_history.append(w)
        f_history.append(obj_func(w, xy))
        diff = np.absolute(f_history[-1] - f_history[-2])
        
    return w_history, f_history

def gradient_descent(max_iterations, threshold, w_init, obj_func, grad_func, extra_param=[], learning_rate=0.05, momentum=0.8):
    w = w_init
    w_history = [w]
    f_history = [obj_func(w, extra_param)]
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    
    while i < max_iterations and diff > threshold:
        delta_w = -learning_rate * grad_func(w, extra_param) + momentum * delta_w
        w = w + delta_w
        w_history.append(w)
        f_history.append(obj_func(w, extra_param))
        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])
    
    return w_history, f_history

def adagrad(max_iterations, threshold, w_init, obj_func, grad_func, extra_param=[], initial_learning_rate=0.05, epsilon=1e-8):
    w = w_init
    w_history = [w]
    f_history = [obj_func(w, extra_param)]
    grad_squared = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    error_history = []

    while i < max_iterations and diff > threshold:
        grad = grad_func(w, extra_param)
        grad_squared += grad ** 2
        adjusted_learning_rate = initial_learning_rate / (np.sqrt(grad_squared) + epsilon)
        w -= adjusted_learning_rate * grad
        w_history.append(w)
        mse = obj_func(w, extra_param)
        f_history.append(mse)
        error_history.append(mse)  # MSE 저장
        diff = np.absolute(f_history[-1] - f_history[-2])
        i += 1

    return w_history, error_history

def rmsprop(max_iterations, threshold, w_init, obj_func, grad_func, extra_param=[], initial_learning_rate=0.05, decay_rate=0.9, epsilon=1e-8):
    w = w_init
    w_history = [w]
    f_history = [obj_func(w, extra_param)]
    rmsprop_cache = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    error_history = []

    while i < max_iterations and diff > threshold:
        grad = grad_func(w, extra_param)
        rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * grad ** 2
        adjusted_learning_rate = initial_learning_rate / (np.sqrt(rmsprop_cache) + epsilon)
        w -= adjusted_learning_rate * grad
        w_history.append(w)
        mse = obj_func(w, extra_param)
        f_history.append(mse)
        error_history.append(mse)  # MSE 저장
        diff = np.absolute(f_history[-1] - f_history[-2])
        i += 1

    return w_history, error_history

# (xy) is the (training_set, target) pair
def stochastic_gradient_descent(max_epochs, threshold, w_init,
                                obj_func, grad_func, xy,
                                learning_rate=0.05, momentum=0.8):
    (x_train, y_train) = xy
    w = w_init
    w_history = [w]
    f_history = [obj_func(w, xy)]
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    rows = x_train.shape[0]
    
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
        w_history.append(w)
        f_history.append(obj_func(w, xy))
        diff = np.absolute(f_history[-1] - f_history[-2])
        
    return w_history, f_history

# Generate some random data
x, y = dt.make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize weights and learning rate
np.random.seed(11)
w_init = np.random.uniform(-1, 1, x_train.shape[1]) * 0.000001
eta = 1e-3

# Run Adagrad
w_history_adagrad, error_history_adagrad = adagrad(1000, 0.01, w_init, mse, grad_mse, (x_train, y_train), initial_learning_rate=eta, epsilon=1e-8)

# Run RMSprop
w_history_rmsprop, error_history_rmsprop = rmsprop(1000, 0.01, w_init, mse, grad_mse, (x_train, y_train), initial_learning_rate=eta, decay_rate=0.9, epsilon=1e-8)

# Run SGD
w_history_sgd, error_history_sgd = stochastic_gradient_descent(1000, 0.01, w_init, mse, grad_mse, (x_train, y_train), learning_rate=eta, momentum=0.8)

plt.plot(np.arange(len(error_history_adagrad)), error_history_adagrad, label='Adagrad')
plt.plot(np.arange(len(error_history_rmsprop)), error_history_rmsprop, label='RMSprop')
plt.plot(np.arange(len(error_history_sgd)), error_history_sgd, label='SGD')
plt.title('Adagrad vs RMSprop vs SGD - Learning Curves')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Parameter 변화에 따른 시각화
param_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

for param in param_values:
    # Run Adagrad
    w_history_adagrad, error_history_adagrad = adagrad(1000, 0.01, w_init, mse, grad_mse, 
                                                       (x_train, y_train), initial_learning_rate=param, epsilon=1e-8)

    # Run RMSprop
    w_history_rmsprop, error_history_rmsprop = rmsprop(1000, 0.01, w_init, mse, grad_mse, 
                                                        (x_train, y_train), initial_learning_rate=param, decay_rate=0.9, epsilon=1e-8)

    # Run SGD
    w_history_sgd, error_history_sgd = stochastic_gradient_descent(1000, 0.01, w_init, mse, grad_mse, 
                                                                   (x_train, y_train), learning_rate=param, momentum=0.8)

    axs[0].plot(np.arange(len(error_history_adagrad)), error_history_adagrad, label='Adagrad, lr=' + str(param))
    axs[1].plot(np.arange(len(error_history_rmsprop)), error_history_rmsprop, label='RMSprop, lr=' + str(param))
    axs[2].plot(np.arange(len(error_history_sgd)), error_history_sgd, label='SGD, lr=' + str(param))

axs[0].set_title('Adagrad - Learning Curves')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('MSE')
axs[0].legend()

axs[1].set_title('RMSprop - Learning Curves')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('MSE')
axs[1].legend()

axs[2].set_title('SGD - Learning Curves')
axs[2].set_xlabel('Iterations')
axs[2].set_ylabel('MSE')
axs[2].legend()

plt.tight_layout()
plt.show()
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

# Generate some random data
x, y = dt.make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

rand = np.random.RandomState(11)
w_init = rand.uniform(-1, 1, x_train.shape[1]) * 0.000001
eta = 1e-6
for alpha, ind in zip([10, 20, 30], [1, 2, 3]):

    w_history_stoch, mse_history_stoch = stochastic_gradient_descent(
                                100, 0.01, w_init,
                              mse, grad_mse, (x_train, y_train),
                             learning_rate=eta * alpha, momentum=0)
    
    # Plot the MSE
    plt.subplot(3, 1, ind)
    plt.plot(np.arange(len(mse_history_stoch)), mse_history_stoch, color='blue')
    plt.legend(['stochastic'])
    
    # Display total iterations
    plt.text(3, -45, 'Stochastic: Iterations=' +
             str(len(mse_history_stoch)))
    plt.title('batch size = ' + str(alpha))   
    
    train_error_stochastic = error(w_history_stoch[-1], (x_train, y_train))
    test_error_stochastic = error(w_history_stoch[-1], (x_test, y_test))
    
    print('batch size = ' + str(alpha))
    
    print('\tStochastic:')
    print('\t\tTrain error: ' + "{:.2f}".format(train_error_stochastic))
    print('\t\tTest error: ' + "{:.2f}".format(test_error_stochastic))
        
plt.show()

# Nesterov Accelerated Gradient Descent
def nag(max_epochs, threshold, w_init, obj_func, grad_func, xy, 
        learning_rate=0.05, momentum=0.8):
    (x_train, y_train) = xy
    w = w_init
    w_history = [w]
    f_history = [obj_func(w, xy)]
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    rows = x_train.shape[0]
    
    while i < max_epochs and diff > threshold:
        np.random.seed(i)
        p = np.random.permutation(rows)
        
        for x, y in zip(x_train[p, :], y_train[p]):
            w_ahead = w + momentum * delta_w
            delta_w = momentum * delta_w - learning_rate * grad_func(w_ahead, (np.array([x]), y))
            w = w + delta_w
            
        i += 1
        w_history.append(w)
        f_history.append(obj_func(w, xy))
        diff = np.absolute(f_history[-1] - f_history[-2])
        
    return w_history, f_history

# Stochastic Gradient Descent with Momentum
def sgd_with_momentum(max_epochs, threshold, w_init, obj_func, grad_func, xy,
                       learning_rate=0.05, momentum=0.8):
    (x_train, y_train) = xy
    w = w_init
    w_history = [w]
    f_history = [obj_func(w, xy)]
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    rows = x_train.shape[0]
    
    while i < max_epochs and diff > threshold:
        np.random.seed(i)
        p = np.random.permutation(rows)
        
        for x, y in zip(x_train[p, :], y_train[p]):
            delta_w = momentum * delta_w - learning_rate * grad_func(w, (np.array([x]), y))
            w = w + delta_w
            
        i += 1
        w_history.append(w)
        f_history.append(obj_func(w, xy))
        diff = np.absolute(f_history[-1] - f_history[-2])
        
    return w_history, f_history

# Parameter 변화에 따른 시각화 (NAG와 SGD with Momentum)
def visualize_parameter_variation(param_values, title_prefix):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    
    for param in param_values:
        w_history_nag, mse_history_nag = nag(100, 0.01, w_init, 
                                              mse, grad_mse, (x_train, y_train),
                                              learning_rate=eta, momentum=param)
        w_history_sgd_momentum, mse_history_sgd_momentum = sgd_with_momentum(100, 0.01, w_init, 
                                                                              mse, grad_mse, (x_train, y_train),
                                                                              learning_rate=eta, momentum=param)
        
        axs[0].plot(np.arange(len(mse_history_nag)), mse_history_nag, label='momentum=' + str(param))
        axs[1].plot(np.arange(len(mse_history_sgd_momentum)), mse_history_sgd_momentum, label='momentum=' + str(param))
    
    axs[0].set_title(title_prefix + ' - Learning Curves (NAG)')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('MSE')
    axs[0].legend()
    
    axs[1].set_title(title_prefix + ' - Learning Curves (SGD with Momentum)')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('MSE')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

# Generate some random data
x, y = dt.make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize weights and learning rate
rand = np.random.RandomState(11)
w_init = rand.uniform(-1, 1, x_train.shape[1]) * 0.000001
eta = 1e-6

# Parameter values to vary
momentum_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Visualize parameter variation for NAG and SGD with Momentum
visualize_parameter_variation(momentum_values, 'NAG and SGD with Momentum')
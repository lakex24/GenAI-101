# sample N parabola coefficients,
# so that the parabola will pass through point (x0, y0)
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.ticker import MultipleLocator


def generate_parabola(x0, y0, line_count):
    """
    Generate N parabola coefficients, so that the parabola will pass through point (x0, y0)
    """
    # generate N random parabola coefficients
    # y = a*x^2 + b*x + c
    min_val = -0.05
    max_val = 0.05
    a = np.random.uniform(min_val, max_val, line_count)
    b = np.random.uniform(min_val, max_val, line_count)

    c = y0 - a * x0 ** 2 - b * x0

    return a, b, c


# Sample N y values at x intervals, given a, b, c
def get_parabola_y_values(a, b, c, x_values=np.linspace(0, 10, 11)):
    """
    Get N samples of y values at x intervals, given a, b, c
    Return 'y_values', a list of length = line_count,
        each item in the list is an ndarray of length = len(x_values)
    """
    
    # Calculate y values for each set of coefficients
    line_count = len(a)
    y_values = [a[i] * x_values**2 + b[i] * x_values + c[i] for i in range(line_count)]

    # y_values will be a list of length = line_count
    # each item in the list will be an ndarray of length = len(x_values)
    return y_values


def plot_parabola(x_values, y_values):
    # Plotting
    plt.figure(figsize=(10, 6))  # Set the figure size
    for i in range(len(y_values)):
        plt.plot(x_values, y_values[i], label=f'Parabola {i+1}')

    # Optional: enable this if you want to see the labels (might clutter the plot if N is large)
    # plt.legend()
    
    # Get the current axes
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    # Set major tick locator to 1.0 intervals for both axes
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(1))


    plt.title('Plot of N Parabolas')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)  # Enable grid for better readability
    plt.show()


def unit_test_run():
    
    line_count = 20
    a, b, c = generate_parabola(0, 0, line_count)  # a.shape = (line_count,), same for b, c

    x_count = 10
    x_values = np.linspace(0, x_count, x_count + 1)  # x_values.shape = (x_count + 1,)
    
    y_values = get_parabola_y_values(a, b, c, x_values)  # y_values.shape = (line_count, x_count + 1)

    plot_parabola(x_values, y_values)

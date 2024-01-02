import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# calculate values for a and b
def my_linfit(x, y):
    # size for n
    n = len(x)
    # sum for each combination
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    square_x = np.sum([x_i ** 2 for x_i in x])
    sum_xy = np.sum([x_i * y_i for x_i, y_i in zip(x, y)])

    # calculate the a and b
    a = (n * sum_xy - sum_x * sum_y) / (n * square_x - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    return a, b


x_vals = []
y_vals = []


# Save clicked values and plot line when right clicking
def onclick(event):
    if event.button == 1:
        x_vals.append(event.xdata)
        y_vals.append(event.ydata)
    elif event.button == 3 and len(x_vals) > 1:
        a, b = my_linfit(x_vals, y_vals)
        xp = np.linspace(-10, 10, 10)
        plt.plot(xp, a * xp + b, 'r-')
        print(f"Myfit:a={a} and b={b}")


# Create plot for clicked places
fig, ax = plt.subplots()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.autoscale(False)
fig.canvas.mpl_connect('button_press_event', onclick)


# Animate values in real time to the plot
def animate(i):
    # Cheks if there is something to plot
    if x_vals:
        plt.plot(x_vals, y_vals, 'kx')


ani = FuncAnimation(plt.gcf(), animate, cache_frame_data=False, interval=500)
plt.show()

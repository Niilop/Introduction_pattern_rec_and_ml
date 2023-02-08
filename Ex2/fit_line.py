from matplotlib import pyplot as plt
import numpy as np


# Linear solver
def my_linfit(x, y):
    N = len(x)

    a1 = N * sum(y * x)
    a2 = (sum(y)) * (sum(x))
    a3 = N * sum(x ** 2) - sum(x) ** 2

    a = (a1 - a2) / a3
    b = (sum(y) - a * sum(x)) / N
    return a, b


# Function for mouse inputs
def onclick(event):
    # When left mouse button is clicked -> appends x,y coordinates to a list
    if event.button == 1:
        x_crds.append(event.xdata)
        y_crds.append(event.ydata)

        # Show the datapoint on graph
        ax.plot(event.xdata, event.ydata, 'kx')
        plt.show()

    # When right mouse button is clicked -> fits a line from the collected points
    if event.button == 3:
        x = np.array(x_crds)
        y = np.array(y_crds)

        a, b = my_linfit(x, y)

        xp = np.arange(0, 5, 0.1)
        ax.plot(xp, a * xp + b, 'r-')

        plt.show()


# Main

x_crds = []
y_crds = []

plt.rcParams['backend'] = 'TkAgg'

fig, ax = plt.subplots()
ax.axis([0, 1, 0, 1])

# onclick event bind to mpl click event
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

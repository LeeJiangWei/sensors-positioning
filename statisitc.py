import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_history(file: str):
    history = np.load(file)
    x = np.arange(0, len(history))
    plt.plot(x, history)
    plt.show()


def plot_points(anchors, sensors, max_range):
    fig, ax = plt.subplots()
    plt.xlim(0, max_range)
    plt.ylim(0, max_range)
    ax.scatter(sensors[:, 0], sensors[:, 1], s=1, c="blue")
    ax.scatter(anchors[:, 0], anchors[:, 1], c="red")
    plt.show()


if __name__ == '__main__':
    plot_fitness_history("./history/NP10CR0.1F0.5.npy")

    from main import anchors, sensors, LOCATION_RANGE
    plot_points(anchors, sensors, LOCATION_RANGE)

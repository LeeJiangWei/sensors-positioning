import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_history(file: str, real_file: str):
    history = np.load(file)
    real_history = np.load(real_file)
    x = np.arange(0, len(history))
    rx = np.arange(0, len(real_history)) * 20
    plt.plot(x, history, label=file)
    plt.plot(rx, real_history, label=real_file)


def plot_points(anchors, sensors, max_range):
    fig, ax = plt.subplots()
    plt.xlim(0, max_range)
    plt.ylim(0, max_range)
    ax.scatter(sensors[:, 0], sensors[:, 1], s=1, c="blue")
    ax.scatter(anchors[:, 0], anchors[:, 1], c="red")
    plt.show()


if __name__ == '__main__':
    plot_fitness_history("./history/NP100CR0.1F0.5.npy", "./history/NP100CR0.1F0.5_real.npy")
    plot_fitness_history("./history/NP100CR0.1F0.5_jade.npy", "./history/NP100CR0.1F0.5_real_jade.npy")
    plt.legend()
    plt.show()

    from main import anchors, sensors, LOCATION_RANGE

    # plot_points(anchors, sensors, LOCATION_RANGE)

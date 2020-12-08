import random
import numpy as np
import matplotlib.pyplot as plt

NUM_ANCHORS = 5  # M
NUM_SENSORS = 200  # N
NOISE_FACTOR = 0.2

LOCATION_RANGE = 100

np.random.seed(0)


def initialize():
    _anchors = np.random.rand(NUM_ANCHORS, 2) * LOCATION_RANGE
    _sensors = np.random.rand(NUM_SENSORS, 2) * LOCATION_RANGE
    return _anchors, _sensors


anchors, sensors = initialize()


def calculate_distance(anchors_positions: np.ndarray, sensors_positions: np.ndarray):
    n = NUM_ANCHORS + NUM_SENSORS
    positions = np.concatenate([anchors_positions, sensors_positions])
    distance_matrix = np.zeros([n, n])

    for x in range(n):
        for y in range(n):
            if x != y:
                distance_matrix[x][y] = np.sqrt(np.sum(np.power(positions[x] - positions[y], 2)))

    return distance_matrix


def add_noise(distance_matrix: np.ndarray):
    n = distance_matrix.shape[0]
    noised_matrix = np.array(distance_matrix)
    for x in range(n):
        for y in range(n):
            if x >= NUM_ANCHORS and y >= NUM_ANCHORS:  # only add noises to distances between sensors
                if x < y:
                    noised_matrix[x][y] += NOISE_FACTOR * np.random.normal()
                else:
                    noised_matrix[x][y] = noised_matrix[y][x]

    return noised_matrix


def is_symmetric(matrix):
    height, width = matrix.shape

    for x in range(height):
        for y in range(width):
            if matrix[x][y] != matrix[y][x]:
                return False

    return True


if __name__ == '__main__':
    fig, ax = plt.subplots()
    plt.xlim(0, LOCATION_RANGE)
    plt.ylim(0, LOCATION_RANGE)

    distance = calculate_distance(anchors, sensors)
    noised_distance = add_noise(distance)

    ax.scatter(sensors[:, 0], sensors[:, 1], s=1, c="blue")
    ax.scatter(anchors[:, 0], anchors[:, 1], c="red")
    plt.show()

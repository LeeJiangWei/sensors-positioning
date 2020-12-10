import numpy as np
import matplotlib.pyplot as plt

# configs of problem
NUM_ANCHORS = 5  # M
NUM_SENSORS = 200  # N
NOISE_FACTOR = 0.2
LOCATION_RANGE = 100

# configs of algorithm
NP = 200  # size of population
MAX_GEN = 50000  # maximum generations

F = 0.5  # scale factor of DE
CR = 0.8  # cross rate
D = NUM_SENSORS * 2  # len of individual

np.random.seed(0)


def initialize_data():
    _anchors = np.random.rand(NUM_ANCHORS, 2) * LOCATION_RANGE
    _sensors = np.random.rand(NUM_SENSORS, 2) * LOCATION_RANGE
    return _anchors, _sensors


anchors, sensors = initialize_data()


def calculate_full_distance(anchors_positions: np.ndarray, sensors_positions: np.ndarray):
    n = NUM_ANCHORS + NUM_SENSORS
    positions = np.concatenate([anchors_positions, sensors_positions])
    distance_matrix = np.zeros([n, n])

    for x in range(n):
        for y in range(n):
            if x != y:
                distance_matrix[x][y] = np.sqrt(np.sum(np.power(positions[x] - positions[y], 2)))

    return distance_matrix


def calculate_partial_distance(anchors_positions: np.ndarray, sensors_positions: np.ndarray):
    distance_matrix = np.zeros([len(anchors_positions), len(sensors_positions)])
    h, w = distance_matrix.shape

    for x in range(h):
        for y in range(w):
            distance_matrix[x][y] = np.sqrt(np.sum(np.power(anchors_positions[x] - sensors_positions[y], 2)))

    return distance_matrix


def add_noise(distance_matrix: np.ndarray):
    h, w = distance_matrix.shape
    noised_matrix = np.array(distance_matrix)
    if h == w:
        for x in range(h):
            for y in range(w):
                if x >= NUM_ANCHORS and y >= NUM_ANCHORS:  # only add noises to distances between sensors
                    if x < y:
                        noised_matrix[x][y] += NOISE_FACTOR * np.random.normal()
                    else:
                        noised_matrix[x][y] = noised_matrix[y][x]
    else:
        for x in range(h):
            for y in range(w):
                noised_matrix[x][y] += NOISE_FACTOR * np.random.normal()

    return noised_matrix


def is_symmetric(matrix):
    height, width = matrix.shape

    for x in range(height):
        for y in range(width):
            if matrix[x][y] != matrix[y][x]:
                return False

    return True


def initialize_population() -> np.ndarray:
    return np.random.rand(NP, D) * LOCATION_RANGE


def decode(individual: np.ndarray):
    sensors_positions = np.empty([NUM_SENSORS, 2])
    for i in range(len(individual)):
        sensors_positions[int(i / 2)][i % 2] = individual[i]
    return sensors_positions


def evaluate_individual(individual: np.ndarray, judge_matrix: np.ndarray):
    sensors_positions = decode(individual)
    distance_matrix = calculate_partial_distance(anchors, sensors_positions)
    return np.sum(np.abs(distance_matrix - judge_matrix)) / distance_matrix.size


def evaluate_population(population: np.ndarray, judge_matrix: np.ndarray):
    _fitness = np.empty(NP)
    for i in range(NP):
        _fitness[i] = evaluate_individual(population[i], judge_matrix)
    return _fitness


if __name__ == '__main__':
    distance = calculate_partial_distance(anchors, sensors)
    noised_distance = add_noise(distance)

    population = initialize_population()
    fitness_vector = evaluate_population(population, noised_distance)

    for _ in range(MAX_GEN):
        for i in range(NP):
            a = np.random.randint(0, NP)
            while a == i:
                a = np.random.randint(0, NP)
            b = np.random.randint(0, NP)
            while b == i or b == a:
                b = np.random.randint(0, NP)
            c = np.random.randint(0, NP)
            while c == i or c == b or c == a:
                c = np.random.randint(0, NP)

            trial = np.zeros(D)
            p = np.random.randint(0, D)
            for j in range(D):
                if np.random.rand() < CR or j == p:
                    trial[j] = population[a][j] + F * (population[b][j] - population[c][j])
                else:
                    trial[j] = population[i][j]

            new_fitness = evaluate_individual(trial, noised_distance)

            if new_fitness <= fitness_vector[i]:
                fitness_vector[i] = new_fitness
                population[i] = trial
                print("updated!")

        print(np.min(fitness_vector))

    fig, ax = plt.subplots()
    plt.ion()
    plt.xlim(0, LOCATION_RANGE)
    plt.ylim(0, LOCATION_RANGE)
    ax.scatter(sensors[:, 0], sensors[:, 1], s=1, c="blue")
    ax.scatter(anchors[:, 0], anchors[:, 1], c="red")
    plt.pause(1)
    plt.ioff()

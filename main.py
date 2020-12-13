import matplotlib.pyplot as plt
import numpy as np

# configs of problem
NUM_ANCHORS = 5  # M
NUM_SENSORS = 200  # N
NOISE_FACTOR = 0.2
LOCATION_RANGE = 100

# configs of algorithm
NP = 10  # size of population

F = 0.5  # scale factor of DE
CR = 0.1  # cross rate
D = NUM_SENSORS * 2  # len of individual

STOP_THRESHOLD = 0.2

np.random.seed(0)


def initialize_data():
    _anchors = np.random.rand(NUM_ANCHORS, 2) * LOCATION_RANGE
    _sensors = np.random.rand(NUM_SENSORS, 2) * LOCATION_RANGE
    return _anchors, _sensors


anchors, sensors = initialize_data()


def calculate_partial_distance(anchors_positions: np.ndarray, sensors_positions: np.ndarray) -> np.ndarray:
    distance_matrix = np.zeros([len(anchors_positions), len(sensors_positions)])
    h, w = distance_matrix.shape

    for x in range(h):
        for y in range(w):
            distance_matrix[x][y] = np.sqrt(np.sum(np.power(anchors_positions[x] - sensors_positions[y], 2)))

    return distance_matrix


def add_noise(distance_matrix: np.ndarray) -> np.ndarray:
    h, w = distance_matrix.shape
    noised_matrix = np.array(distance_matrix)

    for x in range(h):
        for y in range(w):
            noised_matrix[x][y] += NOISE_FACTOR * np.random.normal()

    return noised_matrix


def initialize_population() -> np.ndarray:
    return np.random.rand(NP, D) * LOCATION_RANGE


def decode(individual: np.ndarray) -> np.ndarray:
    sensors_positions = individual.reshape([-1, 2])
    return sensors_positions


def decode_population(population: np.ndarray) -> np.ndarray:
    sensors_positions = population.reshape([len(population), -1, 2])
    return sensors_positions


def calculate_avg_distance(distance_matrix, judge_matrix):
    return np.sum(np.abs(distance_matrix - judge_matrix)) / distance_matrix.size


def evaluate_individual(individual: np.ndarray, judge_matrix: np.ndarray) -> np.float:
    sensors_positions = decode(individual)
    distance_matrix = calculate_partial_distance(anchors, sensors_positions)
    return calculate_avg_distance(distance_matrix, judge_matrix)


def evaluate_population(population: np.ndarray, judge_matrix: np.ndarray) -> np.ndarray:
    population_positions = decode_population(population)
    _fitness = np.empty(NP)
    for i in range(NP):
        distance_matrix = calculate_partial_distance(anchors, population_positions[i])
        _fitness[i] = calculate_avg_distance(distance_matrix, judge_matrix)
    return _fitness


if __name__ == '__main__':
    true_distance = calculate_partial_distance(anchors, sensors)
    noised_distance = add_noise(true_distance)

    population = initialize_population()
    fitness_vector = evaluate_population(population, noised_distance)

    # print(calculate_avg_distance(true_distance, noised_distance))  # 0.15~0.16

    min_fitness = np.inf
    curr_gen = 0
    history = []

    try:
        while min_fitness > STOP_THRESHOLD:
            a = population.copy()
            b = population.copy()
            c = population.copy()
            np.random.shuffle(a)
            np.random.shuffle(b)
            np.random.shuffle(c)

            # mutation
            mutant_population = a + F * (b - c)

            # crossover
            crossover_mask = np.less(np.random.random([NP, D]), CR)
            trial_population = crossover_mask * mutant_population + (1 - crossover_mask) * population

            # evaluate and selection
            trial_fitness_vector = evaluate_population(trial_population, noised_distance)
            selection_mask = np.less(trial_fitness_vector, fitness_vector)
            population[selection_mask, :] = trial_population[selection_mask, :]
            fitness_vector[selection_mask] = trial_fitness_vector[selection_mask]

            print(f"Generation {curr_gen}: {min_fitness}")
            curr_gen += 1
            min_fitness = np.min(fitness_vector)
            history.append(min_fitness)
    except KeyboardInterrupt:
        print("Interrupt by keyboard")
    finally:
        history = np.array(history)
        np.save(f"./history/NP{NP}CR{CR}F{F}", history)

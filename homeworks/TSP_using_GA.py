import random
import statistics
import time
import matplotlib.pyplot as plt

# ==========================
#       GA Implementation
# ==========================

# Seed for reproducibility; you can change or remove this for different results.
random.seed(123)

# We will consider a TSP with 30 randomly generated cities.
num_cities = 30
# Each city is represented as a coordinate in 2D space.
cities = [(random.random() * 100, random.random() * 100) for _ in range(num_cities)]

# Create a distance matrix for all pairs of cities.
distance_matrix = [[0] * num_cities for _ in range(num_cities)]
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            dx = cities[i][0] - cities[j][0]
            dy = cities[i][1] - cities[j][1]
            distance_matrix[i][j] = (dx**2 + dy**2)**0.5

# GA parameters
population_size = 60    # Number of individuals in population
generations = 100       # Number of generations
mutation_rate = 0.15     # Probability of mutating an individual
tournament_size = 3     # Tournament size for selection
runs = 10               # Number of independent runs to get statistics

def calculate_distance(route):
    """Calculate the total distance of a given route by summing the distances between consecutive cities."""
    dist = 0
    for i in range(len(route)):
        dist += distance_matrix[route[i]][route[(i+1) % num_cities]]
    return dist

def create_random_individual():
    """Create a random individual (i.e., a random permutation of city indices)."""
    individual = list(range(num_cities))
    random.shuffle(individual)
    return individual

def tournament_selection(pop):
    """
    Tournament selection:
    - Randomly select `tournament_size` individuals.
    - Return the individual (actually the route) with the best (lowest) fitness.
    Each element in 'pop' is a tuple (individual, distance).
    """
    candidates = random.sample(pop, tournament_size)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]

def order_crossover(parent1, parent2):
    """
    Order Crossover (OX):
    - Randomly select a sub-section from parent1 and copy it to the child.
    - Fill the remaining positions with the order of nodes from parent2, skipping any already in the child.
    """
    size = len(parent1)
    start, end = sorted([random.randint(0, size - 1) for _ in range(2)])
    child = [None] * size
    # Copy a substring from parent1
    for i in range(start, end+1):
        child[i] = parent1[i]
    # Fill remaining positions with elements from parent2 that are not in the child
    ptr = 0
    for i in range(size):
        if parent2[i] not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = parent2[i]
    return child

def mutate(individual):
    """
    Mutation (simple swap mutation):
    - With probability 'mutation_rate', swap two elements in the route.
    """
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

def run_ga():
    """
    Run one instance of the Genetic Algorithm:
    - Initialize population
    - Evolve for a given number of generations
    - Keep track of best solutions per generation
    """
    # Initialize population with random individuals
    population = [(create_random_individual(), None) for _ in range(population_size)]
    # Calculate fitness (distance) for each individual
    population = [(ind, calculate_distance(ind)) for (ind, _) in population]

    best_distances = []
    for gen in range(1, generations+1):
        # Sort population by fitness (distance)
        population.sort(key=lambda x: x[1])
        # Record the best distance this generation
        best_distances.append(population[0][1])

        # Create the next generation
        next_pop = population[:2]  # Elitism: carry the best two from previous generation
        while len(next_pop) < population_size:
            # Tournament selection to choose parents
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            # Crossover
            child = order_crossover(p1, p2)
            # Mutation
            mutate(child)
            # Add to new population
            next_pop.append((child, calculate_distance(child)))
        population = next_pop

    # After all generations, sort and return the best solution
    population.sort(key=lambda x: x[1])
    best_route, best_distance = population[0]
    return best_route, best_distance, best_distances

if __name__ == "__main__":
    all_best_distances = []
    all_run_times = []
    all_run_best_curves = []

    print("=== Genetic Algorithm TSP (30 cities) ===")

    # Run the GA multiple times to gather statistics
    for r in range(1, runs+1):
        start_time = time.time()
        print(f"\n-- Run {r} --")
        best_route, best_dist, best_curve = run_ga()
        end_time = time.time()
        run_time = end_time - start_time

        all_run_times.append(run_time)
        all_best_distances.append(best_dist)
        all_run_best_curves.append(best_curve)

        print(f"Best distance: {best_dist:.2f}")
        print(f"Execution time for this run: {run_time:.4f} seconds")

    # Compute statistics over all runs
    mean_best = statistics.mean(all_best_distances)
    median_best = statistics.median(all_best_distances)
    min_best = min(all_best_distances)
    max_best = max(all_best_distances)
    std_best = statistics.pstdev(all_best_distances) if len(all_best_distances) > 1 else 0.0

    mean_time = statistics.mean(all_run_times)
    median_time = statistics.median(all_run_times)
    min_time = min(all_run_times)
    max_time = max(all_run_times)
    std_time = statistics.pstdev(all_run_times) if len(all_run_times) > 1 else 0.0

    print("\n=== Summary (over all runs) ===")
    print(f"Mean best distance: {mean_best:.2f}")
    print(f"Median best distance: {median_best:.2f}")
    print(f"Min best distance: {min_best:.2f}")
    print(f"Max best distance: {max_best:.2f}")
    print(f"Std dev best distance: {std_best:.2f}")

    print("\n=== Execution Time Summary ===")
    print(f"Mean execution time: {mean_time:.4f} s")
    print(f"Median execution time: {median_time:.4f} s")
    print(f"Min execution time: {min_time:.4f} s")
    print(f"Max execution time: {max_time:.4f} s")
    print(f"Std dev execution time: {std_time:.4f} s")

    # Plot the average convergence curve (mean best distance per generation)
    avg_curve = [statistics.mean(gen_vals) for gen_vals in zip(*all_run_best_curves)]
    plt.figure()
    plt.title("GA Convergence (Mean Best Distance per Generation)")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.plot(avg_curve, label="Mean Best Distance (over runs)")
    plt.grid(True)
    plt.legend()
    plt.savefig("ga_convergence.png")
    plt.close()

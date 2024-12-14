import random
import statistics
import time
import matplotlib.pyplot as plt

# For consistency, we use a different seed and new city set for the PSO
# If you want the exact same set as GA, you can re-use the same `cities` and `distance_matrix`.
# Here, we just create a new instance for demonstration.
random.seed(124)
num_cities = 30
cities = [(random.random() * 100, random.random() * 100) for _ in range(num_cities)]

# Distance matrix for the new set of cities for PSO
distance_matrix = [[0] * num_cities for _ in range(num_cities)]
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            dx = cities[i][0] - cities[j][0]
            dy = cities[i][1] - cities[j][1]
            distance_matrix[i][j] = (dx**2 + dy**2)**0.5

# PSO parameters
num_particles = 60
iterations = 100
c1 = 2.0  # Cognitive parameter
c2 = 2.0  # Social parameter
w = 0.7   # Inertia weight
runs = 10

def calculate_distance(route):
    """Calculate the total distance of a given route."""
    dist = 0
    for i in range(len(route)):
        dist += distance_matrix[route[i]][route[(i+1) % num_cities]]
    return dist

def create_random_solution():
    """Create a random solution (permutation of cities) for PSO."""
    sol = list(range(num_cities))
    random.shuffle(sol)
    return sol

def generate_swaps(from_sol, to_sol):
    """
    Generate a list of swaps that transform `from_sol` into `to_sol`.
    This effectively treats the 'velocity' in PSO as a list of swaps.
    """
    fs = from_sol[:]
    swaps = []
    for i in range(num_cities):
        if fs[i] != to_sol[i]:
            correct_idx = fs.index(to_sol[i])
            swaps.append((i, correct_idx))
            fs[i], fs[correct_idx] = fs[correct_idx], fs[i]
    return swaps

def apply_swaps(solution, swaps):
    """Apply a sequence of swaps to a solution to produce a new solution."""
    sol = solution[:]
    for (i, j) in swaps:
        sol[i], sol[j] = sol[j], sol[i]
    return sol

def run_pso():
    """
    Run one instance of the Particle Swarm Optimization:
    - Initialize particles and velocities
    - For each iteration, update particles' positions by applying swaps from pbest and gbest
    - Track and return the best solution found
    """
    particles = []
    velocities = []  # Each particle's velocity is represented as a list of swaps
    pbest = []        # Personal best positions
    pbest_scores = [] # Personal best scores

    # Initialize particles with random solutions
    for _ in range(num_particles):
        sol = create_random_solution()
        score = calculate_distance(sol)
        particles.append(sol)
        velocities.append([])
        pbest.append(sol[:])
        pbest_scores.append(score)

    # Global best (gbest)
    gbest_score = min(pbest_scores)
    gbest = pbest[pbest_scores.index(gbest_score)][:]

    best_distances = []

    for it in range(1, iterations+1):
        fitness_list = []
        for i in range(num_particles):
            current = particles[i]

            # Generate swaps towards pbest (with some probability)
            if random.random() < 0.9:
                p_swaps = generate_swaps(current, pbest[i])
            else:
                p_swaps = []

            # Generate swaps towards gbest (with some probability)
            if random.random() < 0.9:
                g_swaps = generate_swaps(current, gbest)
            else:
                g_swaps = []

            old_vel = velocities[i]
            # Keep a portion of the old velocity (inertia)
            keep_count = int(w * len(old_vel))
            new_velocity = old_vel[:keep_count] + p_swaps + g_swaps

            # Apply the new velocity (swaps) to the current solution
            new_solution = apply_swaps(current, new_velocity)
            new_score = calculate_distance(new_solution)

            # Update particle
            particles[i] = new_solution
            velocities[i] = new_velocity

            # Update personal best if improved
            if new_score < pbest_scores[i]:
                pbest_scores[i] = new_score
                pbest[i] = new_solution[:]

            # Update global best if improved
            if new_score < gbest_score:
                gbest_score = new_score
                gbest = new_solution[:]

            fitness_list.append(new_score)

        # Track the best solution found this iteration
        best_this_iter = min(fitness_list)
        best_distances.append(best_this_iter)

    return gbest, gbest_score, best_distances

if __name__ == "__main__":
    all_best_distances = []
    all_run_times = []
    all_run_best_curves = []

    print("=== Particle Swarm Optimization TSP (30 cities) ===")

    # Run the PSO multiple times to gather statistics
    for r in range(1, runs+1):
        start_time = time.time()
        print(f"\n-- Run {r} --")
        gbest, gbest_score, best_curve = run_pso()
        end_time = time.time()
        run_time = end_time - start_time

        all_run_times.append(run_time)
        all_best_distances.append(gbest_score)
        all_run_best_curves.append(best_curve)

        print(f"Best distance: {gbest_score:.2f}")
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

    # Plot the average convergence curve (mean best distance per iteration)
    avg_curve = [statistics.mean(gen_vals) for gen_vals in zip(*all_run_best_curves)]
    plt.figure()
    plt.title("PSO Convergence (Mean Best Distance per Iteration)")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.plot(avg_curve, label="Mean Best Distance (over runs)")
    plt.grid(True)
    plt.legend()
    plt.savefig("pso_convergence.png")
    plt.close()
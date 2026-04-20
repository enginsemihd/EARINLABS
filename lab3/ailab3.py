import numpy as np
import matplotlib.pyplot as plt


def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


class Chromosome:
    def __init__(self, length, array=None):
        self.length = length
        if array is None:
            # Generate random binary array (0s and 1s)
            self.array = np.random.randint(0, 2, size=length)
        else:
            self.array = np.array(array)

    def decode(self, lower_bound, upper_bound, aoi):
        # Extract the segment for one variable
        segment = self.array[lower_bound : upper_bound + 1]
        
        # Convert binary array segment to string, then to integer
        bit_str = "".join(str(bit) for bit in segment)
        decimal_val = int(bit_str, 2)
        
        # Calculate max possible value for this many bits
        num_bits = len(segment)
        max_val = (2 ** num_bits) - 1
        
        # Normalize to the Area of Interest (aoi)
        return min_max_norm(decimal_val, 0, max_val, aoi[0], aoi[1])

    def mutation(self, probability):
        # Go through each gene and flip it based on the probability
        for i in range(self.length):
            if np.random.rand() < probability:
                self.array[i] = 1 - self.array[i]  # flips 0 to 1, and 1 to 0

    def crossover(self, other):
        # Pick a random crossover point
        crossover_point = np.random.randint(1, self.length)
        
        # Create children by slicing and combining parents
        child1_array = np.concatenate((self.array[:crossover_point], other.array[crossover_point:]))
        child2_array = np.concatenate((other.array[:crossover_point], self.array[crossover_point:]))
        
        return Chromosome(self.length, child1_array), Chromosome(self.length, child2_array)


# Group D Objective Function: Styblinski-Tang
def objective_function(*args):
    # The formula is 0.5 * sum(x^4 - 16x^2 + 5x) for each dimension
    result = 0
    for x in args:
        result += 0.5 * ((x ** 4) - 16 * (x ** 2) + 5 * x)
    return result


class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi,
                 population_size=100, tournament_size=2, mutation_probability=0.05,
                 crossover_probability=0.8, num_steps=50):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_length = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps
        
        # Initialize the population
        self.population = [Chromosome(self.chromosome_length) for _ in range(self.population_size)]

    def eval_objective_func(self, chromosome):
        args = []
        for i in range(self.obj_func_num_args):
            lower_bound = i * self.bits_per_arg
            upper_bound = (i + 1) * self.bits_per_arg - 1
            decoded_val = chromosome.decode(lower_bound, upper_bound, self.aoi)
            args.append(decoded_val)
            
        return self.objective_function(*args)

    def tournament_selection(self):
        parents = []
        # Pre-evaluate fitness to save time
        fitness_scores = [self.eval_objective_func(ind) for ind in self.population]
        
        for _ in range(self.population_size):
            # Pick random individuals for the tournament
            tournament_indices = np.random.choice(self.population_size, self.tournament_size, replace=False)
            
            # Find the best (minimum) among the chosen ones
            best_idx = tournament_indices[0]
            for idx in tournament_indices:
                if fitness_scores[idx] < fitness_scores[best_idx]:
                    best_idx = idx
            
            parents.append(self.population[best_idx])
            
        return parents

    def reproduce(self, parents):
        new_population = []
        
        # Step through parents two at a time
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            # Handle odd population sizes
            parent2 = parents[i + 1] if (i + 1) < self.population_size else parents[i]
            
            # Crossover
            if np.random.rand() < self.crossover_probability:
                child1, child2 = parent1.crossover(parent2)
            else:
                # No crossover, pass exact copies
                child1 = Chromosome(self.chromosome_length, parent1.array.copy())
                child2 = Chromosome(self.chromosome_length, parent2.array.copy())
                
            # Mutation
            child1.mutation(self.mutation_probability)
            child2.mutation(self.mutation_probability)
            
            new_population.append(child1)
            new_population.append(child2)
            
        self.population = new_population[:self.population_size]

    def plot_func(self, trace):
        # Create a grid of points for the contour plot
        x = np.linspace(self.aoi[0], self.aoi[1], 100)
        y = np.linspace(self.aoi[0], self.aoi[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Evaluate function on the grid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.objective_function(X[i, j], Y[i, j])
                
        plt.figure(figsize=(8, 6))
        # Plot the contour
        plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Objective Value')
        
        # Extract trace points
        trace_x = [pt[0] for pt in trace]
        trace_y = [pt[1] for pt in trace]
        
        # Plot the trace with colors progressing from dark red to light red
        colors = np.linspace(0, 1, len(trace))
        plt.scatter(trace_x, trace_y, c=colors, cmap='Reds_r', edgecolor='black', s=50, zorder=5)
        
        plt.title('Styblinski-Tang Function - GA Trace')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    def run(self):
        trace = []
        
        for step in range(self.num_steps):
            # Evaluate current population
            fitness_scores = [self.eval_objective_func(ind) for ind in self.population]
            
            # Find the best individual in the current generation
            best_idx = np.argmin(fitness_scores)
            best_individual = self.population[best_idx]
            
            # Decode the best individual to record its coordinates
            best_args = []
            for i in range(self.obj_func_num_args):
                lower_bound = i * self.bits_per_arg
                upper_bound = (i + 1) * self.bits_per_arg - 1
                val = best_individual.decode(lower_bound, upper_bound, self.aoi)
                best_args.append(val)
                
            trace.append(best_args)
            
            # Print progress every 10 steps
            if step % 10 == 0 or step == self.num_steps - 1:
                print(f"Generation {step:02d} | Best f(x) = {fitness_scores[best_idx]:.4f} at x = {[round(a, 4) for a in best_args]}")
            
            # Create next generation
            parents = self.tournament_selection()
            self.reproduce(parents)
            
        self.plot_func(trace)
        return trace[-1] # Return the final best coordinates


# --- Execution Example ---
if __name__ == "__main__":
    print("Starting Genetic Algorithm for Group D (Styblinski-Tang)...\n")
    
    # Styblinski-Tang is usually evaluated in the domain [-5, 5]
    ga = GeneticAlgorithm(
        chromosome_length=40,  # 20 bits per variable provides good precision
        obj_func_num_args=2,
        objective_function=objective_function,
        aoi=[-5, 5],
        population_size=100,
        tournament_size=3,
        mutation_probability=0.05,
        crossover_probability=0.8,
        num_steps=50
    )
    
    best_solution = ga.run()
    print(f"\nFinal best solution found: x1 = {best_solution[0]:.4f}, x2 = {best_solution[1]:.4f}")
    print(f"Function value at this point: {objective_function(*best_solution):.4f}")
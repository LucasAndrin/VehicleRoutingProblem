from matplotlib import pyplot as plt
import networkx as nx
import random as rd
import numpy as np

class VRPGeneticSolver:
    def __init__(self, clients: int, vehicles: int, capacity: int, gen_size: int, pop_size: int, mut_rate: float):
        self.clients = clients
        self.vehicles = vehicles
        self.capacity = capacity
        self.gen_size = gen_size
        self.pop_size = pop_size
        self.mut_rate = mut_rate

        # Set random seeds for reproducibility
        np.random.seed(42)
        rd.seed(42)

        # Generate graph and demands
        self.graph = self._gen_graph()
        self.demands = self._gen_demands()
        
    def _gen_graph(self):
        """
        Generates a complete graph with random edge weights.

        Returns:
            Graph: Generated graph
        """
        graph = nx.complete_graph(self.clients + 1, nx.DiGraph)  # Includes the depot (node 0)
        for u, v in graph.edges():
            graph[u][v]['weight'] = np.random.randint(1, 20)
        return graph

    def _gen_demands(self):
        """
        Generates random demands for clients.
        
        Returns:
            list[int]: List of random demands
        """
        return [0] + [np.random.randint(1, 10) for _ in range(self.clients)]

    def _gen_solution(self):
        """
        Generates a random solution of paths, ensuring the capacity is not exceeded.
        
        Returns:
            list[list[int]]: Generated solution
        """
        nodes = list(range(1, self.clients + 1))
        rd.shuffle(nodes)
        
        # Sort nodes by demand to avoid uneven capacity distribution
        nodes.sort(key=lambda node: self.demands[node], reverse=True)
        
        paths = [[] for _ in range(self.vehicles)]
        capacities = [0] * self.vehicles
        
        for node in nodes:
            # Find the vehicle with the lowest load that can accommodate the current demand
            possible_vehicles = [i for i in range(self.vehicles) if capacities[i] + self.demands[node] <= self.capacity]
            if not possible_vehicles:
                # If no vehicle can accommodate the node, skip to the next one
                continue
            vehicle = min(possible_vehicles, key=lambda i: capacities[i])  # Choose the vehicle with the least load
            paths[vehicle].append(node)
            capacities[vehicle] += self.demands[node]
        
        return paths

    def _gen_population(self):
        """Generate a random population of solutions

        Returns:
            list[list[list[int]]]: Generated population
        """
        return [self._gen_solution() for _ in range(self.pop_size)]
        

    def show_graph(self):
        """Show current graph of vrp"""
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=400, font_size=10)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        plt.show()

    def evaluate(self, solution):
        """
        Calculates the total cost of a solution
        and applies penalties for capacity violations

        Args:
            solution (list[list[int]]): List of paths for every vehicle to be evaluated

        Returns:
            float: Result of the evaluation
        """
        total_cost = 0
        for path in solution:
            if not path:
                continue

            cost = 0
            capacity = 0
            prev_node = 0  # Start at the depot

            for node in path:
                cost += self.graph[prev_node][node]['weight']
                capacity += self.demands[node]
                prev_node = node
            cost += self.graph[prev_node][0]['weight']  # Return to depot

            if capacity > self.capacity:
                return float('inf')  # Penalty for exceeding capacity
            total_cost += cost
        return total_cost

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent solutions
        and returns it's result

        Args:
            solution1 (list[list[int]]): First solution
            solution2 (list[list[int]]): Secondary solution

        Returns:
            list[list[int]]: New solution
        """
        new_solution = []
        used = set()
        for path in parent1:
            new_path = [node for node in path if node not in used]
            used.update(new_path)
            new_solution.append(new_path)

        for path in parent2:
            for node in path:
                if node not in used:
                    for new_path in new_solution:
                        if len(new_path) < self.clients // self.vehicles:
                            new_path.append(node)
                            used.add(node)
                            break

        return new_solution

    def mutate(self, solution):
        """
        MUtates a solution by swapping nodes between paths.

        Args:
            solution (list[list[int]]): Solution
        """
        for _ in range(len(solution)):
            if rd.random() < self.mut_rate:
                path1, path2 = rd.sample(solution, 2)
                if path1 and path2:
                    i, j = rd.randint(0, len(path1) - 1), rd.randint(0, len(path2) - 1)
                    path1[i], path2[j] = path2[j], path1[i]

    def solve(self):
        """
        Solve the VRP problem using Genetic Algorithm

        Returns:
            tuple[float, list[list[int]]]: Tuple of respectively the best cost and the best solution
        """
        
        # Generate initial population
        pop = self._gen_population()

        for gen in range(self.gen_size):
            # Evaluate population
            fit = [(self.evaluate(sol), sol) for sol in pop]
            fit.sort(key=lambda x: x[0])

            # Select the best solutions
            pop = [sol for _, sol in fit[:self.pop_size // 2]]

            # Crossover
            offspring = []
            for _ in range(self.pop_size // 2):
                parent1, parent2 = rd.sample(pop, 2)
                offspring.append(self.crossover(parent1, parent2))

            pop.extend(offspring)

            # Mutation
            for sol in pop:
                self.mutate(sol)

            # Best solution of the generation
            best_cost, best_solution = fit[0]
            
            if best_cost < 70:
                print(f'Generation {gen + 1}: Best Cost = {best_cost}, Best Solution = {best_solution}')
                

        # Final result
        fit.sort(key=lambda x: x[0])
        best_cost, best_solution = fit[0]
        print("\nBest Solution:")
        for i, path in enumerate(best_solution):
            print(f"Vehicle {i + 1}: {path}")
        print(f"Total Cost: {best_cost}")

if __name__ == "__main__":
    solver = VRPGeneticSolver(
        clients=10,
        vehicles=3,
        capacity=15,
        gen_size=100,
        pop_size=80,
        mut_rate=0.2
    )
    
    solver.solve()
    print(f'Demands: {solver.demands}')
    
    solver.show_graph()

import matplotlib.pyplot as plt
import networkx as nx
import random as rd
import numpy as np

class Client:
    def __init__(self, address: int, demand: int):
        self.address = address
        self.demand = demand

class Vehicle:
    def __init__(self, fuel: int, capacity: int):
        self.fuel = fuel
        self.capacity = capacity

class VRPSolver:
    def __init__(self, graph: nx.DiGraph, clients: list[Client], vehicles: list[Vehicle], gen_size: int, pop_size: int, mut_rate: float):
        self.graph = graph
        self.clients = clients
        self.vehicles = vehicles
        self.gen_size = gen_size
        self.pop_size = pop_size
        self.mut_rate = mut_rate

    def _gen_solution(self):
        """
        Generates a random solution of paths, ensuring the capacity is not exceeded.

        Returns:
            list[list[int]]: Generated solution
        """
        nodes = self.clients[1:]  # Exclui o depósito
        rd.shuffle(nodes)

        paths = [[] for _ in self.vehicles]
        capacities = [0] * len(self.vehicles)

        for client in nodes:
            possible_vehicles = [i for i in range(len(self.vehicles)) if capacities[i] + client.demand <= self.vehicles[i].capacity]
            if possible_vehicles:
                vehicle = min(possible_vehicles, key=lambda i: capacities[i])
                paths[vehicle].append(client.address)
                capacities[vehicle] += client.demand

        return paths

    def _gen_population(self):
        """Generate a random population of solutions

        Returns:
            list[list[list[int]]]: Generated population
        """
        return [self._gen_solution() for _ in range(self.pop_size)]

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
            prev_node = 0

            for node in path:
                cost += self.graph[prev_node][node]['weight']
                prev_node = node
            cost += self.graph[prev_node][0]['weight']
            total_cost += cost
        return total_cost

    def crossover(self, solution1, solution2):
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
        for path in solution1:
            new_path = [node for node in path if node not in used]
            used.update(new_path)
            new_solution.append(new_path)
        for path in solution2:
            for node in path:
                if node not in used:
                    for new_path in new_solution:
                        if len(new_path) < len(self.clients) // len(self.vehicles):
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
        best_cost = float('inf')
        best_solution = []

        for gen in range(self.gen_size):
            fit = [(self.evaluate(sol), sol) for sol in pop]
            fit.sort(key=lambda x: x[0])

            top_cost, top_sol = fit[0]
            if top_cost < best_cost:
                best_cost = top_cost
                best_solution = top_sol
                print(f'Generation {gen + 1}: Best Cost = {best_cost}')

            pop = [sol for _, sol in fit[:self.pop_size // 2]]
            offspring = [self.crossover(rd.choice(pop), rd.choice(pop)) for _ in range(self.pop_size // 2)]
            pop.extend(offspring)

            for sol in pop:
                self.mutate(sol)

        return best_cost, best_solution

    def draw_graph(self, best_solution = None):
        """
        Draw graph using plt lib and

        Args:
            best_solution (list[list[int]]): Best Solution
        """
        pos = nx.spring_layout(self.graph, seed=123)
        labels = {node: f"{node}" for node in self.graph.nodes}

        plt.figure(figsize=(8, 6))
        nx.draw(self.graph, pos, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray', node_size=700, font_size=12, style='dashed')

        edge_labels = {(u, v): f"{self.graph[u][v]['weight']}" for u, v in self.graph.edges()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10)

        nx.draw_networkx_nodes(self.graph, pos, nodelist=[0], node_color='red', node_size=900)

        if best_solution != None:
            colors = ['b', 'g', 'm', 'c', 'y']
            for i, path in enumerate(best_solution):
                edges = [(0, path[0])] + [(path[j], path[j + 1]) for j in range(len(path) - 1)] + [(path[-1], 0)]
                nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color=colors[i % len(colors)], width=2.5)

        plt.title("Best VRP Solution")
        plt.show()

if __name__ == "__main__":
    SEED = 123
    CLIENTS = 10
    VEHICLES = 3
    MAX_DEMAND = 10
    MAX_CAPACITY = 15

    rd.seed(SEED)
    np.random.seed(SEED)

    clients = [Client(0, 0)] + [Client(address, rd.randint(1, MAX_DEMAND)) for address in range(1, CLIENTS + 1)]
    vehicles = [Vehicle(fuel=100, capacity=MAX_CAPACITY) for _ in range(VEHICLES)]

    graph = nx.complete_graph(CLIENTS + 1, nx.DiGraph)
    for u, v in graph.edges():
        graph[u][v]['weight'] = np.random.randint(1, 20)

    solver = VRPSolver(graph, clients, vehicles, gen_size=200, pop_size=100, mut_rate=0.2)
    best_cost, best_solution = solver.solve()
    print(f'Best Cost: {best_cost}')
    print(f'Best Solution: {best_solution}')

    solver.draw_graph(best_solution)
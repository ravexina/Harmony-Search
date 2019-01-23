import numpy as np
from matplotlib import pyplot as plt


class HarmonySearch:

    def __init__(self, N, M, HMCR, PAR, L=(0, 1), max_iter=100, debug=False, end_condition=None):
        """
        :param N: Number of instruments (Dimension)
        :param M: The size of Harmony memory (Population size)
        :param HMCR: Harmony memory consideration rate
        :param PAR: Pitch adjustment rate
        :param L: Possible range of notes
        """
        self.instruments = N  # Dimension
        self.memory_size = M  # Population size
        self.HMCR = HMCR  # Harmony memory consideration rate
        self.PAR = PAR  # Pitch adjust rate
        self.max_iter = max_iter

        self.end_condition = end_condition
        self.debug = debug # Should I print HM in each iter?
        self.lower, self.upper = L

        # Harmony Memory (X)
        self.HM = None

        # Cost of each harmony
        self.costs = None

        # Should be set by user
        self.cost_func = None

        # For piloting purpose
        self.iterations_avg_costs = []
        self.best_so_far = []

    def solve(self):
        self.init()

        for _ in range(self.max_iter):
            if self.debug:
                print(self.HM)
                print('-'*20)

            self.evaluate_costs()
            self.update_best_so_far()
            self.iterations_avg_costs.append(sum(self.costs) / len(self.costs))

            new_harmony = self.improvise()
            self.exchange(new_harmony)

            if self.end_condition is not None:
                if self.end_condition == self.best_cost():
                    break
        else:
            if self.end_condition is not None:
                print('Can\'t converge !')

    def init(self):
        self.HM = np.random.randint(self.lower, self.upper, (self.memory_size, self.instruments))

    # Evaluate a single harmony
    def evaluate(self, args):
        return self.cost_func(args)

    # Evaluate all harmonies in HM
    def evaluate_costs(self):
        self.costs = [self.evaluate(x) for x in self.HM]

    # Returns the index of worst hae
    def best_cost(self):
        return min(self.costs)

    def worst_cost(self):
        return max(self.costs)

    def update_best_so_far(self):
        best_cost = self.best_cost()

        if len(self.best_so_far) == 0:
            self.best_so_far.append(best_cost)
            return

        if best_cost < self.best_so_far[-1]:
            self.best_so_far.append(best_cost)
        else:
            self.best_so_far.append(self.best_so_far[-1])

    @staticmethod
    def adjust(pitch):
        # 50% Going up or down
        if np.random.uniform(0, 1) > 0.5:
            return pitch + 1
        return pitch - 1

    def improvise(self):

        new_harmony = np.full(self.instruments, None)

        for i in range(self.instruments):
            mem_rate = np.random.uniform(0, 1)

            # Use harmony memory or a random pitch from available ones
            if mem_rate > self.HMCR:
                # Random
                new_harmony[i] = np.random.randint(self.lower, self.upper)
            else:
                # From Harmony Memory
                new_harmony[i] = np.random.choice(self.HM[:, i])

            # Pitch adjusting
            if np.random.uniform(0, 1) <= self.PAR:
                new_harmony[i] = self.adjust(new_harmony[i])

        return new_harmony

    # Exchanges the new harmony with worst if only the new
    # harmony was better than of worst one
    def exchange(self, new_harmony):
        if self.evaluate(new_harmony) < self.worst_cost():
            worst_index = self.costs.index(self.worst_cost())
            self.HM[worst_index] = new_harmony

    def results(self):
        self.evaluate_costs()
        best_cost = self.best_cost()
        best_index = self.costs.index(best_cost)
        print(f'Best answer is at: {self.HM[best_index]} with the cost of: {self.costs[best_index]}')

    def plot_results(self):
        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(self.iterations_avg_costs, color='darkviolet')
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(self.best_so_far, color='pink')
        plt.show()

import numpy as np
from fish import Fish


class FishSchoolSearch:
    def __init__(self, iterations_num: int, population_size: int, individual_step: float,
                 selected_func, selected_optimum: bool, coord_count: int, weight_scale: float, min_range: float,
                 max_range: float):
        self._iterations_num = iterations_num
        self._population_size = population_size
        self._selected_optimum = selected_optimum
        self._init_individual_step = individual_step
        self._individual_step = individual_step
        self._init_collective_step = individual_step * 2
        self._coord_count = coord_count
        self._sum_weight = 0
        self._sum_weight_old = population_size * weight_scale / 2
        self._fishes = []
        for _ in range(population_size):
            self._fishes.append(Fish(selected_func, selected_optimum, coord_count, weight_scale, min_range, max_range))
        self._coords = []
        for fish in self.fishes:
            self._coords.append(fish.init_position)

    @property
    def fishes(self):
        return self._fishes

    def get_sum_weight(self):
        return sum(fish.weight for fish in self._fishes)

    def total_migration_step(self):
        dividend = divisor = 0
        for i in range(self._population_size):
            dividend += (self._fishes[i].is_position - self._fishes[i].init_position) * self._fishes[i].delta_fitness
            divisor += self._fishes[i].delta_fitness
        return dividend / divisor if divisor != 0 else 0

    def calculate_barycentre(self):
        dividend = np.zeros(self._coord_count)
        divisor = 0
        for i in range(self._population_size):
            dividend += self._fishes[i].cis_position * self._fishes[i].weight
            divisor += self._fishes[i].weight
        return dividend / divisor if divisor != 0 else np.zeros(self._coord_count)

    def prepare_new_iteration(self):
        self._sum_weight_old = self._sum_weight
        [fish.prepare_new_iteration() for fish in self._fishes]
        self._individual_step -= self._init_individual_step / self._iterations_num
        self._init_collective_step = self._individual_step * 2

    def select_best_fish(self, selected_optimum):
        if selected_optimum:
            best_fish = min(self.fishes, key=lambda fish: fish.init_fitness)
        else:
            best_fish = max(self.fishes, key=lambda fish: fish.init_fitness)
        return best_fish

    def run(self):
        neval = 0
        for i in range(self._iterations_num):
            # print(f"Iteration num: {i}")
            [fish.individual_swim(self._individual_step) for fish in self.fishes]
            max_delta_fitness = max(fish.delta_fitness for fish in self.fishes)
            [fish.feeding(max_delta_fitness) for fish in self.fishes]
            migration_step = self.total_migration_step()
            [fish.collective_instinctive_swim(migration_step) for fish in self.fishes]
            barycentre = self.calculate_barycentre()
            self._sum_weight = self.get_sum_weight()
            [fish.collective_volitive_swim(barycentre, self._init_collective_step, self._sum_weight,
                                           self._sum_weight_old) for fish in self.fishes]
            # for i, fish in enumerate(self.fishes):
            #     if i == 0:
            #         print(fish)
            self.prepare_new_iteration()
            for fish in self.fishes:
                self._coords.append(fish.init_position)

        best_fish = self.select_best_fish(self._selected_optimum)
        neval = sum(fish.neval for fish in self.fishes)
        return best_fish, self._coords, neval

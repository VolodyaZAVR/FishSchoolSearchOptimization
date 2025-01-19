import numpy as np
from func import TestFunc


class Fish:
    def __init__(self, selected_func, selected_optimum: bool, coord_count: int, weight_scale: float, min_range: float,
                 max_range: float):
        self._func = selected_func
        self._selected_optimum = selected_optimum
        self._coord_count = coord_count
        self._weightScale = weight_scale
        self._weight = 0
        self._init_position = np.random.uniform(min_range, max_range, coord_count)  # init position
        self._is_position = np.zeros(coord_count)  # individual swim position
        self._cis_position = np.zeros(coord_count)  # collective instinctive swim position
        self._cvs_position = np.zeros(coord_count)  # collective volitive swim position
        self._init_fitness = 0  # init fitness
        self._is_fitness = 0  # individual swim fitness
        self._delta_pos = np.zeros(coord_count)
        self._delta_fitness = 0
        self._neval = 0
        self._min = min_range
        self._max = max_range
        self._init_fitness = self.calculate_fitness(self._init_position)

    def __str__(self):
        return (f"\nFish result: init pos {self._init_position}, IS pos {self._is_position}, "
                f"CIS pos {self._cis_position}, CVS pos {self._cvs_position}, init fitness {self._init_fitness}, "
                f"IS fitness {self._is_fitness}, delta pos {self._delta_pos}, delta fitness {self._delta_fitness}\n"
                f"weight {self._weight}\n")

    @property
    def weight(self):
        return self._weight

    @property
    def init_position(self):
        return self._init_position

    @property
    def is_position(self):
        return self._is_position

    @property
    def cis_position(self):
        return self._cis_position

    @property
    def delta_fitness(self):
        return self._delta_fitness

    @property
    def init_fitness(self):
        return self._init_fitness

    @property
    def neval(self):
        return self._neval

    def calculate_fitness(self, position):
        fitness = TestFunc.calculate(self._func, position)
        self._neval += 1
        return fitness

    def individual_swim(self, individual_step: float):
        self._is_position = self._init_position + np.random.uniform(-1, 1, self._coord_count) * individual_step
        self._delta_pos = self._is_position - self._init_position
        self.evaluate_fitness(self._selected_optimum)

    def calc_delta_fitness(self, selected_optimum):
        """
        params: selected_optimum: True if searching minimum False is maximum
        """
        delta = 0
        if selected_optimum:
            delta = self._init_fitness - self._is_fitness
        else:
            delta = self._is_fitness - self._init_fitness
        return delta

    def evaluate_fitness(self, selected_optimum):
        self._is_fitness = self.calculate_fitness(self._is_position)
        self._delta_fitness = self.calc_delta_fitness(selected_optimum)

        if (np.any(self._is_position < self._min) or np.any(self._is_position > self._max)) or (
                self._delta_fitness < 0):
            self._delta_fitness = 0
            self._delta_pos.fill(0)
            self._is_fitness = self._init_fitness
            self._is_position = self._init_position

    def feeding(self, max_delta_fitness):
        if max_delta_fitness != 0:
            self._weight += self._delta_fitness / max_delta_fitness
        self._weight = np.clip(self._weight, 1, self._weightScale)

    def collective_instinctive_swim(self, migration_step):
        self._cis_position = self._is_position + migration_step

    @staticmethod
    def euclid_dist(n1, n2):
        result = 0
        if len(n1) != len(n2):
            raise ValueError("Not equal dimensions")
        for i in range(len(n1)):
            result += np.power(n1[i] - n2[i], 2)
        return np.sqrt(result)

    def collective_volitive_swim(self, barycentre, collective_step, sum_weight, sum_weight_old):
        dist = self.euclid_dist(self._cis_position, barycentre)

        if sum_weight > sum_weight_old:
            self._cvs_position = self._cis_position - collective_step * np.random.uniform(0, 1) * (
                    self._cis_position - barycentre) / dist
        else:
            self._cvs_position = self._cis_position + collective_step * np.random.uniform(0, 1) * (
                    self._cis_position - barycentre) / dist

    def prepare_new_iteration(self):
        self._init_position = self._cvs_position
        self._init_fitness = self.calculate_fitness(self._init_position)

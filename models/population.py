import numpy as np
from models import creature


class Population:
    def __init__(self, pop_size: int, gene_count: int) -> None:
        self.creatures = [creature.Creature(gene_count) for _ in range(pop_size)]

    @staticmethod
    def get_fitness_map(fits):
        fitmap = []
        total = 0
        for f in fits:
            total += f
            fitmap.append(total)
        return fitmap

    def select_parent(fitmap):
        r = np.random.rand() * fitmap[-1]
        for i, f in enumerate(fitmap):
            if r < f:
                return i

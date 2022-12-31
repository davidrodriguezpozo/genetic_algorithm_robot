from typing import List

import numpy as np
from models import creature, genome, population, simulation


class GA:
    def run_ga(self):
        pop = population.Population(pop_size=10, gene_count=3)
        sim = simulation.ThreadedSim(pool_size=4)
        for iteration in range(10):
            sim.eval_population(pop, 2400)
            fits = [cr.get_distance_travelled() for cr in pop.creatures]
            fittest = np.max(fits)
            elite = None
            for cr in pop.creatures:
                if cr.get_distance_travelled() == fittest:
                    elite = cr
                    break

            links = [len(cr.get_expanded_links()) for cr in pop.creatures]
            print(
                iteration,
                "- fittest:",
                np.round(np.max(fits), 3),
                "mean:",
                np.round(np.mean(fits), 3),
                "mean links",
                np.round(np.mean(links)),
                "max links",
                np.round(np.max(links)),
            )

            fit_map = population.Population.get_fitness_map(fits)
            new_creatures: List[creature.Creature] = []
            for _ in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                dna = genome.GenomeGenerator.crossover(p1.dna, p2.dna)
                dna = genome.GenomeGenerator.point_mutate(dna, rate=0.25, amount=0.25)
                dna = genome.GenomeGenerator.shrink_mutate(dna, rate=0.25)
                dna = genome.GenomeGenerator.grow_mutate(dna, rate=0.25)
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)
            new_creatures[0] = elite
            pop.creatures = new_creatures


if __name__ == "__main__":
    ga = GA()
    ga.run_ga()

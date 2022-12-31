from models import population, simulation


def test_proc():
    pop = population.Population(pop_size=20, gene_count=3)
    tsim = simulation.ThreadedSim(pool_size=4)
    tsim.eval_population(pop, 2400)
    dists = [cr.get_distance_travelled() for cr in pop.creatures]
    print(dists)

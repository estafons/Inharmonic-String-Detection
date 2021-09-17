from deap import base, creator, tools
from genetic_tools import *
from constants_parser import Constants

def delete_extra_info(tablature : Tablature):
    for tab_instance in tablature.tablature:
        tab_instance.note_audio = []

def genetic(inharmonic_tablature : Tablature, constants : Constants):
    ngen = constants.ngen
    cxpb = constants.cxpb
    mutpb = constants.mutpb
    pop = []
    
    delete_extra_info(inharmonic_tablature)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # create toolbox
    toolbox = base.Toolbox()


    toolbox.register("random_tab", get_random_tablature, inharmonic_tablature, constants)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                        toolbox.random_tab)

    toolbox.register("evaluate", evaluate, inharmonic_tablature = inharmonic_tablature, constants = constants)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual, constants.initial_pop)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, k =constants.no_of_parents, tournsize = constants.tournsize)#, #tournsize=3)
   # pool = multiprocessing.Pool()
   # toolbox.register("map", pool.map)#
    toolbox.register("map", map) # windows workarround. Cant use multiprocessing at the moment
    #create initial population
    pop = toolbox.population()
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    for g in range(ngen):
        print(g)

        offspring = toolbox.select(pop)
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < mutpb:

                toolbox.mutate(mutant, constants)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        offspring.extend(tools.selBest(pop, constants.parents_to_next_gen))
        selected = tools.selBest(offspring, constants.offspring_to_next_gen)
        pop[:] = selected

        res = [x.fitness.values for x in tools.selBest(selected, constants.end_no)] # extra termination condition. if first 100 are the same break and return
        if res.count(res[0]) == len(res):
            break

        #print([x.string for x in res])
    [winner] = tools.selBest(selected, 1)
    return winner, g

#tab = Tablature([(0.1, 110, 0),(0.1, 110, 1),(0.1, 110, 1), (0.1, 110, 1),(0.1, 110, 1),(0.1, 110, 1),(0.1, 110, 1),(0.1, 110, 1),(0.1, 110, 1),(0.1, 110, 1),(0.1, 110, 1),(0.1, 110, 1)], [], True)
#print(genetic(tab))
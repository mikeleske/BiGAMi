import random
from itertools import compress

def mutFlipOne(individual):
    """Aggregate function to flip one attribute of the input individual and return the
    mutant. With 0.5 probability each a random activate gene is disables or an inactive 
    gene is enabled.
    :param individual: Individual to be mutated.
    :returns: A tuple of one individual.
    """
    if random.random() < 0.75 and sum(individual) >= 3:
        return mutFlipOneOff(individual)
    else:
        return mutFlipOneOn(individual)

def mutFlipOneOff(individual):
    """Flip the value of the attributes of the input individual and return the
    mutant. The *individual* is expected to be a :term:`sequence` and the values of the
    attributes shall stay valid after the ``not`` operator is called on them.
    :param individual: Individual to be mutated.
    :returns: A tuple of one individual.
    """
    mask = list(compress(range(len(individual)), individual))
    individual[random.choice(mask)] = 0
    return individual,

def mutFlipOneOn(individual):
    """Flip the value of the attributes of the input individual and return the
    mutant. The *individual* is expected to be a :term:`sequence` and the values of the
    attributes shall stay valid after the ``not`` operator is called on them.
    :param individual: Individual to be mutated.
    :returns: A tuple of one individual.
    """
    mask = list(compress(range(len(individual)), individual))
    individual[random.choice(range(len(individual)))] = 1
    return individual,
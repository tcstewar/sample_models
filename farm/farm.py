"""Farm Planting

This is an agent model of farmers making planting decisions.

Requires FarmGame, available from https://github.com/tcstewar/farm_game
"""

import farm_model
import numpy as np

# Parameters
n_farms = 100                      # number of farms in simulation
n_steps = 20                       # number of time steps to simulate
# parameters for an external intervention affecting the local market
interv_local_time = 10             # time at which intervention starts
interv_local_premium_normal = 0.5  # amount of extra profit on normal produce
interv_local_premium_organic = 3   # amount of extra profit on organic produce
interv_local_cost = 10000          # public cost of intervention

# build the world
world = farm_model.eutopia.Eutopia(farm_count=n_farms, 
                                   rng=np.random.RandomState())

# create the list of external interventions
interventions = [
    farm_model.intervention.LocalMarketIntervention(
        interv_local_time, interv_local_premium_normal,
        interv_local_premium_organic, interv_local_cost)
    ]

results = []
for time in range(n_steps):
    # apply interventions
    for intervention in interventions:
        if time >= intervention.time:
            intervention.apply(world, time)

    # simulate the world
    world.step()

    # record data
    results.append(world.get_activity_count())

import pylab
for name in sorted(world.activities.keys()):
    pylab.plot([r.get(name,0) for r in results], label=name)
pylab.legend(loc='best')
pylab.show()





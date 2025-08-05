import os, shutil
import numpy as np
import time
from tqdm import trange

from models.physical_class.universe import Universe
from models.ticking_class.ticking_earth import TickingEarth
from models.ticking_class.ticking_sun import TickingSun

np.random.seed(0)

def simulation(grid_shape, nb_steps, backend="numpy"):
    iter_steps = np.zeros(nb_steps, dtype=float)
    compile_time = time.time()

    universe = Universe()
    universe.sun = TickingSun()  # Can be replaced with Sun()
    print("Running model with backend:", backend)
    print("Generating the earth...")
    universe.earth = TickingEarth(shape=grid_shape, backend=backend)
    universe.discover_everything()

    # Fills the earth with random GridChunk of water
    universe.earth.fill_with_water()

    print("Done.")
    print("Updating Universe 10 times...")
    print(universe)

    for i in trange(nb_steps):
        iter = time.time()
        universe.update_all()
        iter_steps.put(i, time.time() - iter)

    elapsed = time.time() - compile_time
    mean = np.mean(iter_steps)

    print(universe)
    print(f"Average time per step: {mean} seconds")
    print(f"Simulation took {elapsed} seconds")

    return elapsed, mean

if __name__ == "__main__":
    backend = "numpy"  # Change this to the desired backend
    grid_shape = (50, 50, 80)
    nb_steps = 50

    elapsed, mean = simulation(grid_shape, nb_steps, backend)

    if os.path.isdir(".gt_cache"):
        shutil.rmtree(".gt_cache")
    if os.path.isdir(".dacecache"):
        shutil.rmtree(".dacecache")

    os.system('say "finished"') # Notify end of simulation
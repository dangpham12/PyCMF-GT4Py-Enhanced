import matplotlib.pyplot as plt
import json
import numpy as np
def get_data(shape, type):
    path = f"../data/{shape}_{type}.json"
    # path = f"../data/{shape}_{type}_devicesync.json" # Uncomment this lien for device sync data
    with open(path, "r") as f:
        data = json.load(f)

    backends = data.keys()
    times = [data[backend]["total"] for backend in backends]
    mean_times = [data[backend]["mean"] for backend in backends]
    return backends, times, mean_times

def get_data_gpu(shape, type):
    path = f"../data/{shape}_{type}.json"
    with open(path, "r") as f:
        data = json.load(f)

    desired_backends = ["gt:gpu", "dace:gpu", "cuda"]
    backends = [backend for backend in data.keys() if backend in desired_backends]
    times = [data[backend]["total"] for backend in backends]
    mean_times = [data[backend]["mean"] for backend in backends]
    return backends, times, mean_times

def get_data_jax(shape):
    path = "../data/jax.json"
    with open(path, "r") as f:
        data = json.load(f)

    backend = "jax"
    shapes = data.keys()
    times = [data[shape]["total"] for shape in shapes]
    times2 = [data[shape]["total2"] for shape in shapes]
    mean_times = [data[shape]["mean"] for shape in shapes]
    mean_times2 = [data[shape]["mean2"] for shape in shapes]
    return backend, shapes, times, times2, mean_times, mean_times2

if __name__ == "__main__":
    canvas = [50, 400]
    data_type = ["float64", "float32"]
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

    backends, times, mean_times = get_data(canvas[0], data_type[1])
    backends2, times2, mean_times2 = get_data(canvas[1], data_type[1])
    backend_jax, shapes_jax, time_jax, time2_jax, mean_time_jax, mean_time2_jax = get_data_jax(canvas[0])
    bar_width = 0.4
    x = np.arange(len(shapes_jax))

    #plt.bar(x, mean_time_jax, width=bar_width, color=colors) # Uncomment this line to plot times/backend

    plt.bar(x - bar_width / 2, mean_time_jax, width=bar_width, label='With compilation')
    plt.bar(x + bar_width / 2, mean_time2_jax, width=bar_width, label="Without compilation") # Uncomment these lines to compare


    plt.xlabel('Shapes')
    plt.ylabel('Time (seconds)')
    plt.title(f'Iteration step time JAX 50x50 vs 400x400')
    plt.xticks(x, shapes_jax, fontsize=6, rotation=45)
    plt.legend()

    plt.show()
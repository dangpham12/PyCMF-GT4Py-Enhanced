import matplotlib.pyplot as plt
import json, re
import numpy as np
def get_data(shape, type):
    path = f"{shape}_{type}.json"
    with open(path, "r") as f:
        data = json.load(f)

    backends = data.keys()
    times = [data[backend]["total"] for backend in backends]
    mean_times = [data[backend]["mean"] for backend in backends]
    return backends, times, mean_times

if __name__ == "__main__":
    canvas = [50, 400]
    data_type = [np.float64, np.float32]
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    types = []
    for type in data_type:
        strip_word = re.search(r"\.(\w+)'", str(type))
        types.append(strip_word.group(1))

    backends, times, mean_times = get_data(canvas[1], data_type[0])
    backends2, times2, mean_times2 = get_data(canvas[1], data_type[1])

    bar_width = 0.4
    x = np.arange(len(backends))

    # plt.bar(x, mean_times, width=bar_width, color=colors) # Uncomment this line to plot times/backend

    plt.bar(x - bar_width / 2, mean_times, width=bar_width, label='float64')
    plt.bar(x + bar_width / 2, mean_times2, width=bar_width, label='float32') # Uncomment these lines to compare


    plt.xlabel('Backend')
    plt.ylabel('Time (seconds)')
    plt.title(f'Iteration step time {types[0]} vs {types[1]}')
    plt.xticks(x, backends, fontsize=6, rotation=45)
    plt.legend()

    plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Data for different image processing operations
operations = ['Histogram Equalization', 'Gaussian Blur', 'Edge Detection', 'Image Scaling', 'Color Space Conversion', 'Global Thresholding']
local_times = [0.0045874, 0.0359784, 0.04376, 0.051951, 0.0169314, 0.0035426]
parallel_2_times = [0.0033001, 0.0297597, 0.0277985, 0.0224902, 0.0057443, 0.0009948]
parallel_5_times = [0.0067645, 0.0344425, 0.0444963, 0.0068001, 0.0092409, 0.001739]
parallel_10_times = [0.012726, 0.05104, 0.0630615, 0.006213, 0.0092704, 0.0015721]
parallel_20_times = [0.015526, 0.0853701, 0.0832914, 0.0033031, 0.009499, 0.0016014]
parallel_50_times = [0.0448676, 0.032333, 0.090893, 0.0014168, 0.0101297, 0.0015828]
parallel_70_times = [0.0488115, 0.0186183, 0.0311254, 0.0010316, 0.0083691, 0.0015513]

# Combine the data for parallel processing times
parallel_times = [parallel_2_times, parallel_5_times, parallel_10_times, parallel_20_times, parallel_50_times, parallel_70_times]

# Execution times for different numbers of processes
num_processes = [0, 2, 5, 10, 20, 50, 70]

# Plotting
for i, operation in enumerate(operations):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(num_processes, [local_times[i]] + parallel_times[i], marker='o')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title(operation + ' - Comparison of Local vs. Parallel Execution')
    plt.grid(True)
    plt.xticks(np.arange(0, 71, step=5))
    plt.tight_layout()
    plt.show()

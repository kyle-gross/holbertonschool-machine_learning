#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Plot data
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.hist(student_grades, bins=bins, edgecolor='black')

# Create labels
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')

# Format
plt.xlim(0, 100)
plt.xticks(ticks=bins)
plt.ylim(0, 30)

plt.show()

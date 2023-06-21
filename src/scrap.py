import numpy as np

# Read in ../data/compulsory_buyback.txt
data = np.loadtxt("../data/voluntary_landswap.txt")

# Append annual cost with program as a column
cost_with_program = np.array([12.88, 12.15, 11.43, 10.71, 10.27, 9.98, 9.84, 9.69, 9.62, 9.55, 
                              9.48, 9.42, 9.36, 9.3, 9.26, 9.22, 9.17, 9.14, 9.12, 9.1, 9.09])
data = np.concatenate((data, cost_with_program.reshape(-1, 1)), axis=1)

# Print data
print(data)

# Save data back to same file
np.savetxt("../data/voluntary_landswap.txt", data)
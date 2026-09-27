# Maryam Shamshad 
# 23-NTU-CS-1047
# Data Visualization
# Filling Datapoint with specific Colours
import matplotlib.pyplot as plt
year = [1994, 1995, 1998, 2000]
population = [2.59, 3.69, 5.33, 6.77] 
plt.plot(year, population, 'r', label='Population 1', linewidth=5)
pop2 = [4.44, 3.22, 5.55, 6.88]
plt.plot(year, pop2, 'c', label='Population 2', linewidth=5)
plt.xlabel('Independent var')  
plt.ylabel('Dependent var')  
plt.title('Interesting Graph')  
plt.legend()  
plt.grid(True, color='k')
plt.fill_between(year, population, 0, color='green')
plt.show()
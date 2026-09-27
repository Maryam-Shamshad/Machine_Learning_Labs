# Maryam Shamshad 
# 23-NTU-CS-1047
# Data Visualization
# Legends, Titles & Tables
import matplotlib.pyplot as plt
year=[1994, 1995, 1998, 2000]
population=[2.59,3.69,5.33,6.77]
plt.plot(year,population,label='population 1') 
pop2=[4.44,3.22,5.55,6.88]
plt.plot(year,pop2, label='population 2') 
plt.xlabel('Independent var') 
plt.ylabel('Dependent var') 
plt.title('Interesting Graph') 
plt.legend()
plt.show()
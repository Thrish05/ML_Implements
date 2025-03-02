import numpy as np
import matplotlib.pyplot as plt

#using the same aspects of x and y data points, to perform linear search
x = np.random.normal(10 ,3 , 50)
noise = np.random.normal(0, 5, 50)
y = 2*x + 3 + noise


possible_m1 = np.linspace(-5, 5, 50) # i created 100 possible m1 values in the range of -5 to 5
possible_m2 = np.linspace(-5, 5, 50) # same here

'''To perform the linear search and record the best possible m1 and m2 values according to the 
    mse found, im maintaining the respective global m1, m2 and mse values'''
m1_bestSoFar = None
m2_bestSoFar = None
mse_bestSoFar = float('inf')

countOfBestSoFar = 0
count = 0

#lists to store the values of respective loss w.r.t change in m1 and m2 values (for plotting purposes)
mse_list = []

#performing the linear search over all possible m1 and m2 values
for m1 in possible_m1:
    for m2 in possible_m2:

        y_predicted = m1 * x + m2 #calculating the current set of y_predicted for each m1 and m2
        mse = np.mean((y-y_predicted)**2) #calculating the error encountered

        '''if the error obtained is in any way less than the global error found so far, the error, m1 and m2 values 
        are updated'''
        if mse < mse_bestSoFar: 
            mse_bestSoFar = mse
            m1_bestSoFar = m1
            m2_bestSoFar = m2
            countOfBestSoFar = count

        count += 1 # to store the iteration count for readability while printing
        if(count % 100 == 0):
            print(f"m1: {m1}, m2: {m2}, mse: {mse}, count: {count}")
        mse_list.append(mse_bestSoFar)

print(f"Best m1: {m1_bestSoFar}, Best m2: {m2_bestSoFar}, mse: {mse_bestSoFar}, At count: {countOfBestSoFar}")
y_best = m1_bestSoFar * x + m2_bestSoFar

plt.figure(1) # Plotting the initial data points vs obtained line of best fit
plt.scatter(x, y, color = "black", label = "Data Points")
plt.plot(x, y_best, color = "green", label = "Best Fit Line")
plt.xlabel("Data Points")
plt.ylabel("Best Fit Line")
plt.title("Initial Data points with Best Fit Line")
plt.legend()

plt.figure(2) # Plotting the best mse according to each iteration
iter_list = []

for i in range(2500):
    iter_list.append(i)

plt.plot(iter_list, mse_list)
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.title("Visualizing MSE against Iterations")
plt.show()

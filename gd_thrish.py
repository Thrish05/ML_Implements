import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(10 ,3 , 50) # initially, I created my x array as a normal distribution of 50 values with mean value to be 10 with a spread of 3
noise = np.random.normal(0, 5, 50) # generated 50 corresponding values to act as "noise" to further deviate my y values
y = 2*x + 3 + noise # created the y array by feeding MY assumed line of best fit, plugging in the x array and noise

m1_current = 0 # I want to store the current m1 and m2 values so that I can plot the obtained line of best fit after performing the GD function
m2_current = 0

# necessary lists to store the values of respective loss w.r.t change in m1 and m2 values
mse_list = []
m1_list = []
m2_list = []

iterations = 1000
def gradient_descent(x,y, m1_current, m2_current): # created the GD function utilising the parameters of inital x, y data points AND the current m1 and m2 values

    n = len(x)
    learning_rate = 0.001 #Utilised the optimal learning rate according to the iteration count of 1000 (i found it via trial and error)

    for i in range(iterations):
        y_predicted = (m1_current * x) + m2_current # calculating the predicted set of y values according to the current m1 and m2
        cost = (1/n) * (sum(y - y_predicted)) ** 2
        '''there is no need for the machine to calculate the cost function (aka loss function), it is purely for printing it out after each iteration
            to facilitate MY understanding of the way cost changes over iterations'''
        m1_d = -(2/n) * sum(x*(y - y_predicted))
        '''Im taking the partial derivative of the loss function w.r.t. m1_current
            to determine how much the loss will change with slight change in m1_current'''
        
        m2_d = -(2/n) * sum(y - y_predicted)
        '''Similarly, partial derivative of the loss function w.r.t. m2_current
            to determine how much the loss will change with slight change in m2_current'''
            
        m1_current -= (learning_rate * m1_d)
        m2_current -= (learning_rate * m2_d)
        '''updating the current m1 and m2 values by applying the learning rate and the calculated derivative'''
        if(i % 100 == 0):
            print ("m1: {}, m2: {}, cost: {}, iteration: {}".format(m1_current, m2_current, cost, i))

        mse_list.append(cost)
        m1_list.append(m1_current)
        m2_list.append(m2_current)
    
    return m1_current, m2_current # returning the final m1 and m2 values after performing the GD iterations

m1_current, m2_current = gradient_descent(x,y, m1_current, m2_current)


y_best = m1_current * x + m2_current
plt.figure(1)
plt.scatter(x, y, color = "black", label = "Data Points")
plt.plot(x, y_best, color = "green", label = "Best fit line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Initial Data Points with Best Fit Line")
plt.legend()

plt.figure(2) #Loss vs M1 graph
plt.plot(m1_list, mse_list)
plt.xlabel("M1")
plt.ylabel("MSE")
plt.title("MSE vs. m1")

plt.figure(3) #Loss vs M2 graph
plt.plot(m2_list, mse_list)
plt.xlabel("M2")
plt.ylabel("MSE")
plt.title("MSE vs. m2")

plt.figure(4)
iter_list = []

for i in range(iterations):
    iter_list.append(i)

plt.plot(iter_list, mse_list)
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.title("Visualizing MSE against Iterations")
plt.show()



'''
    Also, I had some trouble initially trying to debug why my cost went on to get higher and higher and reach inf at a certain point. I then
    compared my code with the code provided in the youtube video i referred and realised i made a mistake of coding the error as y_predicted - y
    instead of y - y_predicted in the cost function, when i was writing the code. I also learned that when we DO use the error as y_predicted - y
    we tend to find the respective derivated of the slope and intercept in a way that aims to MAXIMISE the cost function INSTEAD OF minimising it.

    In the end, I've managed to resolve this and now my GD function works as intended. The cost function is decreasing over time and the line of best fit
    is progressively improving as the iterations progress.
'''

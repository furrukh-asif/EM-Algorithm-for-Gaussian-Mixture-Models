import pandas as pd
import numpy as np 
from matplotlib import pyplot
import math
from copy import deepcopy
 

# Reading data from the csv file
data_frame = pd.read_csv('./weight-height.csv')

# Cleaning data
data_frame = data_frame.drop(columns=["Weight"])
df_m = data_frame[data_frame.Gender.eq('Male')]
df_f = data_frame[data_frame.Gender.eq('Female')]
#print(df_f.head())

# Calculating Mean and SD for individual distributions

mu_female = df_f["Height"].mean()
mu_male = df_m["Height"].mean()
print(mu_female, mu_male)
std_female = df_f["Height"].std()
std_male = df_m["Height"].std()
print(std_female, std_male)


def pdf(data, mean: float, variance: float):
  # A normal continuous random variable.
  s1 = 1/(np.sqrt(2*np.pi*variance))
  s2 = np.exp(-(np.square(data - mean)/(2*variance)))
  return s1 * s2

data = np.array(data_frame["Height"])
count, binsm, ignored = pyplot.hist(df_m["Height"], bins=40, density=True, color="navy", label="Male")
countf, binsf, ignoredf = pyplot.hist(df_f["Height"], bins=40, density=True, color = "red", label="Female")
pyplot.legend()
pyplot.savefig("Histogram")
pyplot.show()
bins = np.linspace(np.min(data),np.max(data),100)
pyplot.xlabel("$height$")
pyplot.ylabel("pdf")
pyplot.scatter(np.array(df_m["Height"]), [0.005] * len(np.array(df_m["Height"])), color='navy', marker=2, label="Male Data")
pyplot.scatter(np.array(df_f["Height"]), [0.005] * len(np.array(df_f["Height"])), color='red', marker=2, label="Female Data")
pyplot.plot(bins, pdf(bins, mu_male, std_male**2), color='navy', label="True pdf (Male)")
pyplot.plot(bins, pdf(bins, mu_female, std_female**2), color='red', label="True pdf (Female)")
pyplot.legend()
pyplot.savefig("Original pdfs")
pyplot.show()

pyplot.scatter(data, [0.005] * len(data), color='purple', marker=2, label="Train Data")
pyplot.legend()
pyplot.savefig("Data without Gender")
pyplot.show()

#Initial estimates
k = 2
#Two random values picked from the dataset as the initial estimates for the gaussian means
means_est = np.random.choice(data, k)
#Two random values picked between 0 and 1 as the initial estimates for the gaussian variances
variances_est = np.random.random_sample(size=k)
#Initial probability of a datapoint belonging to a particular Gaussian (equal)
probs_est = np.ones((k))/k

iteration_no = 0
epsilon = 0.001

print(means_est)
print(variances_est)
for j in range(100):
    prev_est = deepcopy(means_est)
    #Plotting
    bins = np.linspace(np.min(data),np.max(data),100)
    pyplot.xlabel("$x$")
    pyplot.ylabel("pdf")
    pyplot.title("Iteration {}".format(j))
    pyplot.scatter(data, [0.005] * len(data), color='purple', marker=2, label="Train Data")
    pyplot.plot(bins, pdf(bins, means_est[0], variances_est[0]), color='blue', label="Cluster 1")
    pyplot.plot(bins, pdf(bins, means_est[1], variances_est[1]), color='green', label="Cluster 2")
    pyplot.plot(bins, pdf(bins, mu_male, std_male**2), color='black', label="True pdf")
    pyplot.plot(binsf, pdf(binsf, mu_female, std_female**2), color='black')
    pyplot.legend()
    pyplot.savefig("./Iterations/Iteration_{0:02d}".format(j))
    pyplot.show()
    #Expectation Step

    #Calculating the pdf of each datapoint for each cluster using the estimated values for mean and variance
    likelihood = []
    for i in range(k):
        likelihood.append(pdf(data, means_est[i], np.sqrt(variances_est[i])))
    likelihood = np.array(likelihood)

    #Calculate the likelihood of each datapoint belonging to a cluster k (using Bayes Theorem)
    b = []
    for i in range(k):
        likelihood_of_i = likelihood[i] * probs_est[i]
        likelihood_total = np.sum([likelihood[j] * probs_est[j] for j in range(k)], axis=0)
        b.append(likelihood_of_i/likelihood_total)
  
    #Maximization Step 

    #Update Estimates
    for i in range(k):
        means_est[i] = np.sum(b[i] * data)/np.sum(b[i])
        variances_est[i] = np.sum(b[i] * np.square(data - means_est[i]))/np.sum(b[i])
        probs_est[i] = np.mean(b[i])

    #Check for termination
    check = True
    for i in range(k):
        if abs(means_est[i] - prev_est[i]) > epsilon:
            check = False
    if check:
        break



print(means_est)
print(variances_est)




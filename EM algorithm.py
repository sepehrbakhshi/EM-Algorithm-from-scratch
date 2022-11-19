# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:05:42 2019

@author: Sepehr
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import collections
#the em algorithm
class EM_Algorithm(object):
    #initilize variables
    def __init__(self, X, k,samples,covariance_method):

        self.first_training_set = X
        self.number_of_clusters = k
        self.covariance_method = covariance_method
        self.mu = np.zeros(shape=(self.number_of_clusters,2))
        #initilize mu
        j=0
        for i in range (self.number_of_clusters):
            tmp= self.first_training_set[samples[j]:samples[j+1]]
            self.mu[i:] = [sum(x) for x in zip(*tmp)]
            self.mu[i:] = self.mu[i:] / (samples[j+1] - samples[j])
            j=j+1
         #initialize covariance
        self.cov = []

        j=0
        for i in range (self.number_of_clusters):
            tmp = (1/samples[j+1] - samples[j])*np.dot((self.first_training_set[samples[j]:samples[j+1]] - self.mu[i:i+1]).T, self.first_training_set[samples[j]:samples[j+1]] - self.mu[i:i+1])
            self.cov.append(tmp)
            j=j+1
        self.cov =  np.array(self.cov)

        #initialize alpha or prior
        self.prior = []
        for i in range (self.number_of_clusters):
            self.prior.append(1/4)
        self.threshold = 0.0001
        self.prev_log_likelihood = 0
        self.new_log_likelihood = 1
        self.num_of_iteration = 0
    #find the pdf
    def pdf(self,cov1,mu1,data):

        multiplication = (((data-mu1))*np.dot(np.linalg.inv(cov1),(data-mu1).T).T)

        multiplication_addup = [sum(i) for i in multiplication]
        multiplication_addup = np.array(multiplication_addup)

        #print(multiplication_addup.shape)
        pdf = []
        pdf = (1/((2* np.pi)*(np.linalg.det(cov1)**0.5)))*np.exp((-0.5)*multiplication_addup)
        pdf =  np.array(pdf)

        return pdf
    #calculate log likelihood
    def log_likelihood(self):
        new_log_likelihood = 0
        temp_likelihood = 0
        tmp = 0
        for i in range (500):
            new_set = []
            temp_likelihood = 0
            copy = 0
            while(copy < self.number_of_clusters):
                new_set.append(self.first_training_set[i,:])
                copy = copy +1
            new_set =  np.array(new_set)

            for j in range (self.number_of_clusters):
               tmp = self.prior[j] * self.pdf(self.cov[j,:],self.mu[j,:],new_set)
               temp_likelihood = temp_likelihood + tmp[0]

            new_log_likelihood = new_log_likelihood + np.log(temp_likelihood)
        return  new_log_likelihood
    #train our data
    def train(self):
        counter = 0
        while(self.new_log_likelihood - self.prev_log_likelihood > self.threshold):
            rc = []

            for i in range (self.number_of_clusters) :
                tmp = self.prior[i] * self.pdf(self.cov[i],self.mu[i],self.first_training_set)
                rc.append(tmp)
            rc =  np.array(rc , float)
            rc = rc.T

            sum_pf_pdfs = [sum(i) for i in rc]
            sum_pf_pdfs =  np.array(sum_pf_pdfs , float)

            for i in range (500):
                rc[i, :] /= sum_pf_pdfs[i]

            total_number =  [sum(i) for i in zip(*rc)]
            self.prior = [number / 500 for number in total_number]
            #update mu
            tmp_mu = np.zeros(shape=(1,2))
            for j in range (self.number_of_clusters):
                tmp_mu = np.zeros(shape=(1,2))
                for i in range (500):
                    tmp = self.first_training_set[i,:]*rc[i,j]
                    tmp_mu =tmp_mu + tmp
                self.mu[j] = tmp_mu / total_number[j]
            #using arbitrary covariance matrix
            if(self.covariance_method == "arbitrary"):
                self.cov = []
                for j in range (self.number_of_clusters):
                   tmp_cov = np.zeros(shape=(2,2))

                   for i in range (500):
                        tmp = self.first_training_set[i,:]-self.mu[j,:]
                        tmp = np.array(tmp)
                        tmp = tmp.reshape(tmp.shape[0],1)
                        tmp = np.dot(tmp,tmp.T)
                        tmp_cov = tmp_cov + (rc[i,j]*tmp)
                   tmp_cov = tmp_cov / total_number[j]

                self.cov =  np.array(self.cov)
            #spherical covariance
            if(self.covariance_method == "spherical"):
                self.cov = []
                for j in range (self.number_of_clusters):
                   tmp_cov = np.zeros(shape=(2,2))
                   for i in range (500):
                        tmp = self.first_training_set[i,:]-self.mu[j,:]
                        tmp = np.sum(tmp**2)

                        tmp_cov = tmp_cov + (rc[i,j]*tmp)

                   tmp_cov = tmp_cov / (total_number[j]*2)
                   self.cov.append(tmp_cov)
                self.cov =  np.array(self.cov)
                for matrix in self.cov:
                    matrix[0,1] = 0
                    matrix[1,0] = 0
            #diagonal covariance matrix
            if(self.covariance_method == "diagonal"):
                self.cov = []
                for j in range (self.number_of_clusters):
                   tmp_cov = np.zeros(shape=(2,2))
                   for i in range (500):
                        tmp = self.first_training_set[i,:]-self.mu[j,:]
                        tmp = tmp**2
                        tmp_cov = tmp_cov + (rc[i,j]*tmp)
                   
                   tmp_cov = tmp_cov / (total_number[j])
                   self.cov.append(tmp_cov)
                self.cov =  np.array(self.cov)

                for matrix in self.cov:
                    matrix[1,0] = 0
                    matrix[0,1] = 0
            self.prev_log_likelihood = self.new_log_likelihood
            self.new_log_likelihood = self.log_likelihood()
            if(counter%10 ==0):
                print("Log Likelihood: ")
                print(self.new_log_likelihood)
                print("\n")
            counter = counter+1
        self.num_of_iteration = counter-1

#######################################
#load data
loadedData = np.loadtxt("dataset.txt")

first_data_set= loadedData[:1000]
second_data_set = loadedData[1000:2000]
third_data_set = loadedData[2000:]

first_training_set = first_data_set[:500,0:2]
second_training_set = second_data_set[:500,0:2]
third_training_set = third_data_set[:500,0:2]

first_test_set = first_data_set[500:1000,0:2]
second_test_set = second_data_set[500:1000,0:2]
third_test_set = third_data_set[500: 1000,0:2]

test_set_starting_point = 500
#calculate log likelihood
samples = [0,125,250,375,500]
em1 = EM_Algorithm(first_training_set , 4 ,samples ,"diagonal")
em1.train()
print("Last Log Likelihhood for the first training set: ")
print(em1.new_log_likelihood)
print("number of iteration for the first training set: ")
print(em1.num_of_iteration)
print("\n")
print("----------------------------------")
print("\n")
samples =[0,250,500]
em2 = EM_Algorithm(second_training_set ,2,samples ,"diagonal")
em2.train()
print("Last Log Likelihhood for the second training set: ")
print(em2.new_log_likelihood)
print("number of iteration for the second training set: ")
print(em2.num_of_iteration)
print("\n")
print("----------------------------------")
print("\n")
samples =[0,250,500]
em3 = EM_Algorithm(third_training_set , 2,samples ,"diagonal")
em3.train()
print("Last Log Likelihhood for the third training set: ")
print(em3.new_log_likelihood)
print("number of iteration for the third training set: ")
print(em3.num_of_iteration)
print("\n")
print("----------------------------------")
print("\n")
#plot function
def myPlot():
    x0 = [row[0] for row in loadedData]
    y0 = [row[1] for row in loadedData]
    x0 = np.array(x0)
    y0 = np.array(y0)
    x0 = x0.reshape(x0.shape[0],1)
    y0 = y0.reshape(y0.shape[0],1)

    first_class = loadedData[loadedData[:, 2] == 1.0, :]
    second_class = loadedData[loadedData[:, 2] == 2.0, :]
    third_class = loadedData[loadedData[:, 2] == 3.0, :]

    x_first_class = [row[0] for row in first_class]
    y_first_class = [row[1] for row in first_class]

    x_second_class = [row[0] for row in second_class]
    y_second_class = [row[1] for row in second_class]

    x_third_class = [row[0] for row in third_class]
    y_third_class = [row[1] for row in third_class]
    plt.figure(figsize=(9,9))
    plt.scatter(x_first_class,y_first_class,marker = '+' ,c='b')
    plt.scatter(x_second_class,y_second_class,marker = 'o' ,c='r' ,facecolors='none')
    plt.scatter(x_third_class,y_third_class, marker = '*',c='g')
    plt.xticks(np.arange(0, 1+0.1, 0.1))
    plt.yticks(np.arange(0, 1+0.1, 0.1))
    plt.gca().set_aspect('equal', adjustable='box')
#predict new result for test set
def predict(test_set,class_label):
    first_estimated = []
    np.array(first_estimated)
    first_estimated = np.zeros(test_set_starting_point)
    second_estimated = []
    np.array(second_estimated)
    second_estimated = np.zeros(test_set_starting_point)
    third_estimated = []
    np.array(third_estimated)
    third_estimated =  np.zeros(test_set_starting_point)
    for i in range(em1.number_of_clusters):
        first_estimated = first_estimated + em1.prior[i] * em1.pdf(em1.cov[i],em1.mu[i], test_set )
    for i in range(em2.number_of_clusters):
        second_estimated = second_estimated + em2.prior[i] * em2.pdf( em2.cov[i], em2.mu[i], test_set)
    for i in range(em3.number_of_clusters):
        third_estimated  = third_estimated  + em3.prior[i] * em3.pdf(em3.cov[i], em3.mu[i],test_set  )
    class_labels = []
    i = 0
    while (i < test_set_starting_point):
        if(first_estimated[i] > second_estimated[i] and first_estimated[i] > third_estimated[i]):
            class_labels.append(1)
        elif(second_estimated[i] > first_estimated[i] and second_estimated[i] > third_estimated[i]):
            class_labels.append(2)
        else :
            class_labels.append(3)
        i = i + 1
    np.array(class_labels)
    counter=collections.Counter(class_labels)
    accuracy = 0
    if(class_label == 1):
        accuracy = counter[1]/(counter[1]+counter[2]+ counter[3])
    elif(class_label == 2):
        accuracy = counter[2]/(counter[1]+counter[2]+ counter[3])
    else:
        accuracy = counter[3]/(counter[1]+counter[2]+ counter[3])
    return counter,accuracy
#plot each guassian distribution
myPlot()
x = np.linspace(0, 1 , 100)
y = np.linspace(0, 1 , 100)
x0, y0 = np.meshgrid(x, y)
whole_data = np.dstack((x0, y0))
my_GM = []
for i in range (em1.number_of_clusters):
    my_GM.append(multivariate_normal(em1.mu[i], em1.cov[i]))
    my_GM[i] =  my_GM[i].pdf(whole_data)
    plt.contour(x0, y0,my_GM[i])
txt = "First class with Arbitrary covariance matrix"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('C:/Users/Sepehr/Documents/1stArbitrary.png')
plt.show()

myPlot()
x = np.linspace(0, 1 , 80)
y = np.linspace(0, 1 , 80)
x0, y0 = np.meshgrid(x, y)
whole_data = np.dstack((x0, y0))
my_GM = []
for i in range (em2.number_of_clusters):

    my_GM.append(multivariate_normal(em2.mu[i], em2.cov[i]))
    my_GM[i] =  my_GM[i].pdf(whole_data)
    plt.contour(x0, y0,my_GM[i])
txt = "Second class with Arbitrary covariance matrix"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('C:/Users/Sepehr/Documents/2ndArbitrary.png')

plt.show()

myPlot()
x = np.linspace(0, 1 , 100)
y = np.linspace(0, 1 , 100)
x0, y0 = np.meshgrid(x, y)
whole_data = np.dstack((x0, y0))
my_GM = []

for i in range (em3.number_of_clusters):
    my_GM.append(multivariate_normal(em3.mu[i], em3.cov[i]))
    my_GM[i] =  my_GM[i].pdf(whole_data)
    plt.contour(x0, y0,my_GM[i])
txt = "Third class with Arbitrary covariance matrix"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('C:/Users/Sepehr/Documents/3rdArbitrary.png')

plt.show()
#calculation accuracy and also data for confusion matrix
counter1 , accuracy1 = predict(first_test_set,1)
counter2 , accuracy2 = predict(second_test_set,2)
counter3 , accuracy3 = predict(third_test_set,3)

counter_train1 , accuracy_train1 = predict(first_training_set,1)
counter_train2 , accuracy_train2 = predict(second_training_set,2)
counter_train3 , accuracy_train3 = predict(third_training_set,3)
print("TEST SET :")
print("--------------------------")
print(counter1)
print("\n")
print("First Gaussian Mixture accuracy: " + str(accuracy1))
print("--------------------------")
print("--------------------------")
print(counter2)
print("\n")
print("Second Gaussian Mixture accuracy: " + str(accuracy2))
print("--------------------------")
print("--------------------------")
print(counter3)
print("\n")
print("Third Gaussian Mixture accuracy: " + str(accuracy3))
print("--------------------------")

overall_accuracy = (counter1[1] + counter2[2] + counter3[3])/(1500)
print("--------------------------")
print("Overall accuracy : " + str(overall_accuracy))
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("\n")

print("Training SET :")
print("--------------------------")
print(counter_train1)
print("\n")
print("First Gaussian Mixture accuracy: " + str(accuracy_train1))
print("--------------------------")
print("--------------------------")
print(counter_train2)
print("\n")
print("Second Gaussian Mixture accuracy: " + str(accuracy_train2))
print("--------------------------")
print("--------------------------")
print(counter_train3)
print("\n")
print("Third Gaussian Mixture accuracy: " + str(accuracy_train3))
print("--------------------------")

overall_accuracy = (counter_train1[1] + counter_train2[2] + counter_train3[3])/(1500)
print("--------------------------")
print("Overall accuracy : " + str(overall_accuracy))

#print the final mu covariance and alpha
print(em1.cov)
print(em2.cov)
print(em3.cov)


print(em1.mu)
print(em2.mu)
print(em3.mu)

print(em1.prior)
print(em2.prior)
print(em3.prior)

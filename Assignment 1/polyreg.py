import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


#Load the data into the program
def load_data():
    data = np.load('./datapoints.npy')
    train_data = data[:,0]
    train_target = data[:,1]
    return train_data, train_target


#Polynomial Regression training with analytical approach
def poly_train(order_number, train_data, train_target):
    N=order_number+1
    size_of_training = np.prod(train_data.shape)
    phi = np.empty([N,size_of_training])
    for x in range(0, size_of_training):
        a=[1]
        current_num = train_data[x]
        for y in range(0, order_number):
            a.append(np.power(current_num, y+1))
        phi[:, x]=a
    a1 = np.dot(phi, phi.transpose())
    a2 = inv(a1)
    a3 = np.dot(a2, phi)
    a4 = np.dot(a3, train_target)
    coeff = a4[::-1]
    return coeff


def poly_train_with_reg(order_number, train_data, train_target):
    N = order_number + 1
    size_of_training = np.prod(train_data.shape)
    phi = np.empty([N, size_of_training])
    for x in range(0, size_of_training):
        a = [1]
        current_num = train_data[x]
        for y in range(0, order_number):
            a.append(np.power(current_num, y + 1))
        phi[:, x] = a
    I = np.identity(N)
    I[0,0]=0
    a1 = np.dot(phi, phi.transpose())
    a2 = a1 + np.multiply(0.1, I)
    a3 = inv(a2)
    a4 = np.dot(a3, phi)
    a5 = np.dot(a4, train_target)
    coeff = a5[::-1]
    return coeff


def poly_train_fit(order_number, train_data, train_target):
    coeff = np.polyfit(train_data, train_target, order_number)
    return coeff


if __name__ == '__main__':
    data, target = load_data()
    coeff = poly_train_with_reg(9,data,target)
    # We can also use polyfit to get the same solution: coeff2 = poly_train_fit(2, data, target)
    print(coeff)
    p = np.poly1d(coeff)
    xp = np.linspace(-2, 2, 100)
    _ = plt.plot(data, target, '.', xp, p(xp), '-')
    plt.show()

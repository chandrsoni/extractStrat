
import pandas as ps
import numpy as np
import linear
import bernauli

# neuronCount = [31, 23, 10, 8, 5, 3,1]

# Wtest = np.random.randn(5,4)
# A0 = np.random.randn(4,1)
# B = np.random.randn(5,1)
# print(Wtest)
# print(A0)
# print(B)
# print(combination(Wtest, A0, B))


# ------------------------------------------------------

# let input be x of 150 values;
# x[:, 0] = constant
# x[:, 1] = day of quarter
# x[:, 2] = time of quarter
# creating sample input for pattern recog
size = 8770
lowvalue = 210
highvalue = 220
nparam = 32
x = np.random.uniform(lowvalue, highvalue, (size, nparam)) 
x[:, 0] = np.ones(size)
x[:, 1] = np.ones(size) # leave for day of quarter
x[:, 2] = np.arange(size) # leave for time of the quarter
x = x.T                                                         # x = 31 X 1500
inputcsv = ps.read_csv('MSFT.csv')
a = np.zeros((4, size))
a[0] = inputcsv['High']
a[1] = inputcsv['Low']
a[2] = inputcsv['Close']
a[3] = inputcsv['Open']
y = np.random.uniform(lowvalue, highvalue, (size))           # y = 1 x 1500

n2Count = 250
w3 = np.random.ranfdn(n2Count, nparam)                           # w3 = 250 X 31
b3 = np.random.uniform(-1, 1 ,(n2Count, 1))                     # b3 = 250 X 1
n1Count = 64
w1 = np.random.randn(n1Count, nparam) * 0.01                    # w1 = 64 X 31
b1 = np.random.uniform(-1, 1, (n1Count, 1))                     # b1 = 61 X 1

w2 = np.random.randn(n2Count, n1Count * 2 + 1) * 0.01           # w2 = 250 X 129
b2 = np.random.uniform(-1, 1 ,(n2Count, 1))                     # b2 = 250 X 1


def forward(x, y, w1, b1, w2, b2, w3, b3, size, alpha):
    # forward function
    a1 = linear.combination(w1, x, b1)                          # a1 = 64 X 1500
    # ready for layer 2
    a1sq = np.square(a1)
    sqinput = np.concatenate((a1, a1sq, np.ones((1, size))), 0) # sqinput = 129 X 1500
    lineara1 = linear.combination(w2, sqinput, b2)              # a2 = 250 X 1500
    a2 = bernauli.sigmoid(lineara1)
    # a2 is now activators for patterns
    patterns = linear.combination(w3, x, b3)                    # patterns = 250 X 1500
    print(patterns)
    print(patterns.shape)
    print(y)
    print(y.shape)
    print(patterns - y)
    error = patterns - y
    errorabs = np.abs(patterns - y)
    classifier = errorabs == np.min(errorabs, 0)
    patternout = classifier * y
    clearout = classifier * error
    dw3 = (1./size)*((error) * (x.T))
    db3 = (1./size)*(error)
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3
    dw2 = (1./size)*((classifier - a2) * (sqinput.T))
    db2 = (1./size)*((classifier - a2))
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    da2 = (1./size)*(w2.T * (classifier - a2))
    dw1 = (1./size)*(da2 * (x.T))
    db1 = (1./size)*(da2)
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    return dw3, db3, dw2, db2, dw1, db1

def backward(x, o, y, w):
    return dw, db, da


# let's make w1 and b1 now to get to second layer
# going with 64 neurons in layer 2

print(forward(x, y, w1, b1, w2, b2, w3, b3, size, 0.1))
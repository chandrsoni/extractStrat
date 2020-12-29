
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

w1read = ps.read_csv('w1.csv').to_numpy()
w2read = ps.read_csv('w2.csv').to_numpy()
w3read = ps.read_csv('w3.csv').to_numpy()
b1read = ps.read_csv('b1.csv').to_numpy()
b2read = ps.read_csv('b2.csv').to_numpy()
b3read = ps.read_csv('b3.csv').to_numpy()

size = 8770
lowvalue = 210
highvalue = 220
nparam = 32
n2Count = 250
w3 = w3read[:,1:]                           # w3 = 250 X 31
b3 = b3read[:,1:]                     # b3 = 250 X 1
n1Count = 64
w1 = w1read[:,1:]                    # w1 = 64 X 31
b1 = b1read[:,1:]                     # b1 = 61 X 1

w2 = w2read[:,1:]           # w2 = 250 X 129
b2 = b2read[:,1:]                     # b2 = 250 X 1


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
    return a2, patterns

print(forward(x, y, w1, b1, w2, b2, w3, b3, size, 1))

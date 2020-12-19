
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
size = 1500
lowvalue = 210
highvalue = 220
nparam = 31
x = np.random.uniform(lowvalue, highvalue, (size, nparam))
x[:, 0] = np.ones(size)
x[:, 1] = np.ones(size)
x[:, 2] = np.arange(size)
x = x.T
output = np.random.uniform(lowvalue, highvalue, (size, 1))

# let's make w1 and b1 now to get to second layer
# going with 64 neurons in layer 2
n1Count = 64
w1 = np.random.randn(n1Count, nparam) * 0.01
b1 = np.random.uniform(-1, 1, (n1Count, 1))

# forward function

a1 = linear.combination(w1, x, b1)

# ready for layer 2
a1sq = np.square(a1)
sqinput = np.concatenate((a1, a1sq, np.ones((1, size))), 0)

n2Count = 250
w2 = np.random.randn(n2Count, n1Count * 2 + 1) * 0.01
b2 = np.random.uniform(-1, 1 ,(n2Count, 1))

lineara1 = linear.combination(w2, sqinput, b2)

a2 = bernauli.sigmoid(lineara1)


# a2 is now activators for patterns

w3 = np.random.randn(n2Count, nparam) * 0.01
b3 = np.random.uniform(-1, 1 ,(n2Count, 1))
patterns = linear.combination(w3, x, b3)

patternImpact = patterns * a2
output = np.sum(patternImpact, 0)

print(output)
# algorithm
# 1. bring last n time series to generate patterns
# 2. fron the patterns, activate non-linear predictors based on the time serieses
# 3. merge the activated predictors to create output predictions


import pandas as ps

train = ps.read_csv('data/newmsft.csv')

train.head()

neuronCount = [31, 23, 10, 8, 5, 3,1]

print(train)

# parameters
# date of quarter
# time
# previous tick - lowlow - lowhigh - highlow - highhigh
# tick before that - lowlow - lowhigh - highlow - highhigh
# tick before that - lowlow - lowhigh - highlow - highhigh
# tick before that - lowlow - lowhigh - highlow - highhigh
# tick before that - lowlow - lowhigh - highlow - highhigh
# tick before that - lowlow - lowhigh - highlow - highhigh
# tick before that - lowlow - lowhigh - highlow - highhigh
# constant

# w1 = 23 x 31 b1 = 23 x 1
# w2 = 10 x 23 b2 = 10 x 1
# w3 = 8 x 10 b3 = 8 x 1
# w4 = 5 x 8 b4 = 5 x 1
# w5 = 3 x 5 b5 = 3 x 1

# 1 - first layers to have a weight and constant input
# a1 = E((W* a0) + b)
# 2nd layer to extracting patterns out of the input prices - bernauli of 3rd orddr functions on a1 to generate a2 - neuron count = 100
# 3rd layer activation out of patterns 
# 4th layer combination of pattern inputs (for culmination of parameters) and bring back input values as well to predict values outs
# 5th layer single neuron to output the prediction
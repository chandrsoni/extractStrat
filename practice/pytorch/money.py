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
# algorithm
1. bring last n time series to generate patterns
2. fron the patterns, activate non-linear predictors based on the time serieses
3. merge the activated predictors to create output predictions

# parameters
date of quarter

time

previous tick - lowlow - lowhigh - highlow - highhigh

tick before that - lowlow - lowhigh - highlow - highhigh

tick before that - lowlow - lowhigh - highlow - highhigh

tick before that - lowlow - lowhigh - highlow - highhigh

tick before that - lowlow - lowhigh - highlow - highhigh

tick before that - lowlow - lowhigh - highlow - highhigh

tick before that - lowlow - lowhigh - highlow - highhigh

constant

# constants
w1 = 23 x 31 b1 = 23 x 1

w2 = 10 x 23 b2 = 10 x 1

w3 = 8 x 10 b3 = 8 x 1

w4 = 5 x 8 b4 = 5 x 1

w5 = 3 x 5 b5 = 3 x 1

# layers

1 - first layers to have a weight and constant input

a1 = E((W* a0) + b)

2nd layer to extracting patterns out of the input prices - bernauli of 3rd orddr 
functions on a1 to generate a2 - neuron count = 100  considering 2** 6, might need to consider 32 and 64 as well
- initialize parameter to be random generation on increased modular/ geeration cap

3rd layer activation out of patterns 

4th layer combination of pattern inputs (for culmination of parameters) and bring back input values as well to predict values outs

5th layer single neuron to output the prediction

# backtracking
y = output of max value on next minute

- find the closest prediction from patterns
- this should generate index matriz of size patterns X inputsize for prediction
- this should also generate the error matrix which is the scalar multiplication of the above matrix with the corresponding pattern
- ise error from 2 to backtrack on the identifier
- use error from 3 to backtrack and train on the predictions

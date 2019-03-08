import numpy as np

#convolution output shape
def c(input_height, input_width, filter_height, filter_width, P, S, K):
    new_height = (input_height - filter_height + 2*P)/S + 1
    new_width = (input_width - filter_width + 2*P)/S + 1
    new_depth = K
    return new_height, new_width, new_depth

print(c(32, 32, 8, 8, 1, 2, 20))
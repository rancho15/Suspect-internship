import numpy as np

# INPUT
x = np.random.randn(32,32)
print "INPUT", x.shape

# CONVOLUTION
s = 2 # stride
p = 1 # padding
f = 3 # filter size
nf = 4 # number of filter

# create conv weight matrices
w = []
for i in range(nf):
    w.append(np.random.randn(f, f))

# add padding
c = x.shape[1]
x = np.vstack([np.zeros((p, c)), x]) # top
x = np.vstack([x, np.zeros((p, c))]) # bottom
r = x.shape[0]
x = np.hstack([np.zeros((r, p)), x]) # left
x = np.hstack([x, np.zeros((r, p))]) # right

o1 = (x.shape[0] - 3)/s + 1 # (I-F)/S +1
o2 = (x.shape[1] - 3)/s + 1

# conv output (placeholder)
y = np.zeros((o1, o2, nf))

# apply conv
for k in range(nf):
    for i in range(o1):
        for j in range(o2):
            t = x[i*s:i*s+f, j*s:j*s+f]
            z = np.sum(np.multiply(t, w[k]))
            y[i, j, k] = z
print "CONVOLUTION", y.shape

# RELU
y[y<0] = 0
print "RELU", y.shape

# FULLY CONNECTED LAYER
y = y.reshape((-1, 1))
w1 = np.random.randn(1024,10)
y = np.matmul(w1.T, y)
print "FULLY CONNECTED LAYER", y.shape

# OUTPUT
print "OUTPUT", y.shape

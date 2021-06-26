import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)  # an affine operation: y=mx+b 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))  # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # if the size is a square, you can specify with a single number
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's. weight

input = torch.randn(1,1,32,32)  # nSample x nChannels x Height x width
out = net(input)
print(out)

net.zero_grad()  # zero the gradient buffers of all parameters
out.backward(torch.rand(1,10))  # backprops with random gradeints

""""
Loss Function
A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target.
"""

target = torch.randn(10) # a dummy target
print(target)

print(out.shape)
print(target.shape)

target = target.view(1,-1) # make the same shape as output
print(target.shape)

criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

"""
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> flatten -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
"""

print(loss.grad_fn) # MSE Loss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLu

""""
Backprop
To backpropagate the error all we have to do is to loss.backward(). You need to clear the existing gradients though, 
else gradients will be accumulated to existing gradients.

"""
net.zero_grad() # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Update the weights
# weight = weight - learning_rate * gradient
learning_rate = 1e-3
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    
# optimizer
import torch.optim as optim

# create optimizer
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# in the training loop
optimizer.zero_grad()  #zero the gradient buffers
out = net(input)
loss = criterion(out, target)
print(loss)
loss.backward()
print(loss)
optimizer.step()
print(loss)


import torch
import numpy as np

torch.cuda.is_available()
some_data = [[1,2], [3,4]]
print(some_data)
type(some_data)
np.array(some_data)
np.asarray(some_data)
some_data = torch.tensor(some_data)  # convert data into tensors
print(some_data)
print(some_data.dtype)

numpy_array = np.random.rand(3,4)
print(numpy_array)

torch.from_numpy(numpy_array) # convert numpy array into tensor
torch.tensor(numpy_array)   # convert numpy array into tensor

torch.ones(3,4)
torch.zeros(3,4)

my_tensor = torch.rand(3,4)
print(my_tensor)

my_tensor.dtype
my_tensor.device
# my_tensor.to('cuda')

# Basic operations
my_tensor[:, 1:3]
my_tensor.mul(my_tensor)
my_tensor * my_tensor
my_tensor.matmul(my_tensor.T)
torch.matmul(my_tensor, my_tensor.T)
my_tensor @ my_tensor.T
my_tensor.sum(axis=0)
torch.cat([my_tensor, my_tensor], axis=1)
torch.nn.functional.softmax(my_tensor)
my_tensor.shape
my_tensor.size()
torch.rand(5, 3, 128, 128)  # batch size, number of channels, height , width
my_tensor.clip(0.2, 0.8) # everything below 0.2 converted to 0.2 and above 0.8 are converted to 0.8


import torch
import torch.onnx as onnx
import torchvision.models as models

# Saving and loading model weights
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_vgg16_pretrained.pth')

"""
To load model weights, you need to create an instance of the same model first, and then load 
the parameters using load_state_dict() method.
"""

model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_vgg16_pretrained.pth'))
model.eval()

# Exporting model to ONNX
input_image = torch.zeros(1,3,224,224)
onnx.export(model, input_image, 'model.onnx')


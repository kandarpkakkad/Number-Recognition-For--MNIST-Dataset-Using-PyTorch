import numpy as np
import torch
import cv2
import torchvision
from torchvision.datasets import MNIST

dataset = MNIST(root='data/', download=True)

#len(dataset)

test_dataset = MNIST(root='data/', train=False)
print(len(test_dataset))

import matplotlib.pyplot as plt
#%matplotlib inline

# image, label = dataset[0]
# plt.imshow(image)
# print('Label: ', label)
#
# image, label = dataset[10]
# plt.imshow(image)
# print('Label: ', label)

import torchvision.transforms as transforms

dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape)
print(label)

print(img_tensor[:, 10:15, 10:15])
print(torch.max(img_tensor))
print(torch.min(img_tensor))

# plt.imshow(img_tensor[0], cmap='gray')

def split_indices(n, val_pct):
  n_val = int(n * val_pct)
  idxs = np.random.permutation(n)
  return idxs[n_val:], idxs[:n_val]

train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)

print(len(train_indices), len(val_indices))
print('sample val indices: ', val_indices[:20])

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

batch_size=100

train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)

val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size, sampler=val_sampler)

import torch.nn as nn

input_size = 28*28
num_classes = 10

model = nn.Linear(input_size, num_classes)

print(model.weight.shape)
print(model.weight)

print(model.bias.shape)
print(model.bias)

#for images, labels in train_loader:
#  print(images.shape)
#  outputs = model(images)
#  break

class MnistModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(input_size, num_classes)
    
  def forward(self, xb):
    xb = xb.reshape(-1, 28*28)
    out = self.linear(xb)
    return out
  
model = MnistModel()

print(model.linear.weight.shape, model.linear.bias.shape)
list(model.parameters())

for images, labels in train_loader:
  outputs = model(images)
  break
  
print('outputs.shape: ', outputs.shape)
print('Sample outputs: \n', outputs[:2].data)

import torch.nn.functional as F

probs = F.softmax(outputs, dim=1)

print("Sample probabilities: \n", probs[:2].data)
print("Sum: ", torch.sum(probs[0]).item())

max_probs, preds = torch.max(probs, dim=1)
print(preds)

print(labels)

def accuracy(l1, l2):
  return torch.sum(l1 == l2).item()/len(l1)

accuracy(preds, labels)

loss_fn = F.cross_entropy

loss = loss_fn(outputs, labels)
print(loss)

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

def loss_batch(model, loss_func, xb, yb, opt = None, metric = None):
  preds = model(xb)
  loss = loss_func(preds, yb)
  
  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()
    
  metric_result = None
  if metric is not None:
    metric_result = metric(preds, yb)
    
  return loss.item(), len(xb), metric_result

def evaluate(model, loss_fn, valid_dl, metric=None):
  with torch.no_grad():
    results = [loss_batch(model, loss_fn, xb, yb, metric = metric) for xb,yb in valid_dl]
    
  losses, nums, metrics = zip(*results)
  total = np.sum(nums)
  total_loss = np.sum(np.multiply(losses, nums))
  avg_loss = total_loss / total
  avg_metric = None
  if metric is not None:
    tot_metric = np.sum(np.multiply(metrics, nums))
    avg_metric = tot_metric / total
  return avg_loss, total, avg_metric

def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1)
  return torch.sum(preds == labels).item() / len(preds)

val_loss, total, val_acc = evaluate(model, loss_fn, val_loader, metric = accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))

accuracies = []
def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
  for epoch in range(epochs):
    for xb, yb in train_dl:
      loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)
      
    result = evaluate(model, loss_fn, valid_dl, metric)
    val_loss, total, val_metric = result
    
    if metric is None:
      print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, val_loss, metric.__name__, val_metric))
    else:
      print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch+1, epochs, val_loss, metric.__name__, val_metric))

    accuracies.append(val_metric)

model = MnistModel()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

fit(30, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)

plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of eo=pochs')
plt.show()

test_dataset = MNIST(root = 'data/', train = False, transform = transforms.ToTensor())

img, label = test_dataset[0]
# plt.imshow(img[0], cmap='gray')
print('Shape: ', img.shape)
print('Label: ', label)

def predict_image(img, model):
  xb = img.unsqueeze(0)
  yb = model(xb)
  _, preds = torch.max(yb, dim=1)
  return preds[0].item()

img, label = test_dataset[0]
# plt.imshow(img[0], cmap = 'gray')
print('Label: ', label, ', Predicted: ', predict_image(img, model))

img, label = test_dataset[10]
# plt.imshow(img[0], cmap = 'gray')
print('Label: ', label, ', Predicted: ', predict_image(img, model))

img, label = test_dataset[20]
# plt.imshow(img[0], cmap = 'gray')
print('Label: ', label, ', Predicted: ', predict_image(img, model))

img, label = test_dataset[1889]
# plt.imshow(img[0], cmap = 'gray')
print('Label: ', label, ', Predicted: ', predict_image(img, model))

test_loader = DataLoader(test_dataset, batch_size = 200)

test_loss, totla, test_acc = evaluate(model, loss_fn, test_loader, metric=accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))

torch.save(model.state_dict(), 'mnist-logistic.pth')

model.state_dict()

model2 = MnistModel()
model2.load_state_dict(torch.load('mnist-logistic.pth'))
model2.state_dict()

test_loss, total, test_acc = evaluate(model2, loss_fn, test_loader, metric = accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import numpy as np

def data_loader(x, t, batch_size, shuffle=False, gpu=False, pin_memory=True):
  x = torch.from_numpy(x).float()
  t = torch.from_numpy(t).long()

  if gpu:
    x = x.cuda()
    t = t.cuda()

  set = TensorDataset(x, t)
  loader = DataLoader(set, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory)
  return loader

def evaluator(model, criterion, data_loader, gpu=True):
  '''
  There is a bug. Shape of data in neural net is not expected shape.
  you can find the bug in code of 'outputs = model(batch_x)'.
  '''
  test_loss = 0
  test_correct = 0
  num_data = 0
  model.eval()
  predict = []
  for x, y in data_loader:
    if gpu:
      batch_x = Variable(x.cuda())
      batch_y = Variable(y.cuda())
    else:
      batch_x = Variable(x)
      batch_y = Variable(y)

    outputs = model(batch_x)
    num_data += outputs.data.shape[0]
    loss = criterion(outputs, batch_y)
    corrects = num_of_correct(outputs, batch_y)

    test_loss += loss.data[0]
    test_correct += corrects
    # outputs = torch.nn.functional.softmax(outputs)
    predict.append(outputs.data.cpu().numpy())
  model.train()
  accuracy = test_correct/num_data

  pre_array = predict[0]
  for i, array in enumerate(predict):
    if i > 0:
      pre_array = np.r_[pre_array, array]
  return test_loss, accuracy, pre_array

def num_of_correct(outputs, t):
    _, predicted = torch.max(outputs, 1)
    corrects = torch.sum(predicted.data == t.data)
    return corrects

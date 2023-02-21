# note to reader: the model and training/testing logic and optimization/regularization is by Jan Rundt, other code by Helsinki University staff

import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#--- notes on results by Jan Rundt:
# accuracy with LR 0.03: 87.649%

#--- regularization:
# accuracy with LR 0.03 and early stopping: 89.378%
# acc with LR 0.03 and early stopping, and dropout: 87.973% (one dropout layer, rate 0.1)
# accuracy with LR 0.03 and early stopping, dropout and batch normalization after convolutions: 89.700%
# accuracy with LR 0.03 and early stopping, no dropout and batch normalization after convolutions: 89.838%

#--- optimization
# benchmark from above with SGD: 89.838%
# with Adam: 89.649%
# with Adagrad - very varying results, some runs with <88% accuracy, and some runs with slightly above 90% accuracy
# I also tried with different momentums for SGD, but the result varied around the same as above (around ~89-90%)

#--- one more tweek to the architecture
# I tried adding one more fully connected layer and this enhanced the accuracy somewhat, stopping at around 90-91% on different runs
# Adding weight decay regularization pushes the results up a notch, mostly varying around ~92%

#--- hyperparameters ---
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.03


#--- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'



# --- Dataset initialization ---

# We transform image files' contents to tensors
# Plus, we can add random transformations to the training data if we like
# Think on what kind of transformations may be meaningful for this data.
# Eg., horizontal-flip is definitely a bad idea for sign language data.
# You can use another transformation here if you find a better one.
train_transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
dev_set   = datasets.ImageFolder(DATA_DIR % 'dev',   transform=test_transform)
test_set  = datasets.ImageFolder(DATA_DIR % 'test',  transform=test_transform)


# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.dropout = nn.Dropout(0.05)
        self.features = nn.Sequential(
          nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.BatchNorm2d(16),
          nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.BatchNorm2d(32)
        )

        self.classify = nn.Sequential(
          nn.Linear(32 * 7 * 7, 60),
          nn.Linear(60, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.classify(x)
        return x


model = CNN(NUM_CLASSES)
print(model)

#--- set up ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.01)

#--- training ---
dev_error = 1
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        result = model.forward(data)
        probs = torch.softmax(result, dim=1)
        winners = probs.argmax(dim=1)
        optimizer.zero_grad()
        loss = loss_function(result, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (winners == target).sum().item()
        total = total + BATCH_SIZE_TRAIN
        print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
              (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
               100. * train_correct / total, train_correct, total))

    dev_loss = 0
    dev_correct = 0
    dev_total = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(dev_loader):
            data, target = data.to(device), target.to(device)
            result = model.forward(data)
            loss = loss_function(result, target)
            dev_loss += loss
            probs = torch.softmax(result, dim=1)
            winners = probs.argmax(dim=1)
            dev_correct += (winners == target).sum().item()
            total += BATCH_SIZE_TEST
            print('Evaluating with dev_set: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
                  (batch_num, len(dev_loader), dev_loss / (batch_num + 1), 
                  100. * dev_correct / total, dev_correct, total))
    if (1 - dev_correct/total) > dev_error:
        print("The dev_error was bigger in this one ", (1 - dev_correct/total), dev_error)
        break
    else:
        dev_error = (1 - dev_correct/total)
        print("new dev_error is ", dev_error)



#--- test ---
test_loss = 0
test_correct = 0
total = 0

with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # WRITE CODE HERE
        result = model.forward(data)
        loss = loss_function(result, target)
        test_loss += loss
        probs = torch.softmax(result, dim=1)
        winners = probs.argmax(dim=1)
        test_correct += (winners == target).sum().item()
        total += BATCH_SIZE_TEST
        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
              (batch_num, len(test_loader), test_loss / (batch_num + 1), 
               100. * test_correct / total, test_correct, total))


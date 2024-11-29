from FInal.preprocess import DataModule
from FInal.config_reader import Config
import torch

config = Config()
print(config.train)

dataModule = DataModule(config.train, config.preprocess)
dataModule.setup()
train_set = dataModule.train

print(len(train_set))

# train_set = train_set[:100]

# print(len(train_set))

print("Hello")

print(train_set[0])
print(train_set[1])
print(train_set[0].shape)

batch = train_set[0:2]
batch = torch.stack(batch)

print(batch)
print(batch.shape)

print("====================")

x = batch[:, :2048]
y = batch[:, 1:].long()
print(x)
print(y)
print(x.shape)

# for train_set[0] in train_loader:
#     print(train_set[0])
#     break

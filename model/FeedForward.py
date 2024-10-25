import torch.nn as nn
from model.Normalizations import LinearZeroMeanOutput
from utils.get_activation import getActivation



class FeedForward(nn.Module):

    def __init__(self, batchSize, contextLength, embeddingDim, dropout, dtype, config):
        super(FeedForward, self).__init__()

        self.batchSize = batchSize
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dropout = dropout
        self.dtype = dtype
        self.config = config

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.inTranform = nn.Linear(embeddingDim, 4*embeddingDim, dtype=self.dtype)
        self.activation = getActivation(config)

        if(self.config["LinearZeroMean"]):
            self.outTransform = LinearZeroMeanOutput(4*embeddingDim, embeddingDim, dtype=self.dtype)
        else:
            self.outTransform = nn.Linear(4*embeddingDim, embeddingDim, dtype=self.dtype)

    def forward(self, x):
        x = self.inTranform(x)
        x = self.dropoutLayer(x)
        x = self.activation(x)
        x = self.dropoutLayer(x)
        x = self.outTransform(x)
        return x

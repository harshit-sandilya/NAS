import torch.nn as nn
import torch
from model.FeedForward import FeedForward
from model.MultiHeadAttention import MultiHeadAttention
from model.Normalizations import LayerNorm, RMSNorm, cRMSNorm

from model.MoE2 import MoE

class DecoderBlock(nn.Module):
    def __init__(
        self,
        batchSize,
        contextLength,
        embeddingDim,
        numHeads,
        dropout,
        dtype,
        config
    ):
        super(DecoderBlock, self).__init__()

        self.batchSize = batchSize
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.numHeads = numHeads
        self.dropout = dropout
        self.dtype = dtype
        self.config = config

        if(self.config["norm_type"].lower() == "rms"):
            self.normalisation_mha = RMSNorm(eps=self.config["norm_eps"])

        elif(self.config["norm_type"].lower() == "crms"):
            self.normalisation_mha = cRMSNorm(eps=self.config["norm_eps"])
        else:
            self.normalisation_mha = LayerNorm(self.embeddingDim,eps=self.config["norm_eps"])

        self.MHA = MultiHeadAttention(
            self.batchSize,
            self.contextLength,
            self.embeddingDim,
            self.numHeads,
            self.dropout,
            self.dtype,
            self.config,
        )
        
        if(self.config["norm_type"].lower() == "rms"):
            self.normalisation_ffn = RMSNorm(eps=self.config["norm_eps"])

        elif(self.config["norm_type"].lower() == "crms"):
            self.normalisation_ffn = cRMSNorm(eps=self.config["norm_eps"])
        else:
            self.normalisation_ffn = LayerNorm(self.embeddingDim,eps=self.config["norm_eps"])

        # self.FF = FeedForward(
        #     self.batchSize,
        #     self.contextLength,
        #     self.embeddingDim,
        #     self.dropout,
        #     self.dtype,
        #     self.config
        # )
        
        self.FF = MoE(self.embeddingDim,4,2)

        

    def forward(self, x):
        
        if(self.config["decoder_architechture"] == "norm_rearrange"):
            x = x + self.normalisation_mha(self.MHA(x))
            x = x + self.normalisation_ffn(self.FF(x))
        elif(self.config["decoder_architechture"] == "post_norm"):
            x = self.normalisation_mha(x + self.MHA(x))
            x = self.normalisation_ffn(x + self.FF(x))
        elif(self.config["decoder_architechture"] == "gpt_j_residual"):
            
            x1,_ = self.FF(self.normalisation_ffn(x))
            x = x + self.MHA(self.normalisation_mha(x))  + x1
        else:
            # print("\n input to mha : \n",x )
            # print("\n<<< x after embed before attention >>>\n",x.shape)
            x = x + self.MHA(self.normalisation_mha(x))
            # print("\n<<< x after attention before ffn >>>\n",x)
            # print("\n before layernorm after attention : \n",x )

            x1 = self.normalisation_ffn(x)
            # print("\n<<< x after attention after prelayernorm of ffn >>>\n",x1)
            x2,_ = self.FF(x1)
            # print("\n<<< x after  ffn >>>\n",x2)
            x = x + x2
            # print("\n<<< x after attention after  ffn residual >>>\n",x)
    
        return x

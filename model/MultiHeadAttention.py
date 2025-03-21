import torch
import torch.nn as nn

from model.Attention import scaledDotProductAttention
from model.PositionalEncoding import Learned, RoPE, Cosine, RotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self, batchSize, contextLength, embeddingDim, numHeads, dropout, dtype, config
    ):
        super(MultiHeadAttention, self).__init__()

        assert embeddingDim % numHeads == 0
        self.batchSize = batchSize
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.numHeads = numHeads
        self.dropout = dropout
        self.dtype = dtype
        self.config = config
        self.external_dtype = dtype

        self.headDim = embeddingDim // numHeads
        self.mask = torch.triu(torch.ones(contextLength, contextLength), diagonal=1)
        self.mask = self.mask.masked_fill(self.mask == 1, float(-1e9))
        self.mask = self.mask.to(self.dtype)

        # self.Wq = nn.Linear(embeddingDim, embeddingDim, bias=False, dtype=self.dtype)
        # self.Wk = nn.Linear(embeddingDim, embeddingDim, bias=False, dtype=self.dtype)
        # self.Wv = nn.Linear(embeddingDim, embeddingDim, bias=False, dtype=self.dtype)

        self.Wqkv = nn.Linear(embeddingDim, 3 * embeddingDim, dtype=self.dtype)

        self.Wo = nn.Linear(embeddingDim, embeddingDim, dtype=self.dtype)
        self.attention = scaledDotProductAttention(
            contextLength, self.headDim, self.dropout, self.dtype, self.headDim
        )

        if config["positional_encoding"] == "rope":
            self.pe = RoPE(self.contextLength, self.embeddingDim, self.external_dtype)
        elif config["positional_encoding"] == "rotary":
            self.pe = RotaryEmbedding(
                self.headDim,
                base=10000,
                max_seq_len=config["context_length"],
                precision=self.external_dtype,
                save_inv_freqs=False,
            )
        else:
            self.pe = Learned(
                self.contextLength, self.embeddingDim, self.external_dtype
            )

    def splitHeads(self, x):
        # (batch,seqlen,numHeads,headDim)
        x = x.reshape(-1, self.contextLength, self.numHeads, self.headDim)

        # (batch,seqlen,numHeads,headDim) -> (batch,numHeads,seqlen,headDim)
        x = x.transpose(1, 2)
        return x

    def combineHeads(self, x):
        # (batch*numHeads,seqlen,headDim) -> (seqlen , batch*numHeads, headDim)
        x = x.transpose(0, 1)

        # (seqlen , batch*numHeads, headDim) -> (seqlen , batch, numHeads * headDim)
        x = x.reshape(self.contextLength, -1, self.embeddingDim)
        return x

    def forward(self, x):

        qkv = self.Wqkv(x)
        qkv = qkv.reshape(self.contextLength, -1, self.numHeads, 3 * self.headDim)

        q, k, v = qkv.split(self.headDim, -1)

        q = self.splitHeads(q)
        k = self.splitHeads(k)
        v = self.splitHeads(v)

        if self.config["positional_encoding"] == "rotary":

            query_rot = q[..., : self.headDim]
            key_rot = k[..., : self.headDim]

            # (seqlen,batch,numHeads,headDim)
            query_rot = query_rot.permute(2, 0, 1, 3)
            key_rot = key_rot.permute(2, 0, 1, 3)
            v = v.permute(2, 0, 1, 3)

            cos, sin = self.pe(v, seq_len=self.contextLength)
            q, k = self.pe.apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=0)

            # (seqlen,batch,numHeads,headDim)
            q = q.reshape_as(v)
            k = k.reshape_as(v)

        else:

            # (seqlen,batch,numHeads,headDim)
            q = q.permute(2, 0, 1, 3)
            k = k.permute(2, 0, 1, 3)
            v = v.permute(2, 0, 1, 3)

        out = self.attention(q, k, v, self.mask)
        out = self.combineHeads(out)
        out = self.Wo(out)

        # (seqlen , batch, numHeads * headDim)  ->  (batch, seqlen,  numHeads * headDim)
        out = out.transpose(0, 1)

        return out

import torch
from torch import nn
import torch.nn.functional as F

import random, math

class SelfAttention(nn.Module):
    """
        MultiHead Attention Layer.

        Implementation modified from OpenNMT-py.
            https://github.com/OpenNMT/OpenNMT-py

        :param size: Size of the model embeddings.
        :param heads: Number of heads of the model.

        Note:
            
        """
    def __init__(self, size: int = 128, heads: int = 4) -> None:
        super().__init__()
        assert size % heads == 0

        self.head_size = head_size = size // heads
        self.model_size = size
        self.heads = heads

        self.k_layer = nn.Linear(size, heads * head_size)
        self.v_layer = nn.Linear(size, heads * head_size)
        self.q_layer = nn.Linear(size, heads * head_size)
        self.output_layer = nn.Linear(size, size)
        
    def forward(self, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                q: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        :param k: Key vectors [batch_size x seq_len x embedding_size]
        :param v: Value vectors [batch_size x seq_len x embedding_size]
        :param q: Value vectors [batch_size x seq_len x embedding_size]
        :param mask: Mask that will 'remove' the attention from some 
            of the key, value vectors. [batch_size x 1 x key_len]
        
        Note: The seq_len of the key, value vectors must be the same but it 
            can differ from the seq_len of the queries.

        :return:
            - Returns a [batch x seq_len x embedding_size] with the contextualized 
                representations of the queries.
        """
        batch_size = k.size(0)
        heads = self.heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, heads, ..]
        k = k.view(batch_size, -1, heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, heads, self.head_size).transpose(1, 2)

        # compute scaled dot-product self-attention
        q = q / math.sqrt(self.head_size)
        
        # for each word Wi the score with all other words Wj 
        # for all heads inside the batch
        # [batch x heads x query_len x key_len]
        dot = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # We add a dimension for the heads to it below: [batch, 1, 1, seq_len]
        if mask is not None:
            dot = dot.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention to convert the dot scores into probabilities.
        attention = F.softmax(dot, dim=-1)

        # We multiply the probabilities with the respective values
        context = torch.matmul(attention, v)
        # Finally, we reshape back to [batch x seq_len x heads * embedding_size]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, heads * self.head_size)
        # We unify the heads by appliying a linear transform from:
        # [batch x seq_len x heads * embedding_size] -> [batch x seq_len x embedding_size]        
        return self.output_layer(context)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block 
        Self attention -> Layer Norm -> Feed Forward -> Layer Norm

    :param emb_size: Size of the model embeddings.
    :param heads: Number of heads of the model.
    :param ff_hidden_mult: Int that will specify the size of the 
        feed forward layer as a multiple of the embedding size.
    :param dropout: Dropout value to be applied between layers.
    """
    def __init__(self, 
                 emb_size: int = 128, 
                 heads: int = 4, 
                 ff_hidden_mult: int = 4, 
                 dropout: float = 0.0) -> None:
        super().__init__()

        self.attention = SelfAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_mult * emb_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb_size, emb_size)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes a sequence by passing it through 4 blocks: 
            Self attention -> Layer Norm -> Feed Forward -> Layer Norm
        
        :param x: Sequence to be encoded. [batch_size x seq_len x embedding_size]
        :param mask: Mask to be passed to the SelfAttention Block -> check SelfAttention.

        :returns:
            - Returns the encoding of x. [batch_size x seq_len x embedding_size]
        """
        # Self Attention Block
        attended = self.attention(x, x, x, mask)

        # Normalization Block
        x = self.norm1(attended + x)
        x = self.do(x)

        # Feedforward Block
        fedforward = self.ff(x)

        # Normalization Block
        x = self.norm2(fedforward + x)        
        return self.do(x)

class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder Block 
        Self Attention -> Layer Norm -> Encoder Attention -> Layer Norm -> Feed Forward -> Layer Norm

    :param emb_size: Size of the model embeddings.
    :param heads: Number of heads of the model.
    :param ff_hidden_mult: Int that will specify the size of the 
        feed forward layer as a multiple of the embedding size.
    :param dropout: Dropout value to be applied between layers.
    """
    def __init__(self, 
                 emb_size: int = 128, 
                 heads: int = 4, 
                 ff_hidden_mult: int = 4, 
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.self_attn = SelfAttention(emb_size, heads) 
        self.src_attn = SelfAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_mult * emb_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb_size, emb_size)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: torch.Tensor, 
                trg_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes a sequence by passing it through 6 blocks: 
            Self Attention -> Layer Norm -> Encoder Attention -> Layer Norm -> Feed Forward -> Layer Norm
        
        :param x: Sequence to be encoded. [batch_size x seq_len x embedding_size]
        :param memory: Sequence to be used as a 'memory'. [batch_size x src_seq_len x embedding_size]

        :param src_mask: Mask to be passed to the SelfAttention Block when computing attention
            over the source sequence/memory. -> check SelfAttention.
        :param trg_mask: Mask to be passed to the SelfAttention Block when computing self attention
            -> check SelfAttention.

        :returns:
            - Returns the encoding of x. [batch_size x seq_len x embedding_size]
        """

        # Self Attention Block
        attended_trg = self.self_attn(x, x, x, trg_mask)

        # Normalization Block
        x = self.norm1(attended_trg + x)
        x = self.do(x)
        
        # Encoder-Decoder Attention Block
        attended_src = self.src_attn(memory, memory, x, src_mask)

        # Normalization Block
        x = self.norm1(attended_src + x)
        x = self.do(x)

        # Feedforward Block
        fedforward = self.ff(x)

        # Normalization Block
        x = self.norm3(fedforward + x)
        return self.do(x)
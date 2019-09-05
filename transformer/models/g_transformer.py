import torch
from torch import nn
import torch.nn.functional as F

from models import TransformerEncoderBlock, TransformerDecoderBlock
from utils import d, subsequent_mask

class GTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, 
                 vocab_size: int, 
                 emb_size: int = 128, 
                 heads: int = 6, 
                 depth: int = 4, 
                 seq_length: int = 30, 
                 dropout: float = 0.0):
        """
        :param vocab_size: Size of the output vocab.
        :param emb_size: Embedding dimension
        :param heads: Number of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param dropout: Dropout to be applied between layers.
        """
        super().__init__()
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=vocab_size)
        self.pos_embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=seq_length)

        self.encoding_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_size=emb_size, heads=heads, dropout=dropout)
                for _ in range(depth)])

        self.decoding_blocks = nn.ModuleList([
            TransformerDecoderBlock(emb_size=emb_size, heads=heads, dropout=dropout)
                for _ in range(depth)])

        self.do = nn.Dropout(dropout)
        self.toprobs = nn.Linear(emb_size, vocab_size)

    def encode(self, 
               src: torch.Tensor, 
               src_mask: torch.Tensor) -> torch.Tensor:
        """
        Function that encodes the source sequence.

        :param src: Our vectorized source sequence. [Batch_size x seq_len]
        :param src_mask: Mask to be passed to the SelfAttention Block when encoding 
                the src sequence-> check SelfAttention.
        
        :returns:
            -  Returns the source sequence embedded [Batch_size x seq_len x embedding_size]
        """
        tokens = self.token_embedding(src)
        b, t, e = tokens.size()
        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)

        embed = tokens + positions
        embed = self.do(embed)

        for block in self.encoding_blocks:
            embed = block(embed, src_mask)
        return embed

    def decode(self, 
               trg: torch.Tensor, 
               trg_mask: torch.Tensor,
               memory: torch.Tensor,
               src_mask: torch.Tensor) -> torch.Tensor:

        """
        Function that encodes the target sequence.

        :param trg: Our vectorized target sequence. [Batch_size x trg_seq_len]
        :param trg_mask: Mask to be passed to the SelfAttention Block when encoding 
                the trg sequence-> check SelfAttention.
        :param memory: Our src sequence encoded by the encoding function. This will be used
                as a memory over the source sequence.
        :param src_mask: Mask to be passed to the SelfAttention Block when computing attention
                over the source sequence/memory. -> check SelfAttention.

        :returns:
            -  Returns the log probabilities of the next word over the entire target sequence.
                [Batch_size x trg_seq_len x vocab_size]
        """
        tokens = self.token_embedding(trg)
        b, t, e = tokens.size()
            
        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        trg_mask = trg_mask & subsequent_mask(t).type_as(trg_mask)
        for block in self.decoding_blocks:
            x = block(x, memory, src_mask, trg_mask)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.vocab_size)
        return F.log_softmax(x, dim=2)


    def forward(self, 
                src: torch.Tensor, 
                trg: torch.Tensor, 
                src_mask: torch.Tensor, 
                trg_mask: torch.Tensor) -> torch.Tensor:
        """
        Function that
        :param src: Our vectorized source sequence. [Batch_size x src_seq_len]
        :param trg: Our vectorized target sequence. [Batch_size x trg_seq_len]
        :param src_mask: Mask to be passed to the SelfAttention Block when computing attention
                over the source sequence/memory. -> check SelfAttention.
        :param trg_mask: Mask to be passed to the SelfAttention Block when encoding 
                the trg sequence-> check SelfAttention.

        :returns:
            -  Returns the log probabilities of the next word over the entire target sequence.
                [Batch_size x trg_seq_len x vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(trg, trg_mask, memory, src_mask)

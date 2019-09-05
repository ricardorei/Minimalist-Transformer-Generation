import math
import pickle

import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from models import GTransformer
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.metrics import get_token_accuracy
from torchnlp.utils import lengths_to_mask, collate_tensors
from utils import get_iterators, prepare_sample, set_seed

# Samples used to test the model.
SAMPLES = [
    {'source': '18 17 4 11 19 15 18 2', 'target': '2 18 15 19 11 4 17 18'}, 
    {'source': '0 15 8 17 7 6 15 17 17 15 12 4 7 4 16 12 0 2 5 18', 'target': '18 5 2 0 12 16 4 7 4 12 15 17 17 15 6 7 17 8 15 0'}, 
    {'source': '9 0', 'target': '0 9'},
    {'source': '15 19 12 13 12 18 14 4 11', 'target': '11 4 14 18 12 13 12 19 15'}, 
    {'source': '1 4 15 6', 'target': '6 15 4 1'}, 
    {'source': '13 9 13 16 12 18 11 17 18', 'target': '18 17 11 18 12 16 13 9 13'}, 
    {'source': '18 7 10 0 8 19 5 10 17 18 18 3 6 18', 'target': '18 6 3 18 18 17 10 5 19 8 0 10 7 18'}, 
    {'source': '9 3 2 15 15 2 11 2 13', 'target': '13 2 11 2 15 15 2 3 9'}, 
    {'source': '0 9 13 13 3', 'target': '3 13 13 9 0'}, 
    {'source': '19 19', 'target': '19 19'}
]

PAD_IDX = 0
INK_UDX = 1
EOS_IDX = 2
BOS_IDX = 3

def train_manager(configs: dict) -> None :
    """
    Model Training functions.
    :param configs: Dictionary with the configs defined in default.yaml
    """
    with open('.preprocess.pkl', 'rb') as preprocess_file:
            text_encoder, train, test = pickle.load(preprocess_file)

    set_seed(configs.get('seed', 3))
    print (f'- nr. of training examples {len(train)}')
    print (f'- nr. of test examples {len(test)}')
    print (f'- vocab size: {text_encoder.vocab_size}')

    # Build Transformer model
    model = GTransformer(
                emb_size=configs.get('embedding_size', 128), 
                heads=configs.get('num_heads', 8), 
                depth=configs.get('depth', 6), 
                seq_length=configs.get('max_length', 1000), 
                vocab_size=text_encoder.vocab_size
            )
    model.cuda()

    # Build Optimizer
    opt = torch.optim.Adam(lr=configs.get('lr', 0.0001), params=model.parameters())

    # Training Loop
    model = train_loop(configs, model, opt, train, test, text_encoder)
        
    # Now that the model is trained lets try to see what is the model output!
    sample = collate_tensors(SAMPLES)
    src_seqs, src_lengths = text_encoder.batch_encode(sample['source'])
    src_mask = lengths_to_mask(src_lengths).unsqueeze(1)
    ys, lengths = greedy_decode(model, src_seqs, src_mask)
    ys = text_encoder.batch_decode(ys, lengths)
    for i in range(len(SAMPLES)):
        print ('\nTarget: {}\nModel:  {}'.format(SAMPLES[i]['target'], ys[i]))

def train_loop(
        configs: dict, 
        model: GTransformer, 
        opt: torch.optim.Adam, 
        train: Dataset, 
        test: Dataset, 
        text_encoder: WhitespaceEncoder) -> GTransformer:
    """
    Main training loop.

    :param configs: Configs defined on the default.yaml file.
    :param model: Sequence-to-sequence transformer.
    :param opt: Adam optimizer.
    :param train: The dataset used for training.
    :param test: The dataset used for validation.
    :param text_encoder: Torch NLP text encoder for tokenization and vectorization.
    """
    for e in range(configs.get('num_epochs', 8)):
        print(f'\n Epoch {e}')
        model.train()
        
        nr_batches = math.ceil(len(train)/configs.get('batch_size', 8))
        train_iter, test_iter = get_iterators(configs, train, test)
        total_loss, steps = 0, 0

        for sample in tqdm.tqdm(train_iter, total=nr_batches):
            # 0) Zero out previous grads
            opt.zero_grad()

            # 1) Prepare Sample
            src, src_lengths, trg, shifted_trg, trg_lengths = prepare_sample(sample, text_encoder)
            

            # 2) Run model
            lprobs = model(
                src=src.cuda(), 
                trg=shifted_trg.cuda(),
                src_mask=lengths_to_mask(src_lengths).unsqueeze(1).cuda(),
                trg_mask=lengths_to_mask(trg_lengths).unsqueeze(1).cuda()
            )

            # 3) Compute loss
            loss = F.nll_loss(lprobs.transpose(2, 1), trg.cuda(), reduction='mean')
            loss.backward()

            # 4) Update training metrics
            total_loss += float(loss.item())
            steps += int(trg.ne(0).sum())

            # 5) clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if configs.get('gradient_clipping', -1) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), configs.get('gradient_clipping'))
            
            # 6) Optim step
            opt.step()

        print(f'-- total train loss {total_loss:.4}')
        total_steps = steps*(e+1)
        print(f'-- train steps {total_steps}')
        validate(model, test_iter, text_encoder)
    return model

def validate(model: GTransformer, iterator, text_encoder: WhitespaceEncoder) -> None:
    """
    Function that computes the loss over the validation set.

    :param model: Sequence-to-sequence transformer model.
    :param iterator: Iterator object over the test Dataset.
    :param text_encoder: Torch NLP text encoder for tokenization and vectorization. 
    """
    total_loss, steps = 0, 0
    # Testing
    with torch.no_grad():
        model.train(False)
        for sample in iterator:
            # 1) Prepare Sample
            src, src_lengths, trg, shifted_trg, trg_lengths = prepare_sample(sample, text_encoder)
            # 2) Run model
            lprobs = model(
                src=src.cuda(), 
                trg=shifted_trg.cuda(),
                src_mask=lengths_to_mask(src_lengths).unsqueeze(1).cuda(),
                trg_mask=lengths_to_mask(trg_lengths).unsqueeze(1).cuda()
            )
            # 3) Compute loss
            loss = F.nll_loss(lprobs.transpose(2, 1), trg.cuda(), reduction='mean')
            # 4) Update training metrics
            total_loss += float(loss.item())
            steps += int(trg.ne(PAD_IDX).sum())
    print(f'-- total test loss {total_loss:.4}')
    print(f'-- test steps {steps}')

def greedy_decode(
        model: GTransformer, 
        src_input: torch.Tensor, 
        src_mask: torch.Tensor, 
        max_length: int = 30) -> (torch.Tensor, torch.Tensor):
    """
    Greedy Search Decoding function.

    :param model: Sequence-to-sequence transformer model.
    :param src_input: Input to feed the transformer model.
    :param src_mask: Mask for the source sequence.
    :param max_length: Maximum number of decoding steps.
    """
    batch_size = src_mask.size(0)
    with torch.no_grad():
        encoder_out =  model.encode(src_input.cuda(), src_mask.cuda())
        # start with BOS-symbol for each sentence in the batch
        ys = encoder_out.new_full([batch_size, 1], BOS_IDX, dtype=torch.long)
        # a subsequent mask is intersected with this in decoder forward pass
        ys_mask = (ys != PAD_IDX).unsqueeze(1)
        for _ in range(max_length-1):
            lprobs = model.decode(ys, ys_mask.cuda(), encoder_out, src_mask.cuda())
            lprobs = lprobs[:, -1]
            _, next_word = torch.max(lprobs, dim=1)
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            ys_mask = (ys != PAD_IDX).unsqueeze(1)

        ys = ys[:, 1:]  # remove BOS-symbol
    # we ignore output tokens that are 0 (pad), 1 (unk), 2 (eos)
    ys_lengths = [l.index(2) for l in ys.cpu().tolist()]
    return ys, ys_lengths

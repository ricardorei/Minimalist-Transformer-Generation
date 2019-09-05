import numpy as np
import torch

from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.samplers.bucket_batch_sampler import BucketBatchSampler
from torchnlp.utils import collate_tensors, sampler_to_iterator


def set_seed(seed: int, cuda: bool=True) -> None:
    """
    Sets a numpy and torch seeds.
    :param seed: the seed value.
    :param cuda: if True sets the torch seed directly in cuda.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def d(tensor: torch.Tensor = None) -> torch.Tensor:
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def subsequent_mask(size: int) -> torch.Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.
    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0

def prepare_sample(
        sample: dict, 
        text_encoder: WhitespaceEncoder
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Function that receives a sample from the Dataset iterator and prepares t
    he input to feed the transformer model.

    :param sample: dictionary containing the inputs to build the batch 
        (e.g: [{'source': '9 0', 'target': '0 9'}, {'source': '34 3 4', 'target': '4 3 34'}])
    :param text_encoder: Torch NLP text encoder for tokenization and vectorization.
    """
    sample = collate_tensors(sample)
    input_seqs, input_lengths   = text_encoder.batch_encode(sample['source'])
    target_seqs, target_lengths = text_encoder.batch_encode(sample['target'])
    # bos tokens to initialize decoder
    bos_tokens = torch.full([target_seqs.size(0), 1], text_encoder.stoi['<s>'], dtype=torch.long)
    shifted_target = torch.cat((bos_tokens, target_seqs[:, :-1]), dim=1)
    return input_seqs, input_lengths, target_seqs, shifted_target, target_lengths

def get_iterators(configs, train, test):
    """
    Function that receives the training and testing Datasets and build an iterator over them.

    :param configs: dictionary containing the configs from the default.yaml file.
    :param train: Dataset obj for training.
    :param test: Dataset obj for testing.
    """
    train_sampler = BucketBatchSampler(data=train,
            sort_key=lambda i: len(i['source'].split()),
            batch_size=configs['batch_size'], 
            drop_last=False
        )
    test_sampler = BucketBatchSampler(data=test,
            sort_key=lambda i: len(i['source'].split()),
            batch_size=configs.get('batch_size', 4), 
            drop_last=False
        )
    train_iter = sampler_to_iterator(train, train_sampler)
    test_iter = sampler_to_iterator(test, test_sampler)
    
    return train_iter, test_iter

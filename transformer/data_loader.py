import random
from torchnlp.datasets.dataset import Dataset


def reverse_dataset(
        train_rows=10000, 
        dev_rows=1000, 
        test_rows=1000, 
        seq_max_length=20,
        random_seed: int = 3) -> (Dataset, Dataset, Dataset):
    """
    This Reverse dataset is used for testing generation models.
    The task consists of given a sequence of integers, revert the sequence.
    :param train_rows: the number of rows for training to be generated.
    :param dev_rows: the number of development rows to be generated.
    :param test_rows: the number of test rows to be generated.
    :param seq_max_length: the max sequence lengths to be generated.
    """
    ret = []
    random.seed(random_seed)
    for n_rows in [train_rows, dev_rows, test_rows]:
        rows = []
        for i in range(n_rows):
            length = random.randint(1, seq_max_length)
            seq = []
            for _ in range(length):
                seq.append(str(random.randint(0, seq_max_length-1)))
            input_ = ' '.join(seq)
            output = ' '.join(reversed(seq))+ ' </s>'
            rows.append({'source': input_, 'target': output})
        ret.append(Dataset(rows))
    assert len(ret) == 3
    return ret[0], ret[1], ret[2]

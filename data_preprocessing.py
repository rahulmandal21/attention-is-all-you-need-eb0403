import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math

class DataPreprocessing:
    """
    A class used to preprocess data for a transformer model.

    Attributes:
    ----------
    d_model : int
        The dimension of the model.
    max_len : int
        The maximum length of a sequence.
    """

    def __init__(self, d_model: int, max_len: int):
        """
        Initializes the DataPreprocessing class.

        Parameters:
        ----------
        d_model : int
            The dimension of the model.
        max_len : int
            The maximum length of a sequence.
        """
        self.d_model = d_model
        self.max_len = max_len

    def create_positional_encodings(self) -> torch.Tensor:
        """
        Creates positional encodings for a sequence.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (max_len, d_model) containing the positional encodings.
        """
        pe = torch.zeros(self.max_len, self.d_model)
        for pos in range(self.max_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                if i + 1 < self.d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / self.d_model)))
        return pe

    def create_embedding(self, vocab_size: int) -> nn.Embedding:
        """
        Creates a learned embedding for a vocabulary.

        Parameters:
        ----------
        vocab_size : int
            The size of the vocabulary.

        Returns:
        -------
        nn.Embedding
            A learned embedding for the vocabulary.
        """
        return nn.Embedding(vocab_size, self.d_model)

    def preprocess_data(self, sequences: list) -> torch.Tensor:
        """
        Preprocesses a list of sequences by converting them to tensors and adding positional encodings.

        Parameters:
        ----------
        sequences : list
            A list of sequences.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (batch_size, max_len, d_model) containing the preprocessed sequences.
        """
        pe = self.create_positional_encodings()
        embedding = self.create_embedding(max(max(seq) for seq in sequences) + 1)
        tensors = [embedding(torch.tensor(seq)) for seq in sequences]
        padded_tensors = pad_sequence(tensors, batch_first=True, padding_value=0)
        return padded_tensors + pe[:padded_tensors.shape[1], :]

class TokenizedTextDataset(Dataset):
    """
    A dataset class for tokenized text sequences.
    """

    def __init__(self, sequences: list):
        """
        Initializes the TokenizedTextDataset class.

        Parameters:
        ----------
        sequences : list
            A list of tokenized text sequences.
        """
        self.sequences = [torch.tensor(seq) for seq in sequences]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        -------
        int
            The length of the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a tensor representing a sequence in the dataset.

        Parameters:
        ----------
        idx : int
            The index of the sequence.

        Returns:
        -------
        torch.Tensor
            A tensor representing the sequence.
        """
        return self.sequences[idx]

def collate_fn(batch: list) -> torch.Tensor:
    """
    A function to collate a batch of sequences.

    Parameters:
    ----------
    batch : list
        A list of sequences.

    Returns:
    -------
    torch.Tensor
        A tensor representing the batch of sequences.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)

if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(512, 100)
    sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    dataset = TokenizedTextDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        preprocessed_batch = data_preprocessing.preprocess_data([seq.tolist() for seq in batch])
        print(preprocessed_batch.shape)
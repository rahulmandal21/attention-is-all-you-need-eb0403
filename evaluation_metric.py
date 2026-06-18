import sacrebleu
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

class BLEUEvaluator:
    """
    A class to evaluate the performance of a machine translation model using the BLEU score.
    """

    def __init__(self):
        """
        Initializes the BLEUEvaluator class.
        """
        pass

    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> float:
        """
        Computes the BLEU score for a list of predictions and references.

        Args:
        predictions (List[str]): A list of predicted translations.
        references (List[List[str]]): A list of reference translations.

        Returns:
        float: The BLEU score.
        """
        bleu = sacrebleu.corpus_bleu(predictions, references)
        return bleu.score

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Tuple[float, float, float, float]:
        """
        Evaluates the performance of a model using the BLEU score.

        Args:
        predictions (List[str]): A list of predicted translations.
        references (List[List[str]]): A list of reference translations.

        Returns:
        Tuple[float, float, float, float]: The BLEU score and its components.
        """
        bleu = sacrebleu.corpus_bleu(predictions, references)
        return bleu.score, bleu.precisions[0], bleu.precisions[1], bleu.precisions[2]

class LabelSmoothingLoss(nn.Module):
    """
    A class to compute the cross-entropy loss with label smoothing.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initializes the LabelSmoothingLoss class.

        Args:
        num_classes (int): The number of classes.
        smoothing (float, optional): The smoothing factor. Defaults to 0.1.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss with label smoothing.

        Args:
        predictions (torch.Tensor): The predicted logits.
        targets (torch.Tensor): The target labels.

        Returns:
        torch.Tensor: The loss.
        """
        return self.criterion(predictions, targets)

class TokenizedTextDataset(Dataset):
    """
    A class to represent a dataset of tokenized text sequences.
    """

    def __init__(self, sequences: List[List[int]]):
        """
        Initializes the TokenizedTextDataset class.

        Args:
        sequences (List[List[int]]): A list of tokenized text sequences.
        """
        self.sequences = [torch.tensor(seq) for seq in sequences]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        int: The length of the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a tokenized text sequence at the given index.

        Args:
        idx (int): The index of the sequence.

        Returns:
        torch.Tensor: The tokenized text sequence.
        """
        return self.sequences[idx]

def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    A function to collate a batch of tokenized text sequences.

    Args:
    batch (List[torch.Tensor]): A list of tokenized text sequences.

    Returns:
    torch.Tensor: The collated batch.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)

if __name__ == "__main__":
    evaluator = BLEUEvaluator()
    predictions = ["This is a test prediction", "This is another test prediction"]
    references = [["This is a reference translation", "This is another reference translation"], ["This is another reference translation"]]
    bleu_score = evaluator.compute_bleu(predictions, references)
    print(f"BLEU score: {bleu_score}")

    loss_fn = LabelSmoothingLoss(num_classes=10)
    predictions = torch.randn(1, 10)
    targets = torch.randint(0, 10, (1,))
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss}")

    dataset = TokenizedTextDataset([[1, 2, 3], [4, 5, 6]])
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch)
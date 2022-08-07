from dataclasses import dataclass, field
from typing import Optional
from src.BaseDataTrainingArguments import BaseDataTrainingArguments

@dataclass
class MLMDataTrainingArguments(BaseDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

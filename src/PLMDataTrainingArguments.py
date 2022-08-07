from dataclasses import dataclass, field
from src.BaseDataTrainingArguments import BaseDataTrainingArguments

@dataclass
class PLMDataTrainingArguments(BaseDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    plm_probability: float = field(
        default=1/6,
        metadata={
            "help": (
                "Ratio of length of a span of masked tokens to surrounding context length for "
                "permutation language modeling."
            )
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union

from transformers.utils import PaddingStrategy
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


@dataclass
class IndicDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as
        # this method won't pad them and needs them of the same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            # fairseq by defaults right pad the labels for seq2seq tasks
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = feature["labels"] + remainder
                else:
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)

        self.tokenizer.padding_side = "left"
        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features

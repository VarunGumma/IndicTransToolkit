import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union, List, Dict


@dataclass
class IndicDataCollator:
    from transformers.utils import PaddingStrategy
    from transformers.tokenization_utils import PreTrainedTokenizerBase
    from transformers.data.data_collator import pad_without_fast_tokenizer_warning

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict], return_tensors: Optional[str] = None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Early return if no features
        if not features:
            return {}

        # Check if labels exist in the first feature
        has_labels = "labels" in features[0]

        if has_labels:
            self._pad_labels(features)

        # Set padding side once
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            # Pad features
            features = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                features,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors=return_tensors,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )

            # Prepare decoder input ids if needed
            if (
                has_labels
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
            ):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                    labels=features["labels"]
                )
                features["decoder_input_ids"] = decoder_input_ids

        finally:
            # Restore original padding side
            self.tokenizer.padding_side = original_padding_side

        return features

    def _pad_labels(self, features: List[Dict]) -> None:
        """Pad labels to the same length."""
        # Extract labels and find max length
        labels = [feature["labels"] for feature in features]
        max_label_length = max(len(label) for label in labels)

        # Adjust for pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad each label sequence
        for feature in features:
            current_labels = feature["labels"]
            padding_length = max_label_length - len(current_labels)

            if padding_length > 0:
                padding = [self.label_pad_token_id] * padding_length

                # Ensure consistent numpy array output
                if isinstance(current_labels, list):
                    feature["labels"] = np.array(
                        current_labels + padding, dtype=np.int64
                    )
                else:
                    feature["labels"] = np.concatenate(
                        [
                            current_labels.astype(np.int64),
                            np.array(padding, dtype=np.int64),
                        ]
                    )
            else:
                # Ensure numpy array type even when no padding needed
                if isinstance(current_labels, list):
                    feature["labels"] = np.array(current_labels, dtype=np.int64)
                else:
                    feature["labels"] = current_labels.astype(np.int64)

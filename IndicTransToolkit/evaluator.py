from typing import List, Union, Dict, Optional
from sacrebleu.metrics import CHRF, BLEU

from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


class IndicEvaluator:
    def __init__(self):
        # === Metrics ===
        self._chrf2_metric = CHRF(word_order=2)
        self._bleu_metric_13a = BLEU(tokenize="13a")
        self._bleu_metric_none = BLEU(tokenize="none")

        # === Normalizer factory and cache ===
        self._indic_norm_factory = IndicNormalizerFactory()
        self._normalizer_cache = {}  # Cache normalizers by iso_lang

        # === FLORES -> ISO codes ===
        # Using a more memory-efficient approach with fewer duplicates
        self._flores_codes = {
            "asm_Beng": "as",
            "awa_Deva": "hi",
            "ben_Beng": "bn",
            "bho_Deva": "hi",
            "brx_Deva": "hi",
            "doi_Deva": "hi",
            "eng_Latn": "en",
            "gom_Deva": "kK",
            "gon_Deva": "hi",
            "guj_Gujr": "gu",
            "hin_Deva": "hi",
            "hne_Deva": "hi",
            "kan_Knda": "kn",
            "kas_Arab": "ur",
            "kas_Deva": "hi",
            "kha_Latn": "en",
            "lus_Latn": "en",
            "mag_Deva": "hi",
            "mai_Deva": "hi",
            "mal_Mlym": "ml",
            "mar_Deva": "mr",
            "mni_Beng": "bn",
            "mni_Mtei": "hi",
            "npi_Deva": "ne",
            "ory_Orya": "or",
            "pan_Guru": "pa",
            "san_Deva": "hi",
            "sat_Olck": "or",
            "snd_Arab": "ur",
            "snd_Deva": "hi",
            "tam_Taml": "ta",
            "tel_Telu": "te",
            "urd_Arab": "ur",
            "unr_Deva": "hi",
        }

        # Pre-cache the trivial_tokenize function to avoid attribute lookup
        self._trivial_tokenize = indic_tokenize.trivial_tokenize

    def _get_normalizer(self, iso_lang: str):
        """
        Return a cached normalizer for a given iso_lang.
        """
        if iso_lang not in self._normalizer_cache:
            self._normalizer_cache[iso_lang] = self._indic_norm_factory.get_normalizer(
                iso_lang
            )
        return self._normalizer_cache[iso_lang]

    def _preprocess(self, sentences: List[str], lang: str) -> List[str]:
        """
        Preprocess the sentences using IndicNLP:
        1) Normalization (using a cached normalizer),
        2) Trivial tokenization.

        Optimized version with:
        - List comprehension for better performance
        - Pre-cached function references
        - Single-pass processing
        """
        iso_lang = self._flores_codes.get(lang, "hi")
        normalizer = self._get_normalizer(iso_lang)

        # Cache function references locally for faster access
        normalize_fn = normalizer.normalize
        tokenize_fn = self._trivial_tokenize

        # Use list comprehension for better performance
        # Combine all operations in a single pass
        return [
            " ".join(tokenize_fn(normalize_fn(line.strip()), iso_lang))
            for line in sentences
        ]

    def _read_file_efficiently(self, filepath: str) -> List[str]:
        """
        Read file more efficiently with better memory usage.
        """
        lines = []
        with open(filepath, "r", encoding="utf-8") as f:
            # Use readlines() for better performance on reasonable file sizes
            # For very large files, consider using a generator approach
            lines = [line.strip() for line in f.readlines()]
        return lines

    def evaluate(
        self,
        tgt_lang: str,
        preds: Union[List[str], str],
        refs: Union[List[str], str],
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Evaluate BLEU and chrF2++ scores for the given predictions and references.

        Optimized version with:
        - More efficient file reading
        - Reduced variable assignments
        - Direct dictionary construction
        """
        assert (
            preds is not None and refs is not None
        ), "Predictions and References cannot be None"

        # Convert file paths to lists if needed
        if isinstance(preds, str):
            preds = self._read_file_efficiently(preds)
        if isinstance(refs, str):
            refs = self._read_file_efficiently(refs)

        assert len(preds) == len(
            refs
        ), "Number of predictions and references do not match"

        # Direct evaluation based on language
        if tgt_lang != "eng_Latn":
            # Preprocess for non-English languages
            preds_processed = self._preprocess(preds, tgt_lang)
            refs_processed = self._preprocess(refs, tgt_lang)

            # Compute scores directly
            bleu_score = self._bleu_metric_none.corpus_score(
                preds_processed, [refs_processed]
            )
            chrf_score = self._chrf2_metric.corpus_score(
                preds_processed, [refs_processed]
            )

            return {
                "bleu": {
                    "score": round(bleu_score.score, 1),
                    "signature": self._bleu_metric_none.get_signature().format(),
                },
                "chrF2++": {
                    "score": round(chrf_score.score, 1),
                    "signature": self._chrf2_metric.get_signature().format(),
                },
            }
        else:
            # For English, use 13a tokenization
            bleu_score = self._bleu_metric_13a.corpus_score(preds, [refs])
            chrf_score = self._chrf2_metric.corpus_score(preds, [refs])

            return {
                "bleu": {
                    "score": round(bleu_score.score, 1),
                    "signature": self._bleu_metric_13a.get_signature().format(),
                },
                "chrF2++": {
                    "score": round(chrf_score.score, 1),
                    "signature": self._chrf2_metric.get_signature().format(),
                },
            }


# Optional: For very large files, you could add a generator-based approach
class IndicEvaluatorLargeFiles(IndicEvaluator):
    """
    Extended version for handling very large files with generator-based processing.
    """

    def _read_file_generator(self, filepath: str):
        """
        Generator for reading very large files line by line.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()

    def _preprocess_generator(self, sentences_gen, lang: str):
        """
        Generator-based preprocessing for memory efficiency with very large datasets.
        """
        iso_lang = self._flores_codes.get(lang, "hi")
        normalizer = self._get_normalizer(iso_lang)
        normalize_fn = normalizer.normalize
        tokenize_fn = self._trivial_tokenize

        for line in sentences_gen:
            yield " ".join(tokenize_fn(normalize_fn(line), iso_lang))

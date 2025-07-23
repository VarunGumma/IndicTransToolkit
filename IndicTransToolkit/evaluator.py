from pathlib import Path
from typing import List, Union, Dict, Generator
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


class IndicEvaluator:
    def __init__(self):
        from sacrebleu.metrics import CHRF, BLEU

        # === Metrics ===
        self._chrf2_metric = CHRF(word_order=2)
        self._bleu_metric_13a = BLEU(tokenize="13a")
        self._bleu_metric_none = BLEU(tokenize="none")

        # === Normalizer factory and cache ===
        self._indic_norm_factory = IndicNormalizerFactory()
        self._normalizer_cache = {}  # Cache normalizers by iso_lang

        # === FLORES -> ISO codes (using frozenset for immutable lookup) ===
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
        """Return a cached normalizer for a given iso_lang."""
        if iso_lang not in self._normalizer_cache:
            self._normalizer_cache[iso_lang] = self._indic_norm_factory.get_normalizer(
                iso_lang
            )
        return self._normalizer_cache[iso_lang]

    def _preprocess_batch(self, sentences: List[str], lang: str) -> List[str]:
        """
        Preprocess sentences with batch optimization.

        Optimized version with:
        - Single pass processing
        - Cached function references
        - Reduced function call overhead
        """
        if not sentences:
            return []

        iso_lang = self._flores_codes.get(lang, "hi")
        normalizer = self._get_normalizer(iso_lang)

        # Cache function references locally for faster access
        normalize_fn = normalizer.normalize
        tokenize_fn = self._trivial_tokenize

        # Process all sentences in a single comprehension
        return [
            " ".join(tokenize_fn(normalize_fn(line.strip()), iso_lang))
            for line in sentences
            if line.strip()  # Skip empty lines
        ]

    def _read_file(self, filepath: str) -> List[str]:
        """
        Read file efficiently with proper error handling.
        """
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            # Direct list comprehension is more efficient than readlines()
            with path.open("r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        except UnicodeDecodeError as e:
            raise ValueError(f"Unable to decode file {filepath}: {e}")
        except Exception as e:
            raise IOError(f"Error reading file {filepath}: {e}")

    def _compute_scores(
        self, preds: List[str], refs: List[str], use_13a: bool = False
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Compute BLEU and chrF2++ scores efficiently.
        """
        bleu_metric = self._bleu_metric_13a if use_13a else self._bleu_metric_none

        # Compute scores in parallel-friendly way
        bleu_score = bleu_metric.corpus_score(preds, [refs])
        chrf_score = self._chrf2_metric.corpus_score(preds, [refs])

        return {
            "bleu": {
                "score": round(bleu_score.score, 1),
                "signature": bleu_metric.get_signature().format(),
            },
            "chrF2++": {
                "score": round(chrf_score.score, 1),
                "signature": self._chrf2_metric.get_signature().format(),
            },
        }

    def evaluate(
        self,
        tgt_lang: str,
        preds: Union[List[str], str],
        refs: Union[List[str], str],
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Evaluate BLEU and chrF2++ scores for predictions and references.

        Optimized version with:
        - Better error handling
        - Reduced code duplication
        - More efficient file reading
        - Streamlined processing flow
        """
        if preds is None or refs is None:
            raise ValueError("Predictions and References cannot be None")

        # Convert file paths to lists if needed
        if isinstance(preds, str):
            preds = self._read_file(preds)
        if isinstance(refs, str):
            refs = self._read_file(refs)

        if len(preds) != len(refs):
            raise ValueError(
                f"Number of predictions ({len(preds)}) and references ({len(refs)}) do not match"
            )

        if not preds:
            raise ValueError("No data to evaluate")

        # Process based on language
        if tgt_lang == "eng_Latn":
            # For English, use 13a tokenization directly
            return self._compute_scores(preds, refs, use_13a=True)
        else:
            # For non-English languages, preprocess first
            preds_processed = self._preprocess_batch(preds, tgt_lang)
            refs_processed = self._preprocess_batch(refs, tgt_lang)

            if len(preds_processed) != len(refs_processed):
                raise ValueError("Preprocessing resulted in mismatched lengths")

            return self._compute_scores(preds_processed, refs_processed, use_13a=False)

    def evaluate_streaming(
        self,
        tgt_lang: str,
        preds_file: str,
        refs_file: str,
        batch_size: int = 1000,
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Evaluate large files using streaming/batching for memory efficiency.
        """

        def read_batches(
            filepath: str, batch_size: int
        ) -> Generator[List[str], None, None]:
            """Generate batches of lines from file."""
            batch = []
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            batch.append(line)
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                    if batch:  # Yield remaining lines
                        yield batch
            except Exception as e:
                raise IOError(f"Error reading file {filepath}: {e}")

        # Process files in batches
        all_preds_processed = []
        all_refs_processed = []

        preds_gen = read_batches(preds_file, batch_size)
        refs_gen = read_batches(refs_file, batch_size)

        try:
            for preds_batch, refs_batch in zip(preds_gen, refs_gen):
                if len(preds_batch) != len(refs_batch):
                    raise ValueError(
                        f"Batch size mismatch: {len(preds_batch)} vs {len(refs_batch)}"
                    )

                if tgt_lang != "eng_Latn":
                    preds_batch = self._preprocess_batch(preds_batch, tgt_lang)
                    refs_batch = self._preprocess_batch(refs_batch, tgt_lang)

                all_preds_processed.extend(preds_batch)
                all_refs_processed.extend(refs_batch)

        except StopIteration:
            pass  # Normal termination

        if not all_preds_processed:
            raise ValueError("No data processed")

        return self._compute_scores(
            all_preds_processed, all_refs_processed, use_13a=(tgt_lang == "eng_Latn")
        )


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

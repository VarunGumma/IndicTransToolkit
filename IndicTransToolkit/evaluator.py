from typing import List, Union
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

    def _get_normalizer(self, iso_lang: str):
        """
        Return a cached normalizer for a given iso_lang.
        """
        if iso_lang not in self._normalizer_cache:
            self._normalizer_cache[iso_lang] = self._indic_norm_factory.get_normalizer(iso_lang)
        return self._normalizer_cache[iso_lang]

    def _preprocess(self, sentences: List[str], lang: str) -> List[str]:
        """
        Preprocess the sentences using IndicNLP: 
        1) Normalization (using a cached normalizer), 
        2) Trivial tokenization.
        """
        iso_lang = self._flores_codes.get(lang, "hi")
        # Fetch from cache to avoid reconstructing the normalizer
        normalizer = self._get_normalizer(iso_lang)

        # Local references for speed
        trivial_tokenize = indic_tokenize.trivial_tokenize
        normalize_fn = normalizer.normalize

        processed_sentences = []
        for line in sentences:
            # single .strip() before normalizing
            line = line.strip()
            norm_line = normalize_fn(line)
            tokens = trivial_tokenize(norm_line, iso_lang)
            processed_sentences.append(" ".join(tokens))

        return processed_sentences

    def evaluate(
        self,
        tgt_lang: str,
        preds: Union[List[str], str],
        refs: Union[List[str], str],
    ):
        """
        Evaluate BLEU and chrF2++ scores for the given predictions and references.
        - If preds/refs are strings (filenames), read them from disk.
        - If they are lists, evaluate them directly.
        - For non-English languages, applies Indic NLP preprocessing before scoring.
        """
        assert preds is not None and refs is not None, "Predictions and References cannot be None"

        # Convert file paths to lists if needed
        if isinstance(preds, str):
            with open(preds, "r", encoding="utf-8") as fp:
                preds = [line.strip() for line in fp]
        if isinstance(refs, str):
            with open(refs, "r", encoding="utf-8") as fr:
                refs = [line.strip() for line in fr]

        assert len(preds) == len(refs), "Number of predictions and references do not match"

        # Local references to metrics for speed
        bleu_none = self._bleu_metric_none
        bleu_13a = self._bleu_metric_13a
        chrf2 = self._chrf2_metric

        scores = {}

        # For English (eng_Latn), skip Indic NLP normalization
        if tgt_lang != "eng_Latn":
            preds_ = self._preprocess(preds, tgt_lang)
            refs_ = self._preprocess(refs, tgt_lang)

            bleu_score = bleu_none.corpus_score(preds_, [refs_])
            chrf_score = chrf2.corpus_score(preds_, [refs_])

            scores["bleu"] = {
                "score": round(bleu_score.score, 1),
                "signature": bleu_none.get_signature().format(),
            }
            scores["chrF2++"] = {
                "score": round(chrf_score.score, 1),
                "signature": chrf2.get_signature().format(),
            }

        else:
            # For English, 13a tokenization is standard
            bleu_score = bleu_13a.corpus_score(preds, [refs])
            chrf_score = chrf2.corpus_score(preds, [refs])

            scores["bleu"] = {
                "score": round(bleu_score.score, 1),
                "signature": bleu_13a.get_signature().format(),
            }
            scores["chrF2++"] = {
                "score": round(chrf_score.score, 1),
                "signature": chrf2.get_signature().format(),
            }

        return scores

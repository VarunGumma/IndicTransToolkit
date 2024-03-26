from typing import List, Union
from sacrebleu.metrics import CHRF, BLEU

from joblib import Parallel, delayed
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

_chrf2_metric = CHRF(word_order=2)
_bleu_metric_13a = BLEU(tokenize="13a")
_bleu_metric_none = BLEU(tokenize="none")
_indic_norm_factory = IndicNormalizerFactory()

_flores_codes = {
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

_indic_preprocess_line = lambda line, normalizer, iso_lang: " ".join(
    indic_tokenize.trivial_tokenize(normalizer.normalize(line.strip()), iso_lang)
)


def indic_preprocess(sentences: List[str], lang: str) -> List[str]:
    iso_lang = _flores_codes.get(lang, "hi")
    normalizer = _indic_norm_factory.get_normalizer(iso_lang)

    processed_sentences = Parallel(n_jobs=-1)(
        delayed(_indic_preprocess_line)(line, normalizer, iso_lang)
        for line in sentences
    )
    return processed_sentences


def indic_evaluate(
    tgt_lang: str,
    preds: Union[List[str], str],
    refs: Union[List[str], str],
):
    """
    Evaluate BLEU and chrF2++ scores for the given predictions and references
    :param tgt_lang: Target language
    :param preds: List of predictions or path to predictions file
    :param refs: List of references or path to references file
    :param outfname: Path to store the scores
    :return: Dictionary containing BLEU and chrF2++ scores
    """

    assert (
        preds is not None or refs is not None
    ), "Predictions and References cannot be None"

    if isinstance(preds, str):
        with open(preds, "r", encoding="utf-8") as fp:
            preds = [line.strip() for line in fp]

    if isinstance(refs, str):
        with open(refs, "r", encoding="utf-8") as fr:
            refs = [line.strip() for line in fr]

    assert len(preds) == len(refs), "Number of predictions and references do not match"

    score = {}

    if tgt_lang != "eng_Latn":
        preds_ = indic_preprocess(preds, tgt_lang)
        refs_ = indic_preprocess(refs, tgt_lang)

        score["bleu"] = {
            "score": round(_bleu_metric_none.corpus_score(preds_, [refs_]).score, 1),
            "signature": _bleu_metric_none.get_signature().format(),
        }
        score["chrF2++"] = {
            "score": round(_chrf2_metric.corpus_score(preds_, [refs_]).score, 1),
            "signature": _chrf2_metric.get_signature().format(),
        }
    else:
        score["bleu"] = {
            "score": round(_bleu_metric_13a.corpus_score(preds, [refs]).score, 1),
            "signature": _bleu_metric_13a.get_signature().format(),
        }
        score["chrF2++"] = {
            "score": round(_chrf2_metric.corpus_score(preds, [refs]).score, 1),
            "signature": _chrf2_metric.get_signature().format(),
        }

    return score

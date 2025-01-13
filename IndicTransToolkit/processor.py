import regex as re
from tqdm import tqdm
from queue import Queue
from typing import List, Union

from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

# Optional parallelization
from concurrent.futures import ThreadPoolExecutor


class IndicProcessor:
    # =====================
    # REGEX PRECOMPILATION
    # =====================
    _MULTISPACE_REGEX = re.compile(r"[ ]{2,}")
    _DIGIT_SPACE_PERCENT = re.compile(r"(\d) %")
    _DOUBLE_QUOT_PUNC = re.compile(r"\"([,\.]+)")
    _DIGIT_NBSP_DIGIT = re.compile(r"(\d) (\d)")
    _END_BRACKET_SPACE_PUNC_REGEX = re.compile(r"\) ([\.!:?;,])")

    _URL_PATTERN = re.compile(
        r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b"
    )
    _NUMERAL_PATTERN = re.compile(
        r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
    )
    _EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
    _OTHER_PATTERN = re.compile(r"[A-Za-z0-9]*[#|@]\w+")

    # Consolidate many small punctuation replacements into regex pairs for single-pass sub
    _PUNC_REPLACEMENTS = [
        (re.compile(r"\r"), ""),  # remove carriage returns
        (re.compile(r"\(\s*"), "("),  # fix bracket spacing (opening)
        (re.compile(r"\s*\)"), ")"),  # fix bracket spacing (closing)
        (re.compile(r"\s:\s?"), ":"),
        (re.compile(r"\s;\s?"), ";"),
        (re.compile(r"[`´‘‚’]"), "'"),
        (re.compile(r"[„“”«»]"), '"'),
        (re.compile(r"[–—]"), "-"),
        (re.compile(r"\.\.\."), "..."),
        (re.compile(r" %"), "%"),  # remove non-breaking space before percent
        (re.compile(r"nº "), "nº "),
        (re.compile(r" ºC"), " ºC"),
        (re.compile(r" [?!;]"), lambda m: m.group(0).strip()),
        (re.compile(r", "), ", "),
    ]

    # "ID" translations for placeholders
    _INDIC_FAILURE_CASES = [
        "آی ڈی ",
        "ꯑꯥꯏꯗꯤ",
        "आईडी",
        "आई . डी . ",
        "आई . डी .",
        "आई. डी. ",
        "आई. डी.",
        "आय. डी. ",
        "आय. डी.",
        "आय . डी . ",
        "आय . डी .",
        "ऐटि",
        "آئی ڈی ",
        "ᱟᱭᱰᱤ ᱾",
        "आयडी",
        "ऐडि",
        "आइडि",
        "ᱟᱭᱰᱤ",
    ]

    def __init__(self, inference=True):
        self.inference = inference

        # ==============================
        # FLORES -> ISO LANGUAGE CODES
        # ==============================
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

        # ==============================
        # CREATING DIGITS TRANSLATION TABLE
        # (combines your _indic_num_map into str.translate)
        # ==============================
        indic_digits_map = {}
        digits_dict = {
            "\u09e6": "0",
            "\u0ae6": "0",
            "\u0ce6": "0",
            "\u0966": "0",
            "\u0660": "0",
            "\uabf0": "0",
            "\u0b66": "0",
            "\u0a66": "0",
            "\u1c50": "0",
            "\u06f0": "0",
            "\u09e7": "1",
            "\u0ae7": "1",
            "\u0967": "1",
            "\u0ce7": "1",
            "\u06f1": "1",
            "\uabf1": "1",
            "\u0b67": "1",
            "\u0a67": "1",
            "\u1c51": "1",
            "\u0c67": "1",
            "\u09e8": "2",
            "\u0ae8": "2",
            "\u0968": "2",
            "\u0ce8": "2",
            "\u06f2": "2",
            "\uabf2": "2",
            "\u0b68": "2",
            "\u0a68": "2",
            "\u1c52": "2",
            "\u0c68": "2",
            "\u09e9": "3",
            "\u0ae9": "3",
            "\u0969": "3",
            "\u0ce9": "3",
            "\u06f3": "3",
            "\uabf3": "3",
            "\u0b69": "3",
            "\u0a69": "3",
            "\u1c53": "3",
            "\u0c69": "3",
            "\u09ea": "4",
            "\u0aea": "4",
            "\u096a": "4",
            "\u0cea": "4",
            "\u06f4": "4",
            "\uabf4": "4",
            "\u0b6a": "4",
            "\u0a6a": "4",
            "\u1c54": "4",
            "\u0c6a": "4",
            "\u09eb": "5",
            "\u0aeb": "5",
            "\u096b": "5",
            "\u0ceb": "5",
            "\u06f5": "5",
            "\uabf5": "5",
            "\u0b6b": "5",
            "\u0a6b": "5",
            "\u1c55": "5",
            "\u0c6b": "5",
            "\u09ec": "6",
            "\u0aec": "6",
            "\u096c": "6",
            "\u0cec": "6",
            "\u06f6": "6",
            "\uabf6": "6",
            "\u0b6c": "6",
            "\u0a6c": "6",
            "\u1c56": "6",
            "\u0c6c": "6",
            "\u09ed": "7",
            "\u0aed": "7",
            "\u096d": "7",
            "\u0ced": "7",
            "\u06f7": "7",
            "\uabf7": "7",
            "\u0b6d": "7",
            "\u0a6d": "7",
            "\u1c57": "7",
            "\u0c6d": "7",
            "\u09ee": "8",
            "\u0aee": "8",
            "\u096e": "8",
            "\u0cee": "8",
            "\u06f8": "8",
            "\uabf8": "8",
            "\u0b6e": "8",
            "\u0a6e": "8",
            "\u1c58": "8",
            "\u0c6e": "8",
            "\u09ef": "9",
            "\u0aef": "9",
            "\u096f": "9",
            "\u0cef": "9",
            "\u06f9": "9",
            "\uabf9": "9",
            "\u0b6f": "9",
            "\u0a6f": "9",
            "\u1c59": "9",
            "\u0c6f": "9",
        }
        for k, v in digits_dict.items():
            indic_digits_map[ord(k)] = v
        # Also map ASCII '0'-'9' to themselves (optional, ensures no break)
        for c in range(ord("0"), ord("9") + 1):
            indic_digits_map[c] = chr(c)

        self._digits_translation_table = indic_digits_map

        # ==============================
        # PLACEHOLDER QUEUE
        # ==============================
        self._placeholder_entity_maps = Queue()

        # ==============================
        # MOSES FOR ENGLISH
        # ==============================
        self._en_tok = MosesTokenizer(lang="en")
        self._en_normalizer = MosesPunctNormalizer()
        self._en_detok = MosesDetokenizer(lang="en")

        # ==============================
        # TRANSLITERATOR
        # ==============================
        self._xliterator = UnicodeIndicTransliterator()

        # ==============================
        # CACHE FOR NORMALIZERS
        # ==============================
        self._normalizer_cache = {}

    # =======================================================
    # CACHED NORMALIZER ACCESS
    # =======================================================
    def _get_normalizer(self, iso_code: str):
        """
        Return a cached normalizer for the given iso_code.
        """
        if iso_code not in self._normalizer_cache:
            self._normalizer_cache[iso_code] = IndicNormalizerFactory().get_normalizer(
                iso_code
            )
        return self._normalizer_cache[iso_code]

    # =======================================================
    # BATCHES
    # =======================================================
    def get_batches(self, sentences: List[str], batch_size=8):
        """
        Generate batches from a list of sentences.
        """
        for i in range(0, len(sentences), batch_size):
            yield sentences[i : i + batch_size]

    # =======================================================
    # PUNCTUATION NORMALIZATION
    # =======================================================
    def _punc_norm(self, text: str) -> str:
        """
        Apply precompiled regex-based punctuation replacements in fewer passes.
        """
        # local ref to reduce repeated self lookups
        replacements = self._PUNC_REPLACEMENTS

        for pattern, replacement in replacements:
            text = pattern.sub(replacement, text)

        # final special-case regex substitutions
        text = self._MULTISPACE_REGEX.sub(" ", text)
        text = self._END_BRACKET_SPACE_PUNC_REGEX.sub(r")\1", text)
        text = self._DIGIT_SPACE_PERCENT.sub(r"\1%", text)
        text = self._DOUBLE_QUOT_PUNC.sub(r'\1"', text)
        text = self._DIGIT_NBSP_DIGIT.sub(r"\1.\2", text)
        return text.strip()

    # =======================================================
    # WRAP PLACEHOLDERS
    # =======================================================
    def _wrap_with_placeholders(self, text: str) -> str:
        """
        Wrap substrings with matched patterns in the text with placeholders.
        The placeholder map is enqueued in _placeholder_entity_maps.
        """
        serial_no = 1
        placeholder_entity_map = {}

        # local references
        url_pattern = self._URL_PATTERN
        numeral_pattern = self._NUMERAL_PATTERN
        email_pattern = self._EMAIL_PATTERN
        other_pattern = self._OTHER_PATTERN
        indic_failure_cases = self._INDIC_FAILURE_CASES

        # order of searching
        patterns = [email_pattern, url_pattern, numeral_pattern, other_pattern]

        for pattern in patterns:
            matches = set(pattern.findall(text))

            for match in matches:
                # Additional checks for short placeholders
                if pattern is url_pattern:
                    if len(match.replace(".", "")) < 4:
                        continue
                if pattern is numeral_pattern:
                    if (
                        len(match.replace(" ", "").replace(".", "").replace(":", ""))
                        < 4
                    ):
                        continue

                base_placeholder = f"<ID{serial_no}>"

                # Populate placeholder variants
                placeholder_entity_map[f"<ID{serial_no}]"] = match
                placeholder_entity_map[f"< ID{serial_no} ]"] = match
                placeholder_entity_map[f"<ID{serial_no}>"] = match
                placeholder_entity_map[f"< ID{serial_no} >"] = match
                placeholder_entity_map[f"[ID{serial_no}]"] = match
                placeholder_entity_map[f"[ID {serial_no}]"] = match
                placeholder_entity_map[f"[ ID{serial_no} ]"] = match

                for i in indic_failure_cases:
                    placeholder_entity_map[f"<{i}{serial_no}>"] = match
                    placeholder_entity_map[f"< {i}{serial_no} >"] = match
                    placeholder_entity_map[f"< {i} {serial_no} >"] = match
                    placeholder_entity_map[f"<{i} {serial_no}]"] = match
                    placeholder_entity_map[f"< {i} {serial_no} ]"] = match
                    placeholder_entity_map[f"[{i}{serial_no}]"] = match
                    placeholder_entity_map[f"[{i} {serial_no}]"] = match
                    placeholder_entity_map[f"[ {i}{serial_no} ]"] = match
                    placeholder_entity_map[f"[ {i} {serial_no} ]"] = match
                    placeholder_entity_map[f"{i} {serial_no}"] = match
                    placeholder_entity_map[f"{i}{serial_no}"] = match

                # Replace match in text
                text = text.replace(match, base_placeholder)
                serial_no += 1

        text = re.sub(r"\s+", " ", text).replace(">/", ">").replace("]/", "]")
        self._placeholder_entity_maps.put(placeholder_entity_map)
        return text

    # =======================================================
    # NORMALIZE TEXT
    # =======================================================
    def _normalize(self, text: str) -> str:
        """
        Normalizes numerals in one pass, optionally wrapping placeholders.
        """
        # single-pass digit translation
        text = text.translate(self._digits_translation_table)

        if self.inference:
            text = self._wrap_with_placeholders(text)

        return text

    # =======================================================
    # PREPROCESS (SINGLE SENTENCE)
    # =======================================================
    def _preprocess(
        self,
        sent: str,
        src_lang: str,
        tgt_lang: str,
        normalizer: Union[MosesPunctNormalizer, IndicNormalizerFactory],
        is_target: bool = False,
    ) -> str:
        """
        Preprocess a single sentence: punctuation norm, numeral norm, tokenization, optional transliteration,
        and optional insertion of language tags (if not target).
        """
        # local references
        punc_norm = self._punc_norm
        _normalize = self._normalize
        en_tok = self._en_tok
        en_norm = self._en_normalizer
        flores_codes = self._flores_codes
        xlit = self._xliterator

        iso_lang = flores_codes.get(src_lang, "hi")

        # 1) Punctuation normalization
        sent = punc_norm(sent)
        # 2) Numeral & placeholders
        sent = _normalize(sent)

        # Decide if we need to transliterate or not
        script_part = src_lang.split("_")[1]
        transliterate = script_part not in ["Arab", "Aran", "Olck", "Mtei", "Latn"]

        # 3) Tokenize (English vs. Indic)
        if iso_lang == "en":
            sent = en_norm.normalize(sent.strip())
            processed_sent = " ".join(en_tok.tokenize(sent, escape=False))
        else:
            # Normalize + tokenize for Indic
            tokenized_sent = " ".join(
                indic_tokenize.trivial_tokenize(
                    normalizer.normalize(sent.strip()), iso_lang
                )
            )
            if transliterate:
                # Transliterate from iso_lang -> "hi"
                processed_sent = xlit.transliterate(
                    tokenized_sent, iso_lang, "hi"
                ).replace(" ् ", "्")
            else:
                processed_sent = tokenized_sent

        processed_sent = processed_sent.strip()
        # If not the target, add src/tgt language tags
        return (
            f"{src_lang} {tgt_lang} {processed_sent}"
            if not is_target
            else processed_sent
        )

    # =======================================================
    # PREPROCESS BATCH
    # =======================================================
    def preprocess_batch(
        self,
        batch: List[str],
        src_lang: str,
        tgt_lang: str = None,
        is_target: bool = False,
        visualize: bool = False,
    ) -> List[str]:
        """
        Preprocess an array of sentences (normalize, tokenize, transliterate).
        Optionally parallelize if large batches exist.
        """
        iso_code = self._flores_codes.get(src_lang, "hi")
        normalizer = None
        if src_lang != "eng_Latn":
            normalizer = self._get_normalizer(iso_code)

        if visualize:
            iterator = tqdm(
                batch,
                unit="line",
                total=len(batch),
                desc=f" | > Pre-processing {src_lang}",
            )
        else:
            iterator = batch

        return [
            self._preprocess(sent, src_lang, tgt_lang, normalizer, is_target)
            for sent in iterator
        ]

    # =======================================================
    # POSTPROCESS (SINGLE SENTENCE)
    # =======================================================
    def _postprocess(self, sent: str, lang: str = "hin_Deva") -> str:
        """
        Postprocess a single sentence:
          - get the correct placeholder map from the queue
          - fix scripts for Perso-Arabic
          - restore placeholders
          - detokenize (English or Indic with transliteration if needed)
        """
        # local references
        placeholder_entity_map = self._placeholder_entity_maps.get()
        xlit = self._xliterator
        en_detok = self._en_detok

        if isinstance(sent, (tuple, list)):
            # If it was passed as a tuple from e.g. zip, take the first item
            sent = sent[0]

        lang_code, script_code = lang.split("_")
        iso_lang = self._flores_codes.get(lang, "hi")

        # Fixes for Perso-Arabic scripts
        if script_code in ["Arab", "Aran"]:
            sent = (
                sent.replace(" ؟", "؟")
                .replace(" ۔", "۔")
                .replace(" ،", "،")
                .replace("ٮ۪", "ؠ")
            )
        # Oriya fix
        if lang_code == "ory":
            sent = sent.replace("ଯ଼", "ୟ")

        # Restore placeholders
        for k, v in placeholder_entity_map.items():
            sent = sent.replace(k, v)

        # Detokenize for output
        if lang == "eng_Latn":
            return en_detok.detokenize(sent.split(" "))
        else:
            # Transliterate from 'hi' to iso_lang if needed
            return indic_detokenize.trivial_detokenize(
                xlit.transliterate(sent, "hi", iso_lang), iso_lang
            )

    # =======================================================
    # POSTPROCESS BATCH
    # =======================================================
    def postprocess_batch(
        self, sents: List[str], lang: str = "hin_Deva", visualize: bool = False
    ) -> List[str]:
        """
        Postprocess a batch of sentences: restore placeholders, fix scripts,
        optionally parallelize.
        """
        if visualize:
            iterator = tqdm(
                sents,
                unit="line",
                total=len(sents),
                desc=f" | > Post-processing {lang}",
            )
        else:
            iterator = sents

        results = [self._postprocess(sent, lang) for sent in iterator]

        # Clear the placeholder queue so it's fresh for next usage
        self._placeholder_entity_maps.queue.clear()
        return results

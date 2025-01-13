#cython: language_level=3, boundscheck=False
"""
Cython version of the IndicProcessor class with fixes for compilation errors.
Functionality is nearly identical to the original Python code, subject to Cython constraints.
"""

import regex as re
from tqdm import tqdm
from queue import Queue
from typing import List, Dict, Union
# We can't cimport these libraries directly because they don't offer C-extensions.
# We'll just import them as Python objects.
from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator


#########
# NOTE:
# 1) We removed cdef type annotations for Queue, MosesTokenizer, etc. 
#    because they are pure Python objects with no C-level definitions.
# 2) We replaced the generator method get_batches with a normal Python method 
#    or a staticmethod returning a generator. 
# 3) We removed cdef (object, object) pair usage.
#########


cdef class IndicProcessor:
    cdef public bint inference

    # Precompiled regex patterns and placeholders
    cdef object _MULTISPACE_REGEX
    cdef object _DIGIT_SPACE_PERCENT
    cdef object _DOUBLE_QUOT_PUNC
    cdef object _DIGIT_NBSP_DIGIT
    cdef object _END_BRACKET_SPACE_PUNC_REGEX

    cdef object _URL_PATTERN
    cdef object _NUMERAL_PATTERN
    cdef object _EMAIL_PATTERN
    cdef object _OTHER_PATTERN

    cdef list _PUNC_REPLACEMENTS
    cdef list _INDIC_FAILURE_CASES

    cdef dict _flores_codes
    cdef dict _digits_translation_table

    # We store placeholder maps in a Python Queue (treated as `object` for Cython)
    cdef object _placeholder_entity_maps

    # Tools (also Python objects)
    cdef object _en_tok
    cdef object _en_normalizer
    cdef object _en_detok
    cdef object _xliterator

    def __cinit__(self, bint inference=True):
        """
        Constructor for FastIndicProcessor. Called once at object allocation.
        """
        self.inference = inference

        ##############################
        # FLORES -> ISO CODES
        ##############################
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

        ##############################
        # INDIC DIGIT TRANSLATION (str.translate)
        ##############################
        self._digits_translation_table = {}
        digits_dict = {
            "\u09e6": "0", "\u0ae6": "0", "\u0ce6": "0", "\u0966": "0",
            "\u0660": "0", "\uabf0": "0", "\u0b66": "0", "\u0a66": "0",
            "\u1c50": "0", "\u06f0": "0",

            "\u09e7": "1", "\u0ae7": "1", "\u0967": "1", "\u0ce7": "1",
            "\u06f1": "1", "\uabf1": "1", "\u0b67": "1", "\u0a67": "1",
            "\u1c51": "1", "\u0c67": "1",

            "\u09e8": "2", "\u0ae8": "2", "\u0968": "2", "\u0ce8": "2",
            "\u06f2": "2", "\uabf2": "2", "\u0b68": "2", "\u0a68": "2",
            "\u1c52": "2", "\u0c68": "2",

            "\u09e9": "3", "\u0ae9": "3", "\u0969": "3", "\u0ce9": "3",
            "\u06f3": "3", "\uabf3": "3", "\u0b69": "3", "\u0a69": "3",
            "\u1c53": "3", "\u0c69": "3",

            "\u09ea": "4", "\u0aea": "4", "\u096a": "4", "\u0cea": "4",
            "\u06f4": "4", "\uabf4": "4", "\u0b6a": "4", "\u0a6a": "4",
            "\u1c54": "4", "\u0c6a": "4",

            "\u09eb": "5", "\u0aeb": "5", "\u096b": "5", "\u0ceb": "5",
            "\u06f5": "5", "\uabf5": "5", "\u0b6b": "5", "\u0a6b": "5",
            "\u1c55": "5", "\u0c6b": "5",

            "\u09ec": "6", "\u0aec": "6", "\u096c": "6", "\u0cec": "6",
            "\u06f6": "6", "\uabf6": "6", "\u0b6c": "6", "\u0a6c": "6",
            "\u1c56": "6", "\u0c6c": "6",

            "\u09ed": "7", "\u0aed": "7", "\u096d": "7", "\u0ced": "7",
            "\u06f7": "7", "\uabf7": "7", "\u0b6d": "7", "\u0a6d": "7",
            "\u1c57": "7", "\u0c6d": "7",

            "\u09ee": "8", "\u0aee": "8", "\u096e": "8", "\u0cee": "8",
            "\u06f8": "8", "\uabf8": "8", "\u0b6e": "8", "\u0a6e": "8",
            "\u1c58": "8", "\u0c6e": "8",

            "\u09ef": "9", "\u0aef": "9", "\u096f": "9", "\u0cef": "9",
            "\u06f9": "9", "\uabf9": "9", "\u0b6f": "9", "\u0a6f": "9",
            "\u1c59": "9", "\u0c6f": "9",
        }
        for k, v in digits_dict.items():
            self._digits_translation_table[ord(k)] = v

        # Also map ASCII '0'-'9'
        for c in range(ord('0'), ord('9') + 1):
            self._digits_translation_table[c] = chr(c)

        ##############################
        # PLACEHOLDER MAP QUEUE
        ##############################
        self._placeholder_entity_maps = Queue()

        ##############################
        # MOSES (as Python objects)
        ##############################
        self._en_tok = MosesTokenizer(lang="en")
        self._en_normalizer = MosesPunctNormalizer()
        self._en_detok = MosesDetokenizer(lang="en")

        ##############################
        # TRANSLITERATOR (Python object)
        ##############################
        self._xliterator = UnicodeIndicTransliterator()

        ##############################
        # Precompiled Patterns
        ##############################
        self._MULTISPACE_REGEX = re.compile(r"[ ]{2,}")
        self._DIGIT_SPACE_PERCENT = re.compile(r"(\d) %")
        self._DOUBLE_QUOT_PUNC = re.compile(r"\"([,\.]+)")
        self._DIGIT_NBSP_DIGIT = re.compile(r"(\d) (\d)")
        self._END_BRACKET_SPACE_PUNC_REGEX = re.compile(r"\) ([\.!:?;,])")

        self._URL_PATTERN = re.compile(
            r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b"
        )
        self._NUMERAL_PATTERN = re.compile(
            r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
        )
        self._EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
        self._OTHER_PATTERN = re.compile(r"[A-Za-z0-9]*[#|@]\w+")

        # Combined punctuation replacements
        self._PUNC_REPLACEMENTS = [
            (re.compile(r"\r"), ""),
            (re.compile(r"\(\s*"), "("),
            (re.compile(r"\s*\)"), ")"),
            (re.compile(r"\s:\s?"), ":"),
            (re.compile(r"\s;\s?"), ";"),
            (re.compile(r"[`´‘‚’]"), "'"),
            (re.compile(r"[„“”«»]"), '"'),
            (re.compile(r"[–—]"), "-"),
            (re.compile(r"\.\.\."), "..."),
            (re.compile(r" %"), "%"),
            (re.compile(r"nº "), "nº "),
            (re.compile(r" ºC"), " ºC"),
            (re.compile(r" [?!;]"), lambda m: m.group(0).strip()),
            (re.compile(r", "), ", "),
        ]

        self._INDIC_FAILURE_CASES = [
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

    # Instead of a cdef method using 'yield', we can use a normal Python method:
    def get_batches(self, sentences: List[str], batch_size: int = 8):
        """
        Generator for batches from a large list of sentences.
        (Cython doesn't support yield in cdef/cpdef methods)
        """
        for i in range(0, len(sentences), batch_size):
            yield sentences[i : i + batch_size]

    @staticmethod
    def _apply_punc_replacements(text: str, replacements: list) -> str:
        """
        Apply a list of (pattern, replacement) in sequence to text.
        """
        for pair in replacements:
            text = pair[0].sub(pair[1], text)
        return text

    cdef str _punc_norm(self, str text) except *:
        """
        Consolidate punctuation normalization in fewer passes.
        """
        # 1) Apply replacements (call static method as a normal function)
        text = self._apply_punc_replacements(text, self._PUNC_REPLACEMENTS)

        # 2) Additional patterns
        text = self._MULTISPACE_REGEX.sub(" ", text)
        text = self._END_BRACKET_SPACE_PUNC_REGEX.sub(r")\1", text)
        text = self._DIGIT_SPACE_PERCENT.sub(r"\1%", text)
        text = self._DOUBLE_QUOT_PUNC.sub(r'\1"', text)
        text = self._DIGIT_NBSP_DIGIT.sub(r"\1.\2", text)
        return text.strip()

    cdef str _wrap_with_placeholders(self, str text) except *:
        """
        Wrap substrings with matched patterns in the text with placeholders.
        We store the placeholder map in the queue for retrieval in postprocessing.
        """
        cdef int serial_no = 1
        cdef dict placeholder_entity_map = {}
        cdef list patterns = [
            self._EMAIL_PATTERN,
            self._URL_PATTERN,
            self._NUMERAL_PATTERN,
            self._OTHER_PATTERN,
        ]
        cdef object pattern
        cdef set matches
        cdef str match, base_placeholder, i

        for pattern in patterns:
            matches = set(pattern.findall(text))
            for match in matches:
                # Additional checks
                if pattern is self._URL_PATTERN:
                    if len(match.replace(".", "")) < 4:
                        continue
                if pattern is self._NUMERAL_PATTERN:
                    if len(match.replace(" ", "").replace(".", "").replace(":", "")) < 4:
                        continue

                base_placeholder = f"<ID{serial_no}>"
                placeholder_entity_map[f"<ID{serial_no}]"] = match
                placeholder_entity_map[f"< ID{serial_no} ]"] = match
                placeholder_entity_map[f"<ID{serial_no}>"] = match
                placeholder_entity_map[f"< ID{serial_no} >"] = match
                placeholder_entity_map[f"[ID{serial_no}]"] = match
                placeholder_entity_map[f"[ID {serial_no}]"] = match
                placeholder_entity_map[f"[ ID{serial_no} ]"] = match

                for i in self._INDIC_FAILURE_CASES:
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

                text = text.replace(match, base_placeholder)
                serial_no += 1

        text = re.sub(r"\s+", " ", text).replace(">/", ">").replace("]/", "]")
        self._placeholder_entity_maps.put(placeholder_entity_map)
        return text

    cdef str _normalize(self, str text) except *:
        """
        Normalizes numerals and optionally wraps placeholders.
        """
        # single-pass digit translation
        text = text.translate(self._digits_translation_table)

        if self.inference:
            text = self._wrap_with_placeholders(text)
        return text

    cdef str _do_indic_tokenize_and_transliterate(
        self,
        str sentence,
        object normalizer,
        str iso_lang,
        bint transliterate
    ) except *:
        """
        Helper method: normalizes, tokenizes, optionally transliterates from iso_lang -> 'hi'.
        """
        cdef str normed
        cdef list tokens
        cdef str joined, xlated

        normed = normalizer.normalize(sentence.strip())
        tokens = indic_tokenize.trivial_tokenize(normed, iso_lang)
        joined = " ".join(tokens)
        xlated = joined
        if transliterate:
            xlated = self._xliterator.transliterate(joined, iso_lang, "hi")
            xlated = xlated.replace(" ् ", "्")
        return xlated

    cdef str _preprocess(
        self,
        str sent,
        str src_lang,
        str tgt_lang,
        object normalizer,
        bint is_target
    ) except *:
        """
        Preprocess a single sentence: punctuation norm, numeral norm, tokenization, transliteration, language tags.
        """
        cdef str iso_lang = self._flores_codes.get(src_lang, "hi")
        cdef str script_part = src_lang.split("_")[1]
        cdef bint do_transliterate = True
        cdef str e_strip, e_norm
        cdef object e_tokens
        cdef str processed_sent

        # 1) punctuation normalization
        sent = self._punc_norm(sent)
        # 2) numerals & placeholders
        sent = self._normalize(sent)

        if script_part in ["Arab", "Aran", "Olck", "Mtei", "Latn"]:
            do_transliterate = False

        if iso_lang == "en":
            # English path
            e_strip = sent.strip()
            e_norm = self._en_normalizer.normalize(e_strip)
            e_tokens = self._en_tok.tokenize(e_norm, escape=False)
            processed_sent = " ".join(e_tokens)
        else:
            # Indic path
            processed_sent = self._do_indic_tokenize_and_transliterate(sent, normalizer, iso_lang, do_transliterate)

        processed_sent = processed_sent.strip()
        if not is_target:
            return f"{src_lang} {tgt_lang} {processed_sent}"
        else:
            return processed_sent

    cpdef list preprocess_batch(
        self,
        List[str] batch,
        str src_lang,
        str tgt_lang=None,
        bint is_target=False,
        bint visualize=False
    ):
        """
        Preprocess an array of sentences (normalize, tokenize, transliterate).
        We do this serially here (no yield, no concurrency).
        """
        cdef object normalizer = None
        cdef str iso_code = self._flores_codes.get(src_lang, "hi")
        cdef object iterator
        cdef list results

        if src_lang != "eng_Latn":
            normalizer = IndicNormalizerFactory().get_normalizer(iso_code)

        if visualize:
            iterator = tqdm(batch, total=len(batch), desc=f" | > Pre-processing {src_lang}", unit="line")
        else:
            iterator = batch

        results = []
        for s in iterator:
            results.append(self._preprocess(s, src_lang, tgt_lang, normalizer, is_target))

        return results

    cdef str _postprocess(self, object sent, str lang) except *:
        """
        Postprocess a single sentence:
        1) Pull placeholder map from queue
        2) Fix scripts for Perso-Arabic
        3) Restore placeholders
        4) Detokenize
        """
        cdef dict placeholder_entity_map
        cdef str lang_code, script_code, iso_lang, k, v, xlated

        # unwrap if sent is a tuple
        if isinstance(sent, (tuple, list)):
            sent = sent[0]

        placeholder_entity_map = self._placeholder_entity_maps.get()
        lang_code, script_code = lang.split("_", 1)
        iso_lang = self._flores_codes.get(lang, "hi")

        # Fix for Perso-Arabic
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

        # placeholders
        for k, v in placeholder_entity_map.items():
            sent = sent.replace(k, v)

        # detokenize
        if lang == "eng_Latn":
            return self._en_detok.detokenize(sent.split(" "))
        else:
            xlated = self._xliterator.transliterate(sent, "hi", iso_lang)
            return indic_detokenize.trivial_detokenize(xlated, iso_lang)

    cpdef list postprocess_batch(
        self,
        List[str] sents,
        str lang="hin_Deva",
        bint visualize=False
    ):
        """
        Postprocess a batch of sentences:
        restore placeholders, fix script issues, detokenize.
        """
        cdef object iterator
        cdef list results

        if visualize:
            iterator = tqdm(sents, total=len(sents), desc=f" | > Post-processing {lang}", unit="line")
        else:
            iterator = sents

        results = []
        for s in iterator:
            results.append(self._postprocess(s, lang))

        # clear placeholders
        self._placeholder_entity_maps.queue.clear()
        return results
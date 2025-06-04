# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
"""
Cython version of the IndicProcessor class with optimizations for performance.
Only preprocess_batch and postprocess_batch are exposed as cpdef methods.
All other methods are internal (cdef) for optimized Cython usage.
"""

import regex as re
from tqdm import tqdm
from queue import Queue
from typing import List, Dict, Union

# Importing Python objects since these libraries don't offer C-extensions
from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator


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

    # Placeholder maps stored in a Python Queue (treated as `object` for Cython)
    cdef object _placeholder_entity_maps

    # Tools (also Python objects)
    cdef object _en_tok
    cdef object _en_normalizer
    cdef object _en_detok
    cdef object _xliterator

    def __cinit__(self, bint inference=True):
        """
        Constructor for IndicProcessor. Initializes all necessary components.
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
        cdef dict digits_dict = {
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
            "आय . डी ."
            "आइ . डी . ",
            "आइ . डी .",
            "आइ. डी. ",
            "आइ. डी.",
            "ऐटि",
            "آئی ڈی ",
            "ᱟᱭᱰᱤ ᱾",
            "आयडी",
            "ऐडि",
            "आइडि",
            "ᱟᱭᱰᱤ",
        ]

    # Internal Method: Apply punctuation replacements
    cdef str _apply_punc_replacements(self, str text, list replacements) except *:
        """
        Apply a list of (pattern, replacement) in sequence to text.
        """
        cdef int i
        cdef tuple pair
        for i in range(len(replacements)):
            pair = replacements[i]
            text = pair[0].sub(pair[1], text)
        return text

    # Internal Method: Punctuation Normalization
    cdef str _punc_norm(self, str text) except *:
        """
        Consolidate punctuation normalization in fewer passes.
        """
        # 1) Apply replacements
        text = self._apply_punc_replacements(text, self._PUNC_REPLACEMENTS)

        # 2) Additional patterns
        text = self._MULTISPACE_REGEX.sub(" ", text)
        text = self._END_BRACKET_SPACE_PUNC_REGEX.sub(r")\1", text)
        text = self._DIGIT_SPACE_PERCENT.sub(r"\1%", text)
        text = self._DOUBLE_QUOT_PUNC.sub(r'\1"', text)
        text = self._DIGIT_NBSP_DIGIT.sub(r"\1.\2", text)
        return text.strip()

    # Internal Method: Wrap Text with Placeholders
    cdef str _wrap_with_placeholders(self, str text) except *:
        """
        Wrap substrings with matched patterns in the text with placeholders.
        Store the placeholder map in the queue for retrieval in postprocessing.
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
        cdef str match
        cdef str base_placeholder
        cdef int i

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
                # Map various placeholder formats to the matched text
                placeholder_entity_map[f"<ID{serial_no}>"] = match
                placeholder_entity_map[f"< ID{serial_no} >"] = match
                placeholder_entity_map[f"[ID{serial_no}]"] = match
                placeholder_entity_map[f"[ ID{serial_no} ]"] = match
                placeholder_entity_map[f"[ID {serial_no}]"] = match
                placeholder_entity_map[f"<ID{serial_no}]"] = match
                placeholder_entity_map[f"< ID{serial_no}]"] = match
                placeholder_entity_map[f"<ID{serial_no} ]"] = match

                placeholder_entity_map[f"<id{serial_no}>"] = match
                placeholder_entity_map[f"< id{serial_no} >"] = match
                placeholder_entity_map[f"[id{serial_no}]"] = match
                placeholder_entity_map[f"[ id{serial_no} ]"] = match
                placeholder_entity_map[f"[id {serial_no}]"] = match
                placeholder_entity_map[f"<id{serial_no}]"] = match
                placeholder_entity_map[f"< id{serial_no}]"] = match
                placeholder_entity_map[f"<id{serial_no} ]"] = match

                # Handle Indic failure cases
                for i in range(len(self._INDIC_FAILURE_CASES)):
                    indic_case = self._INDIC_FAILURE_CASES[i]
                    placeholder_entity_map[f"<{indic_case}{serial_no}>"] = match
                    placeholder_entity_map[f"< {indic_case}{serial_no} >"] = match
                    placeholder_entity_map[f"< {indic_case} {serial_no} >"] = match
                    placeholder_entity_map[f"<{indic_case} {serial_no}]"] = match
                    placeholder_entity_map[f"< {indic_case} {serial_no} ]"] = match
                    placeholder_entity_map[f"[{indic_case}{serial_no}]"] = match
                    placeholder_entity_map[f"[{indic_case} {serial_no}]"] = match
                    placeholder_entity_map[f"[ {indic_case}{serial_no} ]"] = match
                    placeholder_entity_map[f"[ {indic_case} {serial_no} ]"] = match
                    placeholder_entity_map[f"{indic_case} {serial_no}"] = match
                    placeholder_entity_map[f"{indic_case}{serial_no}"] = match

                # Replace the match with the base placeholder
                text = text.replace(match, base_placeholder)
                serial_no += 1

        # Clean up any remaining placeholder artifacts
        text = re.sub(r"\s+", " ", text).replace(">/", ">").replace("]/", "]")
        self._placeholder_entity_maps.put(placeholder_entity_map)
        return text

    # Internal Method: Normalize Text
    cdef str _normalize(self, str text) except *:
        """
        Normalizes numerals and optionally wraps placeholders.
        """
        # Single-pass digit translation
        text = text.translate(self._digits_translation_table)

        if self.inference:
            text = self._wrap_with_placeholders(text)
        return text

    # Internal Method: Indic Tokenize and Transliterate
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
        cdef str joined
        cdef str xlated

        normed = normalizer.normalize(sentence.strip())
        tokens = indic_tokenize.trivial_tokenize(normed, iso_lang)
        joined = " ".join(tokens)
        xlated = joined
        if transliterate:
            xlated = self._xliterator.transliterate(joined, iso_lang, "hi")
            xlated = xlated.replace(" ् ", "्")
        return xlated

    # Internal Method: Preprocess a Single Sentence
    cdef str _preprocess(
        self,
        str sent,
        str src_lang,
        str tgt_lang,
        object normalizer,
        bint is_target
    ) except *:
        """
        Preprocess a single sentence: punctuation normalization, numeral normalization,
        tokenization, transliteration, and adding language tags if necessary.
        """
        cdef str iso_lang = self._flores_codes.get(src_lang, "hi")
        cdef str script_part = src_lang.split("_")[1]
        cdef bint do_transliterate = True
        cdef str e_strip
        cdef str e_norm
        cdef list e_tokens
        cdef str processed_sent

        # 1) Punctuation normalization
        sent = self._punc_norm(sent)

        # 2) Numerals & placeholders
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

    # Internal Method: Postprocess a Single Sentence
    cdef str _postprocess(self, object sent, str lang, dict placeholder_entity_map=None) except *:
        """
        Postprocess a single sentence:
        1) Use provided placeholder map or pull from queue
        2) Fix scripts for Perso-Arabic
        3) Restore placeholders
        4) Detokenize
        """
        cdef str lang_code
        cdef str script_code
        cdef str iso_lang
        cdef str k
        cdef str v
        cdef str xlated

        # Unwrap if sent is a tuple or list
        if isinstance(sent, (tuple, list)):
            sent = sent[0]

        # Use provided map or get from queue
        if placeholder_entity_map is None:
            placeholder_entity_map = self._placeholder_entity_maps.get()
            
        lang_code, script_code = lang.split("_", 1)
        iso_lang = self._flores_codes.get(lang, "hi")

        # Fix for Perso-Arabic scripts
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

        # Detokenize
        if lang == "eng_Latn":
            return self._en_detok.detokenize(sent.split(" "))
        else:
            xlated = self._xliterator.transliterate(sent, "hi", iso_lang)
            return indic_detokenize.trivial_detokenize(xlated, iso_lang)

    # Exposed Method: Preprocess a Batch of Sentences
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
        This is exposed for external use.
        """
        cdef object normalizer = None
        cdef str iso_code = self._flores_codes.get(src_lang, "hi")
        cdef object iterator
        cdef list results
        cdef int i
        cdef int n = len(batch)

        if src_lang != "eng_Latn":
            normalizer = IndicNormalizerFactory().get_normalizer(iso_code)

        if visualize:
            iterator = tqdm(batch, total=n, desc=f" | > Pre-processing {src_lang}", unit="line")
        else:
            iterator = batch

        return [self._preprocess(s, src_lang, tgt_lang, normalizer, is_target) for s in iterator]

    # Exposed Method: Postprocess a Batch of Sentences
    cpdef list postprocess_batch(
        self,
        List[str] sents,
        str lang="hin_Deva",
        bint visualize=False,
        int num_return_sequences=1
    ):
        """
        Postprocess a batch of sentences:
        Restore placeholders, fix script issues, and detokenize.
        This is exposed for external use.
        
        Args:
            sents: List of sentences to postprocess
            lang: Target language code
            visualize: Whether to show progress bar
            num_return_sequences: Number of sequences returned per input
        """
        cdef object iterator
        cdef list results = []
        cdef list placeholder_maps = []
        cdef dict current_map
        cdef int i, j
        cdef int n = len(sents)
        cdef int num_inputs = n // num_return_sequences
        
        # First, collect all placeholder maps from the queue
        for i in range(num_inputs):
            placeholder_maps.append(self._placeholder_entity_maps.get())
        
        if visualize:
            iterator = tqdm(enumerate(sents), total=n, desc=f" | > Post-processing {lang}", unit="line")
        else:
            iterator = enumerate(sents)

        # Process each sentence with the appropriate placeholder map
        for i, sent in iterator:
            # Determine which placeholder map to use
            map_idx = i // num_return_sequences
            current_map = placeholder_maps[map_idx]
            results.append(self._postprocess(sent, lang, current_map))
        
        self._placeholder_entity_maps.queue.clear()
        
        return results

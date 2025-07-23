# Changelog

# Release v1.1.0
- 🔧 Migrate to Poetry and prepare for PyPi release!

# 📢 Release v1.0.5
- 🔧 Optimized and refactored the code for `IndicDataCollator` and `IndicEvaluator`

# 📢 Release v1.0.4
- 🐛 Bug fix for [Issue #18](https://github.com/VarunGumma/IndicTransToolkit/issues/18). In case of mulitple generations per source sentence, the map now applies to all targets. 

# 📢 Release v1.0.3
- 🚨 The `IndicProcessor` class has been re-written in [Cython](https://github.com/cython/cython) for faster implementation. This gives us atleast `+10 lines/s`.
- A new `visualize` argument as been added to `preprocess_batch` to track the processing with a `tqdm` bar.

# 📢 Release v1.0.2
- The repository has been renamed to `IndicTransToolkit`.
- 🚨 The custom tokenizer is now **removed** from the repository. Please revert to a previous commit ([v1.0.1](https://github.com/VarunGumma/IndicTransToolkit/tree/0e68fb5872f4d821578a5252f90ad43c9649370f)) to use it **(strongly discouraged)**. The official _(and only tokenizer)_ is available on HF along with the models.

# 📢 Release v1.0.0
- The [PreTrainedTokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer) for IndicTrans2 is now available on HF 🎉🎉 Note that, you still need the `IndicProcessor` to pre-process the sentences before tokenization.
- 🚨 **In favor of the standard PreTrainedTokenizer, we deprecated the custom tokenizer. However, this custom tokenizer will still be available here for backward compatibility, but no further updates/bug-fixes will be provided.**
- The `indic_evaluate` function is now consolidated into a concrete `IndicEvaluator` class.
- The data collation function for training is consolidated into a concrete `IndicDataCollator` class.
- A simple batching method is now available in the `IndicProcessor`.
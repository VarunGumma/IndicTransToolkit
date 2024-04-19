# IndicTransTokenizer

The goal of this repository is to provide a simple, modular, and extendable toolkit for [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) and be compatible with the HuggingFace models released. 

# Major Update (v1.0.0)
- The [PreTrainedTokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer) for IndicTrans2 is now available on HF 🎉🎉 Note that, you still need the `IndicProcessor` to pre-process the sentences before tokenization.
- **In favor of the standard PreTrainedTokenizer, we deprecated the custom tokenizer. However, this custom tokenizer will still be available here for backward compatibility, but no further updates/bug-fixes will be provided.**
- The `indic_evaluate` function is now consolidated into a concrete `IndicEvaluator` class.
- The data collation function for training is consolidated into a concrete `IndicDataCollator` class.
- A simple batching method is now available in the `IndicProcessor`.


## Pre-requisites
 - `Python 3.8+`
 - [Indic NLP Library](https://github.com/VarunGumma/indic_nlp_library)
 - Other requirements as listed in `requirements.txt`

## Configuration
 - Editable installation (Note, this may take a while):
```bash 
git clone https://github.com/VarunGumma/IndicTransTokenizer
cd IndicTransTokenizer

pip install --editable ./
```

## Examples
For the training usecase, please refer [here](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface). _Please do not use the custom tokenizer to train/fine-tune models. Training models with the custom tokenizer is untested and can lead to unexpected results._

### PreTainedTokenizer 
```python
import torch
from IndicTransTokenizer import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")
batch = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt")

with torch.inference_mode():
    outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

with tokenizer.as_target_tokenizer():
    # This scoping is absolutely necessary, as it will instruct the tokenizer to tokenize using the target vocabulary.
    # Failure to use this scoping will result in gibberish/unexpected predictions as the output will be de-tokenized with the source vocabulary instead.
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

outputs = ip.postprocess_batch(outputs, lang="hin_Deva")
print(outputs)

>>> ['यह एक परीक्षण वाक्य है।', 'यह एक और लंबा अलग परीक्षण वाक्य है।', 'कृपया 9876543210 पर एक एस. एम. एस. भेजें और 15 अक्टूबर, 2023 तक newemail123@xyz.com पर एक ईमेल भेजें।']
```

### Custom Tokenizer (Deprecated)
```python
import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")
batch = tokenizer(batch, src=True, return_tensors="pt")

with torch.inference_mode():
    outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

outputs = tokenizer.batch_decode(outputs, src=False)
outputs = ip.postprocess_batch(outputs, lang="hin_Deva")
print(outputs)

>>> ['यह एक परीक्षण वाक्य है।', 'यह एक और लंबा अलग परीक्षण वाक्य है।', 'कृपया 9876543210 पर एक एस. एम. एस. भेजें और 15 अक्टूबर, 2023 तक newemail123@xyz.com पर एक ईमेल भेजें।']
```

### Evaluation
`IndicEvaluator` is a python implementation of [compute_metrics.sh](https://github.com/AI4Bharat/IndicTrans2/blob/main/compute_metrics.sh)
```python
from IndicTransTokenizer import IndicEvaluator

# this method returns a dictionary with BLEU and ChrF2++ scores with appropriate signatures
evalutor = IndicEvalutor()
scores = evalutor.evaluate(tgt_lang=tgt_lang, preds=pred_file, refs=ref_file) 

# alternately, you can pass the list of predictions and references instead of files 
# scores = evalutor.evaluate(tgt_lang=tgt_lang, preds=preds, refs=refs)
```

### Batching 
```python
ip = IndicProcessor(inference=True)

for batch in ip.get_batches(source_sentences, batch_size=32):
    # perform necessary operations on the batch
    # ... pre-processing
    # ... tokenization 
    # ... generation 
    # ... decoding
```

## Authors
 - Varun Gumma (varun230999@gmail.com)
 - Jay Gala (jaygala24@gmail.com)
 - Pranjal Agadh Chitale (pranjalchitale@gmail.com)
 - Raj Dabre (prajdabre@gmail.com)


## Bugs and Contribution
Since this a bleeding-edge module, you may encounter broken stuff and import issues once in a while. In case you encounter any bugs or want additional functionalities, please feel free to raise `Issues`/`Pull Requests` or contact the authors. 


## Citation
If you use our codebase, models or tokenizer, please do cite the following paper:
```bibtex
@article{
    gala2023indictrans,
    title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
    author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=vfT4YuzAYA},
    note={}
}
```
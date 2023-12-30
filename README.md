# IndicTransTokenizer

The goal of this repository is to provide a simple, modular, and extendable tokenizer for [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) and be compatible with the HuggingFace models released. 

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

## Usage
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

For using the tokenizer to train/fine-tune the model, just set the `inference` argument of IndicProcessor to `False`.

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

## Note
This tokenizer module is currently **not** compatible with the [PreTrainedTokenizer](https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) module from HuggingFace. Hence, we are actively looking for `Pull Requests` to port this tokenizer to HF. Any leads on that front are welcome!

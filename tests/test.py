import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True
)

translations, bs = [], 16

sentences = [
    "This is a test sentence.",
    "*Apply Euclid's division algorithm to determine the Highest Common Factor (HCF) of $231$ and $396$.",
    "India's foreign exchange reserves increased by USD $1.153 billion to USD $585.895 billion for the week ending October 13, reversing a trend of multiple weeks of decline.",
]

for i in range(0, len(sentences), bs):
    batch = ip.preprocess_batch(
        sentences[i : i + bs], src_lang="eng_Latn", tgt_lang="hin_Deva"
    )
    batch = tokenizer(batch, src=True, return_tensors="pt")

    with torch.inference_mode():
        outputs = model.generate(
            **batch, num_beams=5, num_return_sequences=1, max_length=256
        )

    outputs = tokenizer.batch_decode(outputs, src=False)
    outputs = ip.postprocess_batch(outputs, lang="hin_Deva")
    translations.extend(outputs)

for o in translations:
    print(o)

# KPF BERT

## 사용방법

### Step 1. Installation

python>3.6 이어야 함
```
pip3 install torch>=1.4.0
pip3 install transformer>=4.9.2
```

### Step 2. Load Tokenizer, Model

```
from transformers import BertModel, BertTokenizer

model_name_or_path = "LOCAL_MODEL_PATH"  # Bert 바이너리가 포함된 디렉토리

model = BertModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
tokenizer = BertTokenizer.from_pretrained(model_name_or_path
```

### Step 3. Tokenizer
```
>>> text = "언론진흥재단 BERT 모델을 공개합니다."
>>> tokenizer.tokenize(text)
['언론', '##진흥', '##재단', 'BE', '##RT', '모델', '##을', '공개', '##합니다', '.']
>>> encoded_input = tokenizer(text)
>>> encoded_input
{'input_ids': [2, 7392, 24220, 16227, 28024, 21924, 7522, 4620, 7247, 15801, 518, 3],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

### Step 4. Model Inference

```
>>> import torch
>>> model.eval()
>>> pt_encoded_input = tokenizer(text, return_tensors="pt")
>>> model(**pt_encoded_input, return_dict=False)
(tensor([[[-4.1391e-01,  7.3169e-01,  1.1777e+00,  ...,  1.2273e+00, -4.1275e-01,  2.4145e-03],
          [ 1.6289e+00, -1.9552e-01,  1.6454e+00,  ...,  2.5763e-01, 1.7823e-01, -7.6751e-01],
          [ 7.4709e-01, -4.1524e-01,  3.0054e-01,  ...,  1.1636e+00, -2.3667e-01, -1.0005e+00],
          ...,
          [-7.9207e-01, -2.9005e-01,  1.7217e+00,  ...,  1.5060e+00, -2.3975e+00, -4.3733e-01],
          [-4.1402e-01,  7.3164e-01,  1.1777e+00,  ...,  1.2273e+00, -4.1289e-01,  2.3552e-03],
          [-4.1386e-01,  7.3167e-01,  1.1776e+00,  ...,  1.2273e+00, -4.1259e-01,  2.5745e-03]]],
          grad_fn=<NativeLayerNormBackward>), None)
```

## 총 5개의 모델에 대해서 평가 작업 수행

* kpfBERT base (https://github.com/KPFBERT/kpfbert)
* KLUE BERT base (https://huggingface.co/klue/bert-base)
* ETRI BERT base (KorBERT, https://aiopen.etri.re.kr/service_dataset.php)
* KoBERT (https://github.com/SKTBrain/KoBERT)
* BERT base multilingual cased (https://huggingface.co/bert-base-multilingual-cased)

### Sequence Classification 성능 측정 결과 비교 (10/22/2021):
| 구분 | NSMC | KLUE-NLI | KLUE-STS |
| :---       |     :---      |     :---      |    :---     |
| 데이터 특징 및 규격 | 영화 리뷰 감점 분석,<br> 학습 150,000 문장,<br> 평가: 50,000문장 | 자연어 추론,<br> 학습: 24,998 문장<br> 평가: 3,000 문장 (dev셋) | 문장 의미적 유사도 측정,<br> 학습: 11,668 문장<br> 평가: 519 문장 (dev셋) |
| 평가방법   | accuracy     | accuracy    | Pearson Correlation    |
| KPF BERT     | 91.29%       | 87.67%    | 92.95%      |
| KLUE BERT     | 90.62%       | 81.33%    | 91.14%      |
| KorBERT Tokenizer | 90.46%      | 80.56%    | 89.85%     |
| KoBERT     | 89.92%       |  79.53%    | 86.17%      |
| BERT base multilingual    | 87.33%       | 73.30%    | 85.66 %    |

### Question Answering 성능 측정 결과 비교 (10/22/2021):
| 구분 | KorQuAD v1 | KLUE-MRC |
| :---       |     :---      |      :---       |
| 데이터 특징 및 규격 | 기계독해,<br> 학습: 60,406 건<br> 평가: 5,774 건 (dev셋) | 기계독해,<br> 학습: 17,554 건<br> 평가: 5,841 건 (dev셋) |
| 평가방법   | Exact Match / F1 | Exact Match / Rouge W |
| KPF BERT     | 86.42% / 94.95% | 69.51 / 75.84% |
| KLUE BERT     | 83.84% / 93.23% | 61.91% / 68.38% |
| KorBERT Tokenizer | 20.11% / 82.00% | 30.56% / 58.59% |
| KoBERT     | 16.85% / 71.36% | 28.56% / 42.06 % |
| BERT base multilingual    | 68.10% / 90.02% | 44.58% / 55.92% |


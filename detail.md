
# KPF BERT 구축 전체 작업 일정 계획

1) BERT 를 위한 Vocabulary 제작
2) BERT Pretraining 코드 제작
3) 빅카인즈 뉴스 데이터 추출
4) 뉴스데이터 전처리 및 필터링
5) BERT 학습 말뭉치 선정
6) BERT Pretraining
7) BERT 모델 API 셋팅 및 가이드라인 제작
8) 평가 파이프라인 제작
9) 타 BERT 성능 측정, KPF BERT와 성능 비교
10) 결과분석 및 최종 보고서 작성
11) KPF BERT 활용 방안 연구

# 1. BERT를 위한 Vocabulary 제작

## 1.1. Vocabulary 제작을 위한 말뭉치 선정
### [모두의 말뭉치 원시 데이터](https://corpus.korean.go.kr/)

#### 신문 말뭉치
> 종합지, 전문지, 인터넷 기반 신문 매체의 기사(2009년~2018년)로 구성된 말뭉치

#### 구어 말뭉치
> 방송, 강연 등의 공적 구어 자료, 드라마 대본 등의 준구어 자료로 구성된 말뭉치

#### 문어 말뭉치
> 책, 잡지, 보고서 등으로 구성된 말뭉치

#### 일상대화 말뭉치 2020
> 특정주제 또는 제시 자료로 자유롭게 대화를 나눈 일상 대화 말뭉치

### [Common Crawl 의 한국어 데이터](https://commoncrawl.atlassian.net/wiki/display/CRWL/About+the+Data+Set)
* 웹 데이터를 사용한 이유는 정형화되지 않은 표현이나 이모지 등을 분석이 가능하도록 하기 위함
* 추후 뉴스 댓글 등까지 분석이 가능하도록 하기 위함

### 위의 데이터에서 일부를 Sampling 하여 사용
* 뉴스 데이터 중에서 한자의 비중이 높은 데이터를 많이 넣음
* 총 데이터 양은 20GB 정도

## 1.2. Vocabulary 제작

### [Huggingface tokenizers](https://github.com/huggingface/tokenizers) 라이브러리를 이용하여 Bert Wordpiece용 Vocabulary 제작

### 형태소 기반 하위 단어 토큰화 (morpheme-based subword tokenization)를 사용
* [KLUE Paper에서 소개된 기법](https://arxiv.org/abs/2105.09680)
  - 형태소분석기로 먼저 말뭉치를 pretokenize한 후 wordpiece로 학습 <br>
  - 그리고 실제 사용할 때는 형태소분석기를 사용하지 않고 wordpiece만 사용함
* 형태소 분석기의 경우 [은전한닢의 Mecab](https://bitbucket.org/eunjeon/mecab-ko-dic) 을 사용함

### Vocab 구성
* (1) 위에서 언급한 모든 말뭉치를 사용하여 vocab_size=32000 의 vocabulary를 제작
* (2) 신문말뭉치만 이용하여 vocab_size=32000 의 vocabulary를 제작
* (1)의 vocab에 (2)에서만 존재하는 단어 4440개를 추가하여 vocab 강화
  - 뉴스에서 자주 나오는 단어들에 잘 대응하기 위해 사용한 기법
* unused token은 500개를 추가하여, 추후에 vocabulary에 없는 단어를 추가할 수 있게 함

## 1.3. Tokenizer 파일 산출물
### special_token_map.json

```
{
 "unk_token": "[UNK]",
 "sep_token": "[SEP]",
 "pad_token": "[PAD]",
 "cls_token": "[CLS]",
 "mask_token": "[MASK]"
}
```
### tokenizer_config.json
```
{
 "do_lower_case": false,
 "do_basic_tokenize": true,
 "never_split": null,
 "unk_token": "[UNK]",
 "sep_token": "[SEP]",
 "pad_token": "[PAD]",
 "cls_token": "[CLS]",
 "mask_token": "[MASK]",
 "tokenize_chinese_chars": true,
 "strip_accents": null,
 "special_tokens_map_file": null,
 "tokenizer_file": null,
 "tokenizer_class": "BertTokenizer"
}
```
### 생성된 vocab.txt 파일
> 파일 사이즈 276KB, 36440 Lines
```
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
[unused0]
[unused1]
[unused2]
.
.
.
이용권
##바이러스
패권주의
FS
일로
루지
북고
조련
```
### 1.4. Tokenizer 사용법
```
> pip3 install transformers>=4.9.2

>>> from transformers import BertTokenizer
>>> tokenizer = BertTokenizer.from_pretrained("kpf-tokenizer")
>>> tokenizer.tokenize("선선한 날씨에 곳곳에서 피서객이 몰렸습니다.")
['선선', '##한', '날씨', '##에', '곳곳', '##에서', '피서객', '##이', '몰렸', '##습', '##니다', '.']
```
* 형태소 분석기를 사용하지 않았음에도 형태소 단위로 어느정도 잘 쪼개지는 것을 확인할 수 있음

# 2. BERT Pretraining 코드 제작
> BERT 원본 코드를 기반으로 개선사항들을 추가하며 코드 수정 (아래는 참고한 코드)
* [BERT tensorflow v1 implementation](https://github.com/google-research/bert)
* [BERT tensorflow v2 implementation](https://github.com/tensorflow/models/tree/d4c5f8975a7b89f01421101882bc8922642c2314/official/nlp/bert)
* [ELECTRA implementation](https://github.com/google-research/electra)

## 2.1. TFRecord 파일 생성
* TFRecord 파일은 TensorFlows의 학습 데이타 등을 저장하기 위한 바이너리 데이타 포맷으로, 구글의 Protocol Buffer Format 으로 데이타를 파일에 Serialize 하여 저장한다.
* raw text를 학습을 위한 형태로 변환하는 과정
* 기존 BERT 구현에서 추가적인 기능을 적용 (이를 통해 성능 향상)
  - whole word masking, n gram 적용
    - SpanBERT 등의 여러 논문에서 이 방법을 적용하여 Question Answering 태스크의 성능을 높임
  - 기존 BERT에서는 두 개의 segment ([CLS] text_a [SEP] text_b [SEP])로 붙이는 비율이 높았지만, 이 비율을 줄이고 한 개의 segment로 max_seq_length 512까지 붙이는 비율을 더 많이 높임 (RoBERTa 등의 논문에 기반하여 해당 로직을 채택)
- multiprocessing 적용을 통한 시간 단축 (기존 16일 → 2일)

## 2.2. Pretraining 작업 수행
* 기존 BERT에서 사용된 Next Sentence Prediction (NSP)은 사용하지 않음
  - Masked Language Model만 적용하여 학습
  - BERT 이후의 여러 논문에서 NSP를 사용하지 않을 때 더 좋은 성능을 보였다고 함
* Pretraining Seed를 지정하는 부분 추가
  - 구글의 The MultiBERTs 에 따르면 Pretraining시에 사용된 seed도 성능차이에 영향을 미친다고 나와있고, 이에 다양한 seed로 학습을 시작하여 가장 성능이 좋은 모델을 채택하려 함
  - 그 외에 학습 소요 시간 확인 등의 utility 기능 추가

## 2.3. Convert Tensorflow checkpoint to Pytorch checkpoint
* NSP를 사용하지 않는 모델이어서 이에 맞게 변환 코드를 수정함
  - BertForPretraining → BertForMaskedLM
* 또한 torch>=1.5 이상으로 변환할 경우 torch<=1.4 에서 작동하지 않는 버그가 존재하는데, 이는 아래의 코드로 해결함
```
torch.save(
    model.state_dict(),
    pytorch_dump_path,
    _use_new_zipfile_serialization=False,  # NOTE For compatibility with torch<1.5
)
```

# 3. 빅카인즈 뉴스 데이터 추출

* 추출 데이터 원본 파일 크기 : 778GB
* 전체 뉴스 기사 기간 : 1945년부터 2021년 7월 뉴스기사 데이터
* 전체 뉴스 기사 건수 : 90836373건

# 4. 뉴스 데이터 전처리 및 필터링
## 4.1. 전처리 사용 로직

* 뉴스에서 기자, 뉴스이름 등과 관련된 표현 최대한 제거
  - e.g. (관련기사) , [이데일리 XXX 기자], (저작권자), (무단 전재) , (ⓒ 연합뉴스), (ⓒ 뉴데일리)
  - 해당 데이터는 BERT 학습에 노이즈로 작용하기에 엄격한 기준으로 제거함
* 추가로 제거한 데이터
  - html 태그
  - copyright ( r"ⓒ|©|(copyrights?)|(\\(c\\))|(\\(C\\))" )
  - 이메일
  - URL
  - 기자 이름
  - {IMG:1} {VOD:3}
  - [], 【】,［］,〔〕 로 묶여있는 것은 무조건 제거함
  - ()로 묶인 것 중 (뉴스) , (기자) 와 같은 것은 제거함
  - (사진), (영상), (사진제공) 과 같이 이미지 소스와 관련된 표현 제거
  - (기사입력), (날짜) 와 같이 작성 시간과 관련된 표현 제거
  - double whitespace 제거를 통한 sequence 길이 축소
* [kss](https://github.com/likejazz/korean-sentence-splitter) 를 이용하여 문장 분리 진행
  - 문장분리를 미리하면 BERT의 input을 만들 때 max sequence length인 512에 맞게 효율적으로 제작할 수 있음

## 4.3. 기사 필터링
* 음절 기준 400 이하는 필터링
  - 통계를 낸 결과 전체 연도 중 최소 평균 길이가 400이었음
* 뉴스 기사 중 제목에 아래 단어가 표함된 경우는 제거함
  - 인사, 승진, 부고
  - 사진, PHOTO, 화보, 지도, 그림
  - 장종리포트, 장마감보고서
* 종목, 주식, 보고서, 하락 등의 제목을 가지고 있는 데이터는 제거
  - 시기에 따라서 같은 종목에 대해서 주가 등이 다르기에, 모델 입장에서 학습시 혼동이 될 수 있음
* kss로 문장을 분리했을 때 3문장 이하인 것은 제거
  - T5 논문(https://arxiv.org/abs/1910.10683)에서 C4 (The Colossal Clean Crawled Corpus)를 제작할 때 사용한 방법을 일부 사용
* 중복 제거
  - 하나의 기사에서 3개 이상의 문장이 이미 과거의 기사에서도 등장했으면 사용하지 않음 (이 역시 T5 논문에서 착안)
  - 제목의 경우 한글만 남긴 후, 이미 등장한 제목의 기사일 경우에는 제거
  - 본문의 경우도 중복될 시 제거
  - 위의 문장, 제목 , 본문 의 경우 hash로 변환하여 중복 탐색을 진행함

## 4.4. 기사 필터링 이후의 데이터
* 최종 데이터 크기는 125GB로 줄어듬

# 5. BERT 학습 말뭉치 선정
* 모두의 말뭉치의 기본 원시 말뭉치 사용 (웹, 구어, 문어, 일상대화)
* 언론진흥재단 뉴스 말뭉치는 2000~2021년 데이터를 사용하기로 결정
  - 2000년 이전의 뉴스에서는 한자, 옛날 표현의 비중이 높아 이러한 부분이 BERT 학습에 노이즈로 작용할 수 있다고 판단

# 6. BERT Pretraining

## 6.1. BERT Pretraining 산출물
* [KPF BERT 20211012 Release Version](https://github.com/KPFBERT/kpfbert)
* pytorch_model.bin 과 tf_model.h5 가 각각 4~500MB 용량, 전체 1G 정도 용량임

# 7. BERT 모델 API 셋팅 및 가이드라인 제작

## 7.1. Installation
* python>=3.6 이어야 함
  - pip3 install torch>=1.4.0
  - pip3 install transformers>=4.9.2

## 7.2. 모델 디렉토리 구성
* pytorch_model.bin
  - BERT 바이너리 파일
* config.json
  - BERT 모델 관련 configuration 저장
* vocab.txt
  - tokenizer의 vocabulary
* special_tokens_map.json
  - [CLS], [SEP] 등 special token을 정의
* tokenizer_config.json
  - tokenizer 관련 configuration 저장

## 7.3. Model Configuration
* bert-base의 사이즈를 사용
```
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 36440
}
```

## 7.4. Usage
* 디테일한 사용법은 [transformers documentation](https://huggingface.co/transformers/) 참고 

## 7.5. Load Tokenizer, Model
```
from transformers import BertModel, BertTokenizer

model_name_or_path = "LOCAL_MODEL_PATH"  # Bert 바이너리가 포함된 디렉토리

model = BertModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
```
## 7.6. Tokenizer
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
## 7.7. Model Inference
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
          [-4.1386e-01,  7.3167e-01,  1.1776e+00,  ...,  1.2273e+00, -4.1259e-01,  2.5745e-03]]], grad_fn=<NativeLayerNormBackward>), None)
```

# 8. 평가 파이프라인 제작
* [NSMC : Naver sentiment movie corpus (네이버 영화리뷰 감정분석)](https://github.com/e9t/nsmc)

* [KLUE : Korean Language Understanding Evaluation (한국어 이해 평가)](https://github.com/KLUE-benchmark/KLUE)
  - NLI : Natural Language Inference (자연어 추론)
  - STS : Sentence Textual Similarity (문장 텍스트 유사성)
  - MRC : Machine Reading Comprehension (기계 독해력)

* [KorQuad : The Korean Question Answering Dataset (한국어 MRC 데이터셋)](https://korquad.github.io/KorQuad%201.0/)


## 8.1. Sequence Classification Task
### 8.1.1. nsmc (Naver sentiment movie corpus, 네이버 영화리뷰 감정분석)
* https://github.com/e9t/nsmc
* Single Sentence Classification
  - 영화 리뷰에 대해 긍정, 부정을 예측하는 태스크
* 학습: 150000 문장, 평가 : 50000 문장
* 평가방법 : accuracy
* 데이터 예제
```
id      document        label
9976970 아 더빙.. 진짜 짜증나네요 목소리        0
3819312 흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나        1
10265843        너무재밓었다그래서보는것을추천한다      0
9045019 교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정       0
6483659 사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다  1
5403919 막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.     0
7797314 원작의 긴장감을 제대로 살려내지못했다.  0
9443947 별 반개도 아깝다 욕나온다 이응경 길용우 연기생활이몇년인지..정말 발로해도 그것보단 낫겟다 납치.감금만반복반복..이드라마는 가족도없다 연기못하는사람만모엿네       0
7156791 액션이 없는데도 재미 있는 몇안되는 영화 1
```

### 8.1.2. KLUE NLI (Natural Language Inference, 자연어 추론)
* https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark/klue-nli-v1.1
* Sentence Pair Classification
  - 두 문장 (premise, hypothesis)를 입력 받아 두 문장의 관계를 분류
  - entailment, contradiction, neutral
* 학습 : 24998 문장, 평가 : 3000 문장
* 평가방법: accuracy
* 데이터 예제
```
[
    {
        "guid": "klue-nli-v1_train_00000",
        "genre": "NSMC",
        "premise": "힛걸 진심 최고다 그 어떤 히어로보다 멋지다",
        "hypothesis": "힛걸 진심 최고로 멋지다.",
        "gold_label": "entailment",
        "author": "entailment",
        "label2": "entailment",
        "label3": "entailment",
        "label4": "entailment",
        "label5": "entailment"
    },
    {
        "guid": "klue-nli-v1_train_00001",
        "genre": "NSMC",
        "premise": "100분간 잘껄 그래도 소닉붐땜에 2점준다",
        "hypothesis": "100분간 잤다.",
        "gold_label": "contradiction",
        "author": "contradiction",
        "label2": "contradiction",
        "label3": "contradiction",
        "label4": "neutral",
        "label5": "contradiction"
    },
    {
        "guid": "klue-nli-v1_train_00002",
        "genre": "NSMC",
        "premise": "100분간 잘껄 그래도 소닉붐땜에 2점준다",
        "hypothesis": "소닉붐이 정말 멋있었다.",
        "gold_label": "neutral",
        "author": "neutral",
        "label2": "neutral",
        "label3": "neutral",
        "label4": "neutral",
        "label5": "neutral"
    },
]
```

### 8.1.3. KLUE STS (Semantic textual Similarity, 문장 텍스트 유사성)
* https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark/klue-sts-v1.1
* Sentence Pair Regression
  - 두 문장 사이의 의미적 유사성의 정도를 평가
  - 0 ~ 5 (dissimilar ~ equivalent)
*학습 : 11668 문장, 평가 : 519 문장
*평가방법: pearsonr (Pearson correlation coefficient)
*데이터 예제
```
{
    "guid": "klue-sts-v1_train_00000",
    "source": "airbnb-rtt",
    "sentence1": "숙소 위치는 찾기 쉽고 일반적인 한국의 반지하 숙소입니다.",
    "sentence2": "숙박시설의 위치는 쉽게 찾을 수 있고 한국의 대표적인 반지하 숙박시설입니다.",
    "labels": {
        "label": 3.7,
        "real-label": 3.714285714285714,
        "binary-label": 1
    },
    "annotations": {
        "agreement": "0:0:0:2:5:0",
        "annotators": [
            "07",
            "13",
            "15",
            "10",
            "12",
            "02",
            "19"
        ],
        "annotations": [
            3,
            4,
            4,
            4,
            3,
            4,
            4
        ]
    }
}
```

## 8.2. Question Answering Task
### 8.2.1. KorQuAD v1
* https://korquad.github.io/KorQuad 1.0/
* 학습 : 60406 건, 평가 : 5774 건
* 평가방법
  - exact match: 시스템이 제시한 결과와 정답이 완전히 일치하는 비율
  - f1: 정확률(Precision, 시스템이 결과가 정답인 비율)과 재현률(Recall, 실제 정답을 시스템이 맞춤 비율)의 조화평균
* 데이터 예제
```
{
  "qas": [
    {
      "answers": [
        {
          "text": "허영",
          "answer_start": 100
        }
      ],
      "id": "6548850-1-0",
      "question": "정부의 헌법개정안 준비 과정에 대해서 청와대 비서실이 아니라 국무회의 중심으로 이뤄졌어야 했다고 지적한 원로 헌법학자는?"
    },
    {
      "answers": [
        {
          "text": "10차 개헌안 발표",
          "answer_start": 77
        }
      ],
      "id": "6548850-1-1",
      "question": "'행보가 비서 본연의 역할을 벗어난다', '장관들과 내각이 소외되고 대통령비서실의 권한이 너무 크다'는 의견이 제기된 대표적인 예는?"
    },
    {
      "answers": [
        {
          "text": "제89조",
          "answer_start": 192
        }
      ],
      "id": "6332405-1-0",
      "question": "국무회의의 심의를 거쳐야 한다는 헌법 제 몇 조의 내용인가?"
    },
    {
      "answers": [
        {
          "text": "허영",
          "answer_start": 100
        }
      ],
      "id": "6332405-1-1",
      "question": "법무부 장관을 제쳐놓고 민정수석이 개정안을 설명하는 게 이해가 안 된다고 지적한 경희대 석좌교수 이름은?"
    }
  ],
  "context": "\\"내각과 장관들이 소외되고 대통령비서실의 권한이 너무 크다\\", \\"행보가 비서 본연의 역할을 벗어난다\\"는 의견이 제기되었다. 대표적인 예가 10차 개헌안 발표이다. 원로 헌법학자인 허영 경희대 석좌교수는 정부의 헌법개정안 준비 과정에 대해 \\"청와대 비서실이 아닌 국무회의 중심으로 이뤄졌어야 했다\\"고 지적했다. '국무회의의 심의를 거쳐야 한다'(제89조)는 헌법 규정에 충실하지 않았다는 것이다. 그러면서 \\"법무부 장관을 제쳐놓고 민정수석이 개정안을 설명하는 게 이해가 안 된다\\"고 지적했다. 민정수석은 국회의원에 대해 책임지는 법무부 장관도 아니고, 국민에 대해 책임지는 사람도 아니기 때문에 정당성이 없고, 단지 대통령의 신임이 있을 뿐이라는 것이다. 또한 국무총리 선출 방식에 대한 기자의 질문에 \\"문 대통령도 취임 전에 국무총리에게 실질적 권한을 주겠다고 했지만 그러지 못하고 있다. 대통령비서실장만도 못한 권한을 행사하고 있다.\\"고 답변했다."
}
```

### 8.2.2. KLUE MRC (Machine Reading Comprehension, 기계 독해력)
* https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark/klue-mrc-v1.1
* 학습: 17554 건, 평가: 5841 건
* 평가방법
  - exact match : 시스템이 제시한 결과와 정답이 완전히 일치하는 비율
  - rouge_w : longest common consecutive subsequence (LCCS) 기반 F1
* 데이터 예제
```
{
    "title": "도전적인 일, 리더에 대한 신뢰, 협력 문화 …...가고 싶다, 이 회사 … 베인&컴퍼니·트위터",
    "paragraphs": [
        {
            "context": "가장 일하고 싶은 회사로 미국 컨설팅 업체인 베인&컴퍼니가 선정됐다. 트위터와 링크트인, 페이스북 등 정보기술(IT) 기업에 대한 선호도 높게 나타났다. 미국 경제전문지 포브스는 “채용 사이트인 글라스도어가 미국에서 일하는 50만명의 직장인을 대상으로 설문조사를 벌여 ‘2014년 일하고 싶은 50대 회사’를 선정했다”고 11일(현지시간) 보도했다. 베인&컴퍼니는 2012년에 이어 두 번째로 1위에 올랐다. 하는 일이 재미있고 일이 영향력이 있으며, 함께 일하는 사람들이 똑똑해서 많이 배울 수 있다는 것 등이 직원들이 베인&컴퍼니에 만족하는 주된 이유였다. 서맨사 주판 글라스도어 대변인은 “베인&컴퍼니 직원들은 늘 새로운 것을 배울 수 있고 도전적인 일이 많아 하루도 지루한 날이 없다고 평가했다”며 “이런 경험은 회사를 그만둔 후에도 개인 경력에 도움이 된다”고 말했다. 트위터가 2위를 차지했다. 트위터 직원들은 복잡하고 중요한 일을 매일 하면서 전 세계 수백만명의 사용자들과 소통할 수 있다는 점을 회사에 다니고 싶은 이유로 꼽았다. 트위터에 이어 링크트인과 페이스북이 각각 3위, 5위를 차지하는 등 IT 기업의 선전도 눈에 띄었다. 글라스도어는 “톱 50 가운데 22개가 IT 기업”이라며 “서로 더 좋은 인재를 영입하고, 기존의 인재를 뺏기지 않기 위해 경쟁하는 것이 더 매력적인 기업환경을 만들기 위한 노력으로 이어지고 있다”고 설명했다. 2009년부터 일하고 싶은 회사를 선정해 발표하고 있는 글라스도어는 2012년 11월부터 지난달까지 회사에 대한 만족도, 임금이나 복지 혜택, 일과 가정 사이의 균형, 경영자에 대한 평가 등 18개의 질문으로 구성된 설문조사를 실시했다. 글라스도어는 “구직자들이 어느 회사에서 일할지 결정하는 데 좋은 정보를 주는 자료”라며 “특히 현장에서 일하는 직원의 평가라는 점에서 리스트에 오른 기업들은 자부심을 가져도 될 것”이라고 평가했다. 상위권에 오른 기업들은 공통점을 가지고 있었다. 미국 경제경영전문지 패스트컴퍼니는 일하고 싶은 기업의 6가지 특징으로 목표를 분명하게 설정하는 것, 일하는 과정에서 협동을 중요시하는 것, 도전적인 일을 하는 것, 회사가 꾸준히 발전하는 것, 리더가 자신감을 갖고 투명하게 경영하는 것, 충분한 보상을 하는 것 등을 꼽았다.",
            "qas": [
                {
                    "question": "퇴사하고 싶은 회사로 트위터가 순위에 선정된 해는?",
                    "answers": [],
                    "plausible_answers": [
                        {
                            "text": "2014년",
                            "answer_start": 151
                        },
                        {
                            "text": "2014",
                            "answer_start": 151
                        }
                    ],
                    "question_type": 3,
                    "is_impossible": true,
                    "guid": "klue-mrc-v1_dev_03566"
                },
                {
                    "question": "2014년 일하고 싶은 50대 회사 중에서 5위로 선정된 기업은?",
                    "answers": [
                        {
                            "text": "페이스북",
                            "answer_start": 543
                        }
                    ],
                    "question_type": 2,
                    "is_impossible": false,
                    "guid": "klue-mrc-v1_dev_02673"
                },
                {
                    "question": "포브스의 2014년 일하고 싶은 50대 회사 조사에서 5위를 한 기업은?",
                    "answers": [
                        {
                            "text": "페이스북",
                            "answer_start": 543
                        }
                    ],
                    "question_type": 1,
                    "is_impossible": false,
                    "guid": "klue-mrc-v1_dev_05664"
                }
            ]
        }
    ],
    "news_category": "국제",
    "source": "hankyung"
}
```
* rouge_w 와 f1 의 차이점
  - 기존의 char f1은 char의 순서와 상관없이 단순히 overlap으로만 계산

# 9. 타 BERT 성능 측정, kpfBERT와 성능 비교
## 9.1. Evaluation Code

* kpfbert-evaluation-211022.zip
  - (scikit-learn==0.24 로 변경한 버전)

## 9.2. About
* pytorch-lightning, transformers 를 이용한 모델 평가
* 다양한 학습 기능 구현
  - early stopping 지원
  - model versioning
  - mixed precision 학습 지원 (Tensor Core를 지원하는 Volta 이상 GPU에서 정상적으로 작동)
* 기존 korbert의 tokenizer를 최신 huggingface transformers에 정상 작동하도록 수정함
  - kpfbert_evaluation/tokenizer/tokenization_etribert.py 참고

## 9.3. Requirements
* python>=3.6, torch>=1.6 으로 반드시 진행해야 함
* anaconda 환경 사용을 권장
```
conda create -n kpfbert python=3.6
conda activate kpfbert

conda install pytorch==1.7.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
* pytorch 설치법은 아래의 링크 참고
  - https://pytorch.org/get-started/locally/

## 9.4. Before Evaluation
* 모델 바이너리인 kpfbert-base 와 korbert-base 를 해당 코드와 같은 레벨에 위치시켜야 함
  - 만일 바이너리를 포함한 폴더의 이름이 위와 다를 경우, 위의 이름과 동일하게 변경해야 함

## 9.5. Run Evaluation
* 총 5개의 모델에 대해 평가
  - kpfBERT base (https://github.com/KPFBERT/kpfbert)
  - KorBERT base (https://aiopen.etri.re.kr/service_dataset.php)
  - KoBERT (https://github.com/SKTBrain/KoBERT)
  - KLUE BERT base (https://huggingface.co/klue/bert-base)
  - BERT base multilingual cased (https://huggingface.co/bert-base-multilingual-cased)

* run_all.sh 를 실행
  - 해당 스크립트는 aws의 g4dn.xlarge의 Tesla T4 (16GB) 를 기준으로 작성
  - 사용하는 GPU의 메모리에 맞춰 train_batch_size 와 eval_batch_size 를 조절
  - 일부 GPU에서는 mixed precision을 미지원할 수 있습니다. 이럴 시 --fp16 옵션을 제거하여 돌리면 됩니다 (이 옵션을 제거하면 사용 GPU 메모리와 학습 시간이 늘어납니다.)

* Main Arguments
```
--task TASK           Run one of the task in ['klue-nli', 'klue-sts', 'klue-mrc', 'nsmc', 'korquadv1']
--data_dir DATA_DIR   The input data dir
--output_dir OUTPUT_DIR
                      The output directory where the model predictions and checkpoints will be written.
--max_seq_length MAX_SEQ_LENGTH
                      The maximum total input sequence length after tokenization. Sequences longer
                      than this will be truncated, sequences shorter will be padded.
--num_train_epoch NUM_TRAIN_EPOCH
                      Max training epochs
--train_batch_size TRAIN_BATCH_SIZE
                      Batch size for training
--eval_batch_size EVAL_BATCH_SIZE
                      Batch size for evaluation
--lr_scheduler LR_SCHEULDER
                      Learning rate scheduler. Default as `linear`.
--warmup_ratio WARMUP_RATIO
                      Linear warmup over warmup_step ratio.
--gpus GPUS           Select specific GPU allocated for this, it is by default [] meaning none
--fp16                Whether to use 16-bit (mixed) precision instead of 32-bit
--gradient_accumulation_steps ACCUMULATE_GRAD_BATCHES
                      Number of updates steps to accumulate before
                      performing a backward/update pass.
--seed SEED           random seed for initialization
--metric_key METRIC_KEY
                      The name of monitoring metric (for early stopping)
--patience PATIENCE   The number of validation epochs with no improvement
                      after which training will be stopped. (for early stopping)
--early_stopping_mode {min,max}
                      In min mode, training will stop when the quantity
                      monitored has stopped decreasing; in max mode it will
                      stop when the quantity monitored has stopped
                      increasing;
--val_check_interval VAL_CHECK_INTERVAL
                      Check validation set X times during a training epoch
```

# 10. 결과분석 및 최종 보고서 작성 (11월중)
## 10.1. Sequence Classification 성능 측정 결과 비교

| 구분 | NSMC | KLUE-NLI | KLUE-STS |
| :---       |     :---      |     :---      |    :---     |
| 데이터 특징 및 규격 | 영화 리뷰 감점 분석,<br> 학습 150,000 문장,<br> 평가: 50,000문장 | 자연어 추론,<br> 학습: 24,998 문장<br> 평가: 3,000 문장 (dev셋) | 문장 의미적 유사도 측정,<br> 학습: 11,668 문장<br> 평가: 519 문장 (dev셋) |
| 평가방법   | accuracy     | accuracy    | Pearson Correlation    |
| KPF BERT     | 91.29%       | 87.67%    | 92.95%      |
| KLUE BERT     | 90.62%       | 81.33%    | 91.14%      |
| KorBERT Tokenizer | 90.46%      | 80.56%    | 89.85%     |
| KoBERT     | 89.92%       |  79.53%    | 86.17%      |
| BERT base multilingual    | 87.33%       | 73.30%    | 85.66 %    |

## 10.2. Question Answering 성능 측정 결과 비교
| 구분 | KorQuAD v1 | KLUE-MRC |
| :---       |     :---      |      :---       |
| 데이터 특징 및 규격 | 기계독해,<br> 학습: 60,406 건<br> 평가: 5,774 건 (dev셋) | 기계독해,<br> 학습: 17,554 건<br> 평가: 5,841 건 (dev셋) |
| 평가방법   | Exact Match / F1 | Exact Match / Rouge W |
| KPF BERT     | 86.42% / 94.95% | 69.51 / 75.84% |
| KLUE BERT     | 83.84% / 93.23% | 61.91% / 68.38% |
| KorBERT Tokenizer | 20.11% / 82.00% | 30.56% / 58.59% |
| KoBERT     | 16.85% / 71.36% | 28.56% / 42.06 % |
| BERT base multilingual    | 68.10% / 90.02% | 44.58% / 55.92% |

## 10.3. 공개된 KorBERT (ETRI) 의 평가 결과 데이더 분석
| 구분 | KorQuAD v1 |
| :---       |     :---      |
| 평가방법   | Exact Match / F1 | 
| KPF BERT | 86.42% / 94.95% |
| KorBERT 형태소 분석기 기반 | 86.40% / 94.18% |
| KorBERT Tokenizer기반 | 80.70% / 91.94%<br>(정답 경계 구분을 위해 후처리 수행) |


* ETRI Wordpiece 기반 BERT 의 점수는 etri에서 사용한 후처리 후 점수와 비교
  - https://aiopen.etri.re.kr/service_dataset.php
* 형태소분석기를 추가적으로 사용한 Etri BERT 모델에 비해 KPF BERT 가 더 높은 f1 점수를 보임
  - 후처리를 진행한 ETRI wordpiece BERT 에 비해서도 더 높은 점수를 보임.
  - 자체적인 형태소 분석기 없이도 충분히 좋은 성능을 보일 수 있음.
  - 실사용시 inference 속도 개선 및 tokenizer 파이프라인 단순화 가능

## 10.4. 나올 수 있는 질문, FAQ 정리
* KorBERT(ETRI)의 경우 Wordpiece 기반 과 형태소 기반 모델 중 전자를 이용하여 평가했습니다
  - 형태소 기반 모델의 경우 형태소 분석기 API를 호출해야 하는데, 하루 최대 5000건만 호출할 수 있는 구조여서 평가에 어려움이 있었고, 이에 Wordpiece 기반 모델로 평가
* 왜 KorBERT(ETRI)와 KoBERT(SKT)는 Question Answering 에서 점수가 낮은가
  - 두 모델의 tokenizer는 순수한 bpe 알고리즘만 사용하여 학습했기에 조사 등이 제대로 분리되지 않음
    - 또한 KoBERT의 경우 vocab size가 8002의 작은 사이즈인 것도 원인 중 하나

  - KPF BERT를 제작할 때는 KLUE에서 제시한 방법인 morpheme-based subword tokenization 를 사용하여 이러한 부분을 해소함
  - 참고로 mBERT의 경우 점수가 높게 나오는 이유는, vocab 안의 대부분의 단어가 한글자 (가 , ##가 )로 이루어져 있어 조사 분리 실패 등의 이슈에서 상대적으로 자유로워서임.
    - e.g. ['오', '##늘', '##은', '날', '##씨', '##가', '정', '##말', '화', '##창', '##합', '##니다']
* 왜 KorBERT(ETRI) 의 KorQuAD의 경우 공개된 점수와 직접 평가한 점수 간의 차이가 심한가 
```
(3) [KorBERT 모델] Korean_BERT_WordPiece 모델을 사용하여, 기계독해(MRC) 태스크에 적용시 후처리를 적용해야 하나요?
(답변) WordPiece 모델은 형태소 분석을 수행하지 않는 모델로, 조사/어미와 같은 음절이 선행 음절과 결합되는 경우가 자주 발생합니다. (예: 구성된다 -> 구 + ##성된 + ##다)
예를 들어, WordPiece 모델에서는 "단어는"과 같은 어절을 "단"과 "어는" 처럼 형태소와 다른 단위로 구분합니다. 따라서, 기계독해 모델의 정답이 "단어"일 경우, "단"과 "어는"이라는 wordpiece를 정답 경계로 인식 후, 조사 "는"을 필터링하는 단계가 필요합니다.
구체적인 후처리 규칙은 사용하시는 기계독해 데이터의 dev 셋에서, 시스템 결과의 정답 결과를 비교하여 보시면 후처리 대상 규칙을 정리하실 수 있습니다.
```
> 후처리에 관한 etri 측의 답변

* 위에서 설명한 이유와 연결되는데, 조사 등이 잘 분리되지 않는 형태로 prediction이 나오다보니 후처리를 하지 않으면 점수가 낮게 나오게됨
* 그러나 ETRI 측에서 약 5000개의 결과물에 대해 일일이 후처리를 수행했고, 이에 em과 f1 점수가 높아지게 됨
* 이는 prediction의 양이 늘어날 경우에 현실적으로 이용할 수 있는 방법이 아님
* 모든 모델에 대해 동등한 조건으로 평가하기 위해 후처리를 하지 않은 형태로 5개의 모델을 평가함

# 11. KPF BERT Use Case (향후 추가 예정)

## 11.1. 클러스터링, 카테고리 분류 작업에 활용 검토중
## 11.2. 뉴스 기사 본문 요약 작업에 활용 검토중

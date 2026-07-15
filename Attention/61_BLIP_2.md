# 61. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

## 논문 정보

- 저자: Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi
- 소속: Salesforce Research
- 공개: arXiv:2301.12597v3, 2023년 6월 15일
- 원본 파일: `61_BLIP_2.pdf`
- 논문 링크: https://arxiv.org/abs/2301.12597
- 공개 코드: https://github.com/salesforce/LAVIS/tree/main/projects/blip2

## 한눈에 보는 요약

BLIP-2는 이미 잘 학습된 image encoder와 large language model을 모두 고정하고, 두 모델 사이에 Querying Transformer, 즉 Q-Former를 두어 시각 정보를 언어 모델이 해석할 수 있는 32개의 soft visual prompt로 압축한다. Q-Former는 먼저 frozen image encoder와 함께 시각-언어 표현을 정렬하고, 다음으로 frozen LLM 앞에 연결되어 이미지 조건부 생성을 학습한다.

핵심은 다음과 같다.

1. image encoder와 LLM을 처음부터 end-to-end로 다시 학습하지 않는다.
2. 32개의 learned query가 수백 개의 vision token에서 text와 관련 있는 정보를 뽑는다.
3. 1단계에서 ITC, ITG, ITM 세 objective와 서로 다른 attention mask를 사용해 query가 text-relevant representation을 학습한다.
4. 2단계에서 Q-Former 출력 $[B,32,768]$을 LLM embedding dimension으로 선형 projection해 soft prompt로 붙인다.
5. 최고 zero-shot VQAv2 test-dev 65.0은 Flamingo80B의 56.3보다 8.7 point 높고, 논문은 vision-language pretraining의 trainable parameter가 54배 적다고 보고한다.
6. frozen은 training gradient와 optimizer 비용을 줄일 뿐, deployment에서 3.1B에서 12.1B total parameters를 로드하고 실행해야 한다는 사실을 바꾸지 않는다.

BLIP-2는 CLIP처럼 두 embedding을 비교하는 retrieval model을 넘어, image를 LLM이 읽을 수 있는 token prefix로 바꾸는 초기 modular VLM의 대표 구조다.

## 왜 두 개의 frozen model을 연결하는가

대규모 vision-language pretraining은 image encoder, multimodal fusion, language model을 함께 업데이트해 비용이 매우 크다. 동시에 vision과 NLP 분야에는 이미 강한 unimodal pretrained model이 존재한다. BLIP-2는 이들을 재사용해 다음 두 효과를 노린다.

- compute 절감: frozen parameter에는 gradient와 optimizer state가 필요 없다.
- catastrophic forgetting 완화: LLM의 언어 생성과 instruction-following 능력을 multimodal 학습 중 훼손하지 않는다.

그러나 LLM은 pretraining에서 image를 본 적이 없다. 단순히 vision feature를 projection해 language modeling loss만 적용하면 modality gap을 충분히 메우기 어렵다. BLIP-2는 이 문제를 Q-Former의 별도 representation learning stage로 먼저 해결한다.

## 전체 구조

```text
image [B, 3, H, W]
  -> frozen ViT-L/14 또는 ViT-g/14
  -> vision features V [B, Nv, Dv]

learned queries Q [32, 768]
  -> batch에 expand [B, 32, 768]
  -> Q-Former self-attention + vision cross-attention
  -> query outputs Z [B, 32, 768]

Z
  -> trainable fully-connected projection
  -> visual soft prompts P [B, 32, Dllm]
  -> frozen OPT 또는 FlanT5에 text token과 함께 입력
  -> autoregressive output text
```

Q-Former 출력 token 수는 image resolution과 무관하게 32로 고정된다. 이 fixed bottleneck이 BLIP-2의 효율과 정보 손실 가능성을 동시에 만든다.

## Q-Former architecture

Q-Former는 두 transformer submodule로 설명되지만 self-attention layer를 공유한다.

- image transformer: learned query가 서로 self-attention하고 frozen vision feature에 cross-attention한다.
- text transformer: text encoder 또는 text decoder로 동작한다.
- cross-attention: every other transformer block에 삽입한다.
- initialization: self-attention과 FFN 등은 BERT-base weight로 초기화하고 새 cross-attention은 random initialization한다.
- 총 parameter: 188M, learned query도 model parameter에 포함된다.

논문은 32개 query, query dimension 768을 사용한다.

$$
Q\in\mathbb{R}^{32\times768},
\qquad
Z\in\mathbb{R}^{B\times32\times768}.
$$

ViT-L/14의 frozen feature는 예시로 $[B,257,1024]$다. 따라서 query token은 image token 257개를 32개로 약 8.03배 줄인다. hidden element 수 기준으로는 $257\times1024$에서 $32\times768$로 약 10.71배 작아진다.

### Batch=1 activation payload 계산

FP16 tensor payload만 계산하면 다음과 같다.

- ViT-L feature: $257\times1024\times2$ bytes = 약 514 KiB
- Q-Former output: $32\times768\times2$ bytes = 48 KiB
- 압축 비율: payload 기준 약 10.7배

cross-attention에서 BERT-base의 12 heads를 사용한다고 보면 query-to-image score는 layer당 $12\times32\times257=98,688$ elements, FP16 약 193 KiB다. 12 heads는 BERT-base initialization에서의 reviewer inference이며, 논문 본문이 별도 head table을 제공한 것은 아니다. 실제 peak에는 QKV projection, residual, FFN, vision encoder workspace가 더해진다.

## 1단계: frozen image encoder에서 representation bootstrap

1단계에는 frozen image encoder와 trainable Q-Former가 있다. 같은 image-text pair에 ITC, ITG, ITM 세 objective를 공동 적용하되, query와 text 사이 attention mask를 task별로 바꾼다.

### Image-Text Contrastive Learning, ITC

ITC는 image와 text의 global alignment를 학습한다. text representation $t$는 text transformer의 `[CLS]` output이다. 32개 query output 각각과 $t$의 similarity를 구한 뒤 가장 높은 값을 image-text similarity로 사용한다.

$$
s(I,T)=\max_{m\in\lbrace1,\ldots,32\rbrace}
\mathrm{sim}(Z_m,t).
$$

이후 batch 내 positive와 negatives에 contrastive loss를 적용한다. frozen image encoder 덕분에 GPU당 더 많은 sample을 넣을 수 있어 BLIP의 momentum queue 대신 in-batch negatives를 사용한다.

ITC에서는 uni-modal self-attention mask를 쓴다.

- query는 query끼리만 본다.
- text는 text끼리만 본다.
- query와 text는 서로 보지 못한다.

따라서 text 정보가 query에 미리 새어 들어간 상태에서 similarity를 맞추는 shortcut을 막는다.

### Image-Grounded Text Generation, ITG

ITG는 image를 조건으로 caption을 생성한다. frozen image feature는 text token과 직접 연결되지 않으므로, 생성에 필요한 정보는 반드시 query가 먼저 추출해 text stream으로 전달해야 한다.

multimodal causal self-attention mask는 다음 규칙을 갖는다.

- query는 모든 query를 볼 수 있지만 text token은 볼 수 없다.
- text token $t_i$는 모든 query와 이전 text token $t_{\lt i}$만 볼 수 있다.
- 첫 text token은 `[CLS]` 대신 decoding을 알리는 `[DEC]`를 사용한다.

표준 causal LM과 달리 32개 query prefix는 모든 generation step에서 보이는 visual context다.

### Image-Text Matching, ITM

ITM은 image-text pair가 matched인지 unmatched인지 예측하는 binary classification이다. bi-directional self-attention mask를 사용해 모든 query와 text가 서로 상호작용한다. 각 query output에 2-class linear classifier를 적용하고 32개 logit을 평균해 pair score를 만든다. informative negative를 위해 hard negative mining을 사용한다.

ITC가 coarse embedding alignment라면 ITM은 candidate pair의 fine-grained fusion score를 학습한다. retrieval에서 ITC로 후보를 빠르게 고른 뒤 ITM으로 재정렬하는 이유가 여기에 있다.

### 세 attention mask 비교

| Objective | Query가 text를 봄 | Text가 query를 봄 | Text-to-text | 목적 |
|---|---|---|---|---|
| ITC | 아니오 | 아니오 | 양방향 encoder | 독립 image/text representation 정렬 |
| ITG | 아니오 | 예 | causal | image-conditioned text 생성 |
| ITM | 예 | 예 | 양방향 | fine-grained pair matching |

같은 Q-Former parameter가 mask만 바꾸어 encoder, decoder, fusion encoder 역할을 모두 수행한다.

## 2단계: frozen LLM에서 generation bootstrap

1단계가 끝난 Q-Former와 frozen image encoder를 frozen LLM에 연결한다. Q-Former output을 LLM input embedding dimension으로 projection한다.

$$
P=ZW_p,
\qquad
Z\in\mathbb{R}^{B\times32\times768},
\quad
W_p\in\mathbb{R}^{768\times D_{LLM}},
\quad
P\in\mathbb{R}^{B\times32\times D_{LLM}}.
$$

$P$는 discrete word가 아니라 trainable visual soft prompts다. LLM weight는 고정되어 있으므로 Q-Former와 FC projection이 visual information을 LLM이 이미 이해하는 embedding pattern으로 바꾸어야 한다.

### Decoder-only OPT

OPT에는 32 visual prompts 뒤에 text embedding을 붙인다. frozen OPT가 visual prefix에 조건부로 caption을 autoregressive generation하도록 language modeling loss를 적용한다.

```text
[visual_1 ... visual_32] [text_1 ... text_L]
              -> frozen OPT decoder -> next-token loss
```

### Encoder-decoder FlanT5

text를 prefix와 suffix로 나눈다. encoder에는 visual prompts와 prefix text를 넣고, decoder는 suffix를 생성한다.

```text
encoder input: [visual_1 ... visual_32] [prefix text]
decoder target: [suffix text]
```

instruction-tuned FlanT5가 unsupervised OPT보다 VQA에서 강한 결과를 보인 것은 Q-Former가 연결하더라도 base LLM의 instruction-following 성질이 그대로 중요함을 보여준다.

## 왜 2단계가 모두 필요한가

generation loss만으로 Q-Former를 처음부터 LLM에 연결하는 것은 Frozen이나 Flamingo의 adapter 학습과 유사하다. Figure 5에서 1단계 representation learning을 제거하면 OPT와 FlanT5 모두 zero-shot VQAv2가 크게 낮아진다. 특히 OPT는 학습이 진행될수록 성능이 급격히 악화되어 저자들이 catastrophic forgetting 또는 alignment failure로 해석한다.

정확히는 LLM parameter가 frozen이므로 LLM weight 자체가 잊는 것은 아니다. visual adapter가 잘못된 soft prompt 분포를 학습해 LLM의 기존 능력을 유효하게 끌어내지 못하는 현상으로 이해하는 편이 안전하다. 1단계는 query를 text-relevant bottleneck으로 먼저 정렬해 2단계의 탐색 공간을 줄인다.

## 데이터 구성

총 129M images를 사용한다.

- COCO
- Visual Genome
- Conceptual Captions 3M
- Conceptual Captions 12M
- SBU Captions
- LAION400M에서 115M images

web caption 품질을 높이기 위해 BLIP의 CapFilt를 사용한다. BLIPlarge caption model로 image당 10개 caption을 생성하고, 원본 web caption까지 CLIP ViT-L/14 similarity로 rank한다. 상위 2개를 유지하고 pretraining step마다 하나를 random sample한다.

따라서 BLIP-2의 결과는 Q-Former 구조뿐 아니라 synthetic caption filtering의 영향도 포함한다. 원본 129M pair 그대로 학습한 결과와의 독립 ablation은 논문에 제공되지 않는다.

## 사용한 frozen backbones

### Image encoder

- CLIP ViT-L/14
- EVA-CLIP ViT-g/14

ViT의 마지막 layer 대신 second-last layer output을 사용했을 때 조금 더 좋았다고 보고한다. pretraining image resolution은 224×224다.

### Language model

- decoder-only: OPT 2.7B, OPT 6.7B
- encoder-decoder: FlanT5 XL, FlanT5 XXL

Image encoder나 LLM을 더 강하게 바꾸면 같은 framework에서 VQA가 향상됐다. 그러나 stronger backbone은 total parameter, weight memory, inference latency도 증가시킨다.

## Pretraining recipe

### Stage 1

- 250k steps
- batch size 2,320 for ViT-L, 1,680 for ViT-g
- frozen image encoder, trainable Q-Former
- ITC + ITG + ITM

### Stage 2

- 80k steps
- batch size 1,920 for OPT, 1,520 for FlanT5
- frozen image encoder와 frozen LLM
- trainable Q-Former와 FC projection

공통 설정은 AdamW, weight decay 0.05, peak learning rate $10^{-4}$, 2k linear warm-up, cosine decay다. 논문은 `β1=0.9, β1=0.98`이라고 적었는데 두 번째는 문맥상 $\beta_2=0.98$의 오타로 보인다. 2단계 minimum LR은 $5\times10^{-5}$다.

frozen ViT와 OPT는 FP16, FlanT5는 BF16으로 실행했다. 저자들은 FP32 대비 성능 하락을 관찰하지 않았다고 보고한다. 가장 큰 ViT-g + FlanT5-XXL은 16×A100 40GB 한 대 구성에서 1단계 6일 미만, 2단계 3일 미만이 걸렸다.

## Trainable parameter와 total parameter를 구분하기

논문에는 188M, 103M에서 108M, 1.1B에서 1.2B라는 여러 trainable parameter 수가 등장한다. stage와 downstream 설정이 다르기 때문이다.

- Stage 1 overview: full Q-Former 188M trainable
- Stage 2 zero-shot VQA table: Q-Former의 generation 경로와 projection 약 103M에서 108M trainable
- Caption/VQA fine-tuning: Q-Former와 image encoder를 업데이트하므로 약 1.1B에서 1.2B trainable
- LLM: pretraining과 downstream fine-tuning 모두 frozen

특히 fine-tuning section은 image encoder까지 업데이트한다고 명시한다. “BLIP-2는 항상 image encoder를 frozen으로 둔다”는 설명은 틀리다. frozen은 두 pretraining stage의 핵심 조건이고 downstream caption/VQA/retrieval에서는 task recipe에 따라 image encoder를 푼다.

## 주요 zero-shot 결과

### Visual Question Answering

| 모델 | Trainable | Total | VQAv2 val / test-dev | OK-VQA | GQA |
|---|---:|---:|---:|---:|---:|
| ViT-L + OPT 2.7B | 104M | 3.1B | 50.1 / 49.7 | 30.2 | 33.9 |
| ViT-g + OPT 2.7B | 107M | 3.8B | 53.5 / 52.3 | 31.7 | 34.6 |
| ViT-g + OPT 6.7B | 108M | 7.8B | 54.3 / 52.6 | 36.4 | 36.4 |
| ViT-L + FlanT5 XL | 103M | 3.4B | 62.6 / 62.3 | 39.4 | 44.4 |
| ViT-g + FlanT5 XL | 107M | 4.1B | 63.1 / 63.0 | 40.7 | 44.2 |
| ViT-g + FlanT5 XXL | 108M | 12.1B | 65.2 / 65.0 | 45.9 | 44.7 |

Flamingo80B은 VQAv2 test-dev 56.3, OK-VQA 50.6이다. BLIP-2 최고 모델은 VQAv2에서 8.7 point 높지만 OK-VQA에서는 4.7 point 낮다. 저자들은 Flamingo의 70B Chinchilla가 더 많은 open-world knowledge를 가진 영향으로 해석한다.

zero-shot VQA generation은 OPT에 `Question: {} Answer:`, FlanT5에 `Question: {} Short answer:` prompt를 사용한다. beam width 5, length penalty -1로 짧은 답을 유도한다. prompt와 decoding 설정까지 고정해야 수치를 재현할 수 있다.

### Captioning과 retrieval overview

Table 1의 최고 BLIP-2는 다음을 보고한다.

- NoCaps zero-shot CIDEr 121.6, SPICE 15.8
- Flickr30K zero-shot text retrieval R@1 97.6
- Flickr30K zero-shot image retrieval R@1 89.7

NoCaps 상세에서 ViT-g + FlanT5 XL은 in-domain/near-domain/out-domain CIDEr 123.7/120.2/124.8, overall 121.6이다. out-domain이 낮아지지 않은 점이 language-relevant query의 전이 성능을 뒷받침한다.

## Downstream fine-tuning

### Image captioning

COCO에서 prompt `a photo of`로 caption을 생성한다. LLM은 frozen으로 유지하지만 Q-Former와 image encoder를 업데이트한다. ViT-g image resolution은 364, 5 epochs, batch 256, LR $10^{-5}$, beam size 5다.

ViT-g + OPT 2.7B은 NoCaps overall CIDEr 119.7, COCO CIDEr 145.8을 기록한다. FlanT5 XL은 NoCaps 121.6으로 가장 높지만 COCO CIDEr 144.5로 OPT 2.7B보다 조금 낮다. backbone 크기가 모든 metric을 단조롭게 높이는 것은 아니다.

### VQA

LLM은 frozen, Q-Former와 image encoder는 trainable이다. question token을 LLM뿐 아니라 Q-Former에도 넣어 learned query가 질문과 관련된 image region에 집중하게 한다. image resolution 490, 5 epochs, batch 128, LR $10^{-5}$다.

ViT-g + OPT 6.7B은 VQAv2 test-dev/test-std 82.19/82.30으로 open-ended generation 모델 중 당시 최고 수준이다. trainable parameter는 약 1.2B이므로 pretraining의 108M과 혼동하면 안 된다.

### Retrieval

generation LLM은 필요하지 않다. 1단계 model을 가져와 image encoder와 Q-Former를 COCO에서 ITC+ITM+ITG로 fine-tune한다. inference는 먼저 ITC similarity로 $k=128$ candidates를 고르고 ITM pair score로 rerank한다.

| Image encoder | Flickr I→T / T→I R@1 | COCO I→T / T→I R@1 |
|---|---:|---:|
| ViT-L | 96.9 / 88.6 | 83.5 / 66.3 |
| ViT-g | 97.6 / 89.7 | 85.4 / 68.3 |

ITC+ITM만 사용하면 COCO I→T/T→I R@1이 84.5/67.2이고 ITG를 추가하면 85.4/68.3으로 각각 0.9/1.1 point 오른다. generation objective가 query에 language-relevant detail을 더 담게 해 retrieval도 개선한 ablation이다.

## Visual token bottleneck과 LLM 비용

32개 query는 257 vision tokens를 LLM에 그대로 넣는 것보다 prefill token 수를 약 8배 줄인다. text prompt가 32 tokens라고 가정한 reviewer 계산은 다음과 같다.

- 압축된 prefix: $32+32=64$ tokens, self-attention score $64^2=4,096$
- raw vision prefix: $257+32=289$ tokens, score $289^2=83,521$
- naive score 원소 수 비율: 약 20.4배

실제 OPT/FlanT5 attention 구조와 KV reuse에 따라 wall-clock 비율은 달라진다. 논문은 raw 257-token LLM 입력과 32-query 입력의 latency ablation을 보고하지 않는다.

decoder-only LLM에서 visual token 32개가 만드는 KV cache의 표준적인 하한은 layer당 대략 다음과 같다.

$$
M_{KV,visual}\approx2\times32\times D_{LLM}\times s,
$$

여기서 첫 2는 key와 value, $s$는 element byte 수다. 예를 들어 $D_{LLM}=4096$, FP16이면 layer당 512 KiB다. 32 layers라면 visual prefix만 약 16 MiB다. 이는 이해를 위한 가정 기반 계산이며 BLIP-2 논문이 측정한 peak memory가 아니다.

## On-device memory 관점

frozen parameter도 inference 때는 weight memory를 차지한다. Table 2 total parameter를 단순 dtype payload로 환산한 하한은 다음과 같다.

| Total params | FP16/BF16 | INT8 | W4 |
|---:|---:|---:|---:|
| 3.1B | 약 5.77 GiB | 약 2.89 GiB | 약 1.44 GiB |
| 3.8B | 약 7.08 GiB | 약 3.54 GiB | 약 1.77 GiB |
| 7.8B | 약 14.53 GiB | 약 7.26 GiB | 약 3.63 GiB |
| 12.1B | 약 22.54 GiB | 약 11.27 GiB | 약 5.63 GiB |

W4는 정확히 0.5 byte/weight만 계산한 이론적 하한이다. scale, zero-point, packing, embedding, layernorm, KV cache, activations, runtime workspace가 추가된다. 논문은 quantization을 실험하지 않았으므로 정확도 보존을 가정할 수 없다.

Q-Former 188M 자체도 FP16 weight만 약 359 MiB다. MobileCLIP-S0 전체 53.8M보다 훨씬 크다. BLIP-2의 “lightweight”는 3B에서 12B LLM 및 약 1B vision model과 비교한 상대 표현이지, 모바일에서 작은 block이라는 절대 표현이 아니다.

## Latency와 TTFT를 어떻게 분해해야 하나

BLIP-2 inference의 first-token latency는 다음 합이다.

```text
TTFT = image preprocessing
     + frozen vision encoder
     + Q-Former cross-attention
     + 32 query projection
     + LLM prefill
     + first-token sampling
```

이후 tokens/s는 대부분 frozen LLM의 autoregressive decode와 memory bandwidth에 의해 결정된다. image encoder를 한 번만 실행해도 LLM weight를 token마다 읽는 비용이 남는다.

같은 image에 연속 질문하는 경우 다음 cache가 가능하다.

- image encoder output $V$
- question-independent Q-Former visual queries $Z$
- projected 32 visual prompts
- decoder-only 구조라면 visual prefix의 LLM KV cache

다만 VQA fine-tuning은 question을 Q-Former에도 넣으므로 $Z$가 질문별로 달라져 Q-Former cache를 그대로 재사용할 수 없다. image feature $V$까지만 안전하게 cache할 수 있다.

## 논문이 보고하지 않은 온디바이스 항목

- smartphone CPU/GPU/NPU latency
- vision encoder, Q-Former, projection, LLM prefill breakdown
- p50/p95, cold start, model load
- peak memory와 KV cache memory
- TTFT와 tokens/s
- FP16, INT8, W4A16 비교
- power, temperature, sustained throttling

따라서 54배 fewer trainable parameters를 54배 빠른 inference로 해석하면 안 된다. total parameter와 autoregressive LLM이 실제 deployment 비용을 결정한다.

## 실패 사례와 한계

### In-context learning 부재

few-shot VQA example을 prompt에 넣어도 성능이 향상되지 않았다. pretraining sample이 한 sequence당 image-text pair 하나뿐이라 여러 pair 사이의 관계를 학습하지 못한 것이 원인으로 제시된다. interleaved image-text sequence가 필요한 후속 과제다.

### Hallucination과 stale knowledge

Figure 6은 Einstein image에 다른 사람의 quote를 붙이고, 캐나다 겨울 복장을 날씨와 연결하지 못하며, iPhone 14 image를 iPhone 11로 설명하는 실패를 보여준다. 원인은 부정확한 LLM knowledge, 잘못 활성화된 reasoning path, 최신 정보 부재로 나뉜다. vision grounding이 language knowledge의 사실성을 보장하지 않는다.

### Fixed 32-query bottleneck

32 tokens는 효율적이지만 dense OCR, 작은 객체, 많은 객체, spatial relation을 모두 보존하기에는 부족할 수 있다. 논문은 query 수에 대한 latency-accuracy ablation이나 작은 객체 전용 benchmark를 제공하지 않는다.

### Frozen backbone의 오류 상속

frozen vision model의 시각적 blind spot과 LLM의 bias, offensive language, private information leakage를 그대로 물려받는다. adapter만 학습하면 base model 문제를 수정할 자유도도 제한된다.

### Qualitative instruction examples

Figure 4의 visual conversation, story, reasoning은 흥미롭지만 선별된 예시다. instruction-following 정확도, hallucination rate, calibration을 계량하는 comprehensive benchmark는 없다.

### 공정한 parameter 비교

abstract의 54배는 trainable parameter 비교다. Flamingo80B 총 80B와 BLIP-2 total 12.1B를 비교하면 total size 비율은 약 6.6배다. BLIP-2가 훨씬 효율적인 것은 맞지만 비교 축을 trainable과 total로 분리해야 한다.

## 구현 체크리스트

### Architecture

- [ ] image encoder와 LLM parameter의 `requires_grad=False`를 stage별로 확인했다.
- [ ] Q-Former query 수 32, hidden size 768을 사용했다.
- [ ] cross-attention을 every other block에 삽입했다.
- [ ] ViT second-last layer feature를 사용했다.
- [ ] Q-Former 출력 projection이 LLM embedding dimension과 일치한다.

### Stage 1 masks와 loss

- [ ] ITC에서 query와 text 사이 attention을 완전히 차단했다.
- [ ] ITC similarity를 32 query 중 maximum으로 집계했다.
- [ ] ITG에서 query는 text를 못 보고 text는 query와 과거 text만 보게 했다.
- [ ] ITM에서 query-text 양방향 attention과 hard negatives를 사용했다.
- [ ] ITM의 32 query logits를 평균했다.

### Stage 2와 generation

- [ ] OPT는 visual prompt를 text 앞에 prepend했다.
- [ ] FlanT5는 prefix/suffix split과 encoder-decoder target을 올바르게 구성했다.
- [ ] prompt 문자열, beam size 5, length penalty -1을 평가 기록에 남겼다.
- [ ] trainable 188M/108M과 total 3.1B에서 12.1B를 분리해 보고했다.

### On-device

- [ ] vision, Q-Former, projector, LLM prefill, decode를 각각 profile했다.
- [ ] batch=1 TTFT와 tokens/s를 모두 측정했다.
- [ ] visual token 32개와 query 수 축소 ablation을 수행했다.
- [ ] FP16 vision, INT8 Q-Former, W4A16 LLM 조합을 비교했다.
- [ ] peak memory에 weight, activation, KV cache, workspace를 포함했다.
- [ ] multi-turn에서 image feature와 visual-prefix KV cache hit rate를 측정했다.
- [ ] p50/p95, 전력, 온도, throttling을 기록했다.

## 온디바이스용 축소 실험 제안

원 논문의 12.1B 최고 모델을 그대로 모바일 목표로 삼기보다 module별로 줄이는 편이 현실적이다.

1. vision encoder를 MobileCLIP/efficient ViT로 교체하고 frozen 상태에서 Q-Former만 재학습한다.
2. query 수를 32, 16, 8로 줄여 VQA, OCR, caption 품질과 TTFT를 비교한다.
3. Q-Former 188M을 layer pruning 또는 distillation으로 축소한다.
4. LLM은 1B에서 3B급으로 바꾸고 W4A16을 적용한다.
5. 동일 image의 multi-turn 대화에서는 image feature와 prefix cache를 유지한다.

평가 표에는 VQAv2/자체 VQA 정확도뿐 아니라 작은 글자, 작은 객체, hallucination, TTFT, tokens/s, peak memory를 함께 넣어야 한다. query bottleneck의 이득이 어떤 시각 정보 손실을 만드는지가 핵심 ablation이다.

## 로드맵에서의 위치

CLIP과 MobileCLIP이 image-text embedding을 정렬하는 dual encoder라면 BLIP-2는 그 다음 단계인 vision-to-LLM bridge를 보여준다. 특히 다음 후속 논문을 읽기 위한 기준점이 된다.

- LLaVA: 별도 Q-Former 대신 CLIP patch feature를 단순 projection하고 visual instruction tuning을 수행한다.
- TinyLLaVA/MobileVLM: vision encoder, projector, LLM을 더 작게 만들고 deployment를 겨냥한다.
- FastVLM: 고해상도 vision encoder latency와 LLM으로 보내는 visual token 수를 함께 줄인다.

BLIP-2에서 가져가야 할 실험 질문은 `frozen인가`보다 `LLM에 몇 개의 어떤 visual token을 전달하는가`다. 32-query bottleneck은 이후 visual token compression 연구의 강한 baseline이다.

## 최종 평가

BLIP-2는 pretrained vision model과 LLM을 버리지 않고 작은 bridge만 학습해 강한 VLM을 만드는 modular recipe를 정립했다. ITC, ITG, ITM으로 query를 먼저 언어 관련 시각 정보에 정렬한 뒤 generation stage로 넘어가는 2단계 설계는 단순 language modeling adapter보다 안정적이었다. 또한 32 fixed queries는 LLM visual prefix 비용을 명확히 제한한다.

그러나 188M Q-Former와 최대 12.1B total model은 모바일 관점에서 작지 않다. frozen parameter는 optimizer memory를 줄이지만 inference weight, TTFT, decode bandwidth를 없애지 않는다. 이 논문을 온디바이스 연구로 이어갈 때는 `vision encoder latency + Q-Former latency + visual token 수 + LLM prefill + tokens/s + peak memory`를 분리하고, 32-query bottleneck의 정확도 손실을 작은 객체와 OCR까지 포함해 검증해야 한다.

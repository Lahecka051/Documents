# 63. TinyLLaVA: A Framework of Small-scale Large Multimodal Models

## 논문 정보

- 원본 파일: `63_TinyLLaVA.pdf`
- 제목: **TinyLLaVA: A Framework of Small-scale Large Multimodal Models**
- 저자: Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jie Luo, Xien Liu, Ji Wu, Lei Huang
- 발표: arXiv 2024
- 공식 링크: [arXiv:2402.14289](https://arxiv.org/abs/2402.14289)
- 태스크: small-scale large multimodal model, visual question answering, multimodal instruction following
- 핵심 키워드: small LLM, CLIP, SigLIP, MLP connector, feature alignment, visual instruction tuning, data quality, training recipe

## 한눈에 보는 요약

TinyLLaVA는 새로운 단일 architecture라기보다 작은 VLM의 설계 공간을 같은 틀에서 비교하는 **framework와 empirical study**다. 모델을 vision encoder `V`, connector `P`, small language model `F`로 분해하고 다음 선택을 교차 비교한다.

- LLM: TinyLlama 1.1B, StableLM-2 1.6B, Phi-2 2.7B
- Vision encoder: CLIP-L/14 336, SigLIP-SO400M/14 384
- Connector: 2-layer MLP, resampler
- Data: LLaVA-1.5 mixture, ShareGPT4V mixture
- Recipe: connector-only alignment을 쓰는 `base`, vision/LLM까지 더 많이 푸는 `share`

```text
image X
  -> vision encoder V_phi
  -> M visual patch tokens
  -> connector P_varphi
  -> LLM embedding dimension
  -> concatenate/interleave with text embeddings
  -> small autoregressive LLM F_theta
  -> assistant response
```

논문의 가장 중요한 결론은 "작은 LLM도 큰 LLM을 이긴다"라는 단순 문장보다 정교하다.

1. 동일 recipe에서는 대체로 더 큰 Phi-2가 더 강하다.
2. SigLIP variant는 CLIP variant보다 전반적으로 좋지만 더 높은 resolution과 더 많은 visual token을 함께 사용하므로 encoder 종류만의 순수 ablation은 아니다.
3. 더 크고 다양한 data와 더 많은 trainable parameter는 평균 성능을 높일 수 있지만 POPE hallucination 성능은 오히려 떨어질 수 있다.
4. 최종 TinyLLaVA-share-Sig-Phi 3.1B는 여러 benchmark에서 LLaVA-1.5 7B와 비슷하거나 우수하지만 모든 지표에서 이기는 것은 아니다.

논문은 parameter efficiency를 보여 주지만 실제 phone latency, TTFT, tokens/s, peak memory, 전력, quantization 결과는 보고하지 않는다. 따라서 3.1B라는 숫자만으로 온디바이스 모델이라고 결론내릴 수 없다.

## 연구 배경과 문제의식

LLaVA류 VLM은 pretrained vision encoder의 patch feature를 connector로 LLM embedding 공간에 맞춘 뒤 autoregressive text generation을 수행한다. 초창기 공개 모델은 7B 이상 LLM을 주로 사용했다. 이 구조에서 성능이 낮을 때 원인이 LLM 크기인지, vision encoder인지, connector인지, data인지, freeze recipe인지 분리하기 어렵다.

TinyLLaVA는 작은 LLM 범위에서 구성 요소를 체계적으로 바꾸어 다음 질문을 묻는다.

- LLM parameter를 줄이면 어떤 benchmark가 먼저 무너지는가?
- CLIP 대신 SigLIP을 쓰면 작은 LLM의 시각 이해가 얼마나 좋아지는가?
- 단순 MLP와 token-resampling connector 중 무엇이 유리한가?
- 더 많은 caption과 instruction data가 작은 모델에도 항상 좋은가?
- Vision encoder와 LLM을 얼마나 동결해야 하는가?

온디바이스 연구에서 특히 중요한 관점은 parameter 수만이 아니다. Vision token 수는 LLM prefill, KV cache, TTFT를 바꾸고, freeze 범위는 학습 memory와 배포 architecture를 바꾼다.

## Unified architecture

### Small-scale LLM

LLM `F_theta`는 `d`차원 embedding sequence를 입력받고 다음 token을 autoregressive하게 예측한다.

```math
F_\theta:\{h_i\}_{i=0}^{N-1}
\mapsto p(y_N\mid h_0,\ldots,h_{N-1})
```

Tokenizer와 text embedding은 LLM에 속한다. 논문은 다음 세 모델을 사용한다.

| LLM | Parameter |
| --- | ---: |
| TinyLlama | 1.1B |
| StableLM-2 | 1.6B |
| Phi-2 | 2.7B |

이 비교는 parameter scaling만의 통제 실험은 아니다. 각 LLM은 pretraining corpus, tokenizer, architecture와 language capability가 다르다. "Phi-2가 강하므로 2.7B가 최적"이라는 보편적 결론보다 **해당 checkpoint 조합에서 Phi-2가 가장 강했다**고 읽어야 한다.

### Vision encoder

Image `X`를 `M`개의 patch feature로 바꾼다.

```math
V_\phi(X)=\{v_j\in\mathbb R^{d_x}\}_{j=1}^{M}
```

| Vision encoder | 입력 | Visual tokens | Parameter |
| --- | ---: | ---: | ---: |
| CLIP ViT-L/14 | 336 | 576 | 약 0.3B |
| SigLIP SO400M/14 | 384 | 729 | 약 0.4B |

논문 자체가 SigLIP 비교의 confound를 인정한다. SigLIP은 objective와 pretraining뿐 아니라 입력 resolution, parameter, token 수가 모두 다르다. Fine-grained OCR나 작은 object 성능이 좋아질 수 있지만 LLM 비용도 함께 오른다.

### Connector

Connector는 visual dimension `d_x`를 LLM text dimension `d`로 사상한다.

```math
h_j=P_\varphi(v_j),\qquad
P_\varphi:\mathbb R^{d_x}\rightarrow\mathbb R^d
```

기본 connector는 GELU를 포함한 2-layer MLP다.

```math
P_\varphi(v)=W_2\,\mathrm{GELU}(W_1v+b_1)+b_2
```

Connector는 token 수를 줄이지 않는다. CLIP의 576개, SigLIP의 729개 token이 그대로 LLM prefix로 들어간다. 논문은 resampler도 시험하지만 TinyLlama+CLIP의 비교에서 MLP가 전반적으로 더 좋았다. 이는 특정 구현과 parameter budget의 결과이며 resampling 자체가 원리적으로 나쁘다는 증거는 아니다.

## Training objective

### Stage 1: feature alignment pretraining

Image-caption pair `(X,Y_a)`에서 image를 조건으로 caption token likelihood를 최대화한다.

```math
p(Y_a\mid X)=\prod_{i=1}^{N_a}
F_\theta\left(y_i\mid y_{<i}, P_\varphi(V_\phi(X))\right)
```

```math
\max_{\varphi,\theta',\phi'}
\sum_{i=1}^{N_a}\log
F_\theta\left(y_i\mid y_{<i},P_\varphi(V_\phi(X))\right)
```

`theta'`, `phi'`는 전체 LLM/vision parameter 중 recipe가 학습 가능하게 연 subset이다. Framework의 핵심은 connector-only로 고정하지 않고 작은 LLM의 alignment 능력에 따라 일부 pretrained module도 풀 수 있게 한 것이다.

### Stage 2: supervised fine-tuning

Multi-turn conversation은 human question과 assistant response를 포함한다. Loss는 assistant response token에만 적용한다.

```math
\max_{\varphi,\theta',\phi'}
\sum_{i=1}^{N}\mathbf 1[y_i\in\mathcal A]
\log F_\theta\left(y_i\mid y_{<i},P_\varphi(V_\phi(X))\right)
```

Human prompt token은 conditioning context지만 prediction target으로 학습하지 않는다. 구현에서 label tensor의 user/system/image-placeholder 위치를 `ignore_index`로 mask해야 한다.

## Base와 share recipe

### Base recipe

LLaVA-1.5 방식을 따른다.

- Pretraining: vision encoder와 LLM 동결, connector만 1 epoch 학습
- Pretraining LR: `1e-3`, batch 256
- SFT: vision encoder 동결, connector와 LLM 학습
- SFT LR: `2e-5`, batch 128, 1 epoch

### Share recipe

ShareGPT4V 방식을 따른다.

- Base 단계에서 학습한 connector weight로 초기화
- Pretraining: vision encoder 첫 12 layer는 동결하고 나머지 model을 학습
- Pretraining LR: `2e-5`, batch 256, 1 epoch
- SFT: base recipe와 같은 설정

Share recipe는 단순한 scheduler 변경이 아니다. Data와 초기화, vision freeze 범위, LLM trainability가 함께 달라진다. 따라서 개선을 한 요소에만 귀속하기 어렵다.

## Training data

| Data family | Pretraining | Samples | SFT | Samples |
| --- | --- | ---: | --- | ---: |
| LLaVA-1.5 | LLaVA-1.5-558k | 558k | LLaVA-1.5-mix-665k | 665k |
| ShareGPT4V | ShareGPT4V-PT-1246k | 1,246k | ShareGPT4V-mix-665k | 665k |

ShareGPT4V pretraining은 caption sample 수가 약 2.23배다. SFT sample 수는 같지만 LLaVA-1.5 mixture의 23k detailed description을 ShareGPT4V의 detailed caption으로 교체한다. 논문이 이를 data quality 효과로 논의하지만 pretraining 양과 caption source가 동시에 달라진다.

## Tensor shape와 visual-token 비용

### CLIP + Phi-2 예시

논문 표의 CLIP-L/14 336과 Phi-2 조합을 기준으로 한다. CLIP hidden을 1024, Phi-2 text hidden을 2560으로 두면 다음 흐름이다.

```text
image:                [1, 3, 336, 336]
CLIP patch features:  [1, 576, 1024]
MLP connector output: [1, 576, 2560]
text prompt 64 tokens:[1,  64, 2560]   # illustrative
LLM prefill input:    [1, 640, 2560]
```

SigLIP 조합은 다음처럼 visual prefix가 늘어난다.

```text
image:                 [1, 3, 384, 384]
SigLIP patch features: [1, 729, d_sig]
connector output:      [1, 729, 2560]
text prompt 64 tokens: [1,  64, 2560]
LLM prefill input:     [1, 793, 2560]
```

Text 64개라는 값은 논문 설정이 아니라 비교를 위한 reviewer assumption이다. 이 경우 prefill sequence 길이는 `640 -> 793`, self-attention의 quadratic 부분은 대략 다음 비율로 증가한다.

```math
\left(\frac{793}{640}\right)^2\approx1.53
```

전체 layer 비용에는 linear projection과 MLP도 있으므로 end-to-end latency가 정확히 1.53배라는 뜻은 아니다. 다만 SigLIP의 정확도 이득이 LLM prefill과 KV 비용 증가를 동반한다는 점은 분명하다.

### Connector activation

`576 x 2560` FP16 embedding은 약 `2.81 MiB`, `729 x 2560`은 약 `3.56 MiB`다. 이 tensor 자체보다 각 LLM layer의 intermediate activation과 KV cache가 더 크다.

표준 multi-head KV를 쓰는 `L=32`, hidden `H=2560`, FP16 모델을 단순 가정하면 token당 KV storage는 다음과 같다.

```math
M_{KV/token}=2\cdot L\cdot H\cdot2\ \mathrm{bytes}
=327{,}680\ \mathrm{bytes}
```

576 visual token은 약 `180 MiB`, 729 token은 약 `228 MiB`다. 이 계산은 Phi-2의 정확한 KV layout, head sharing, allocator를 검증하지 않은 설명용 upper-style estimate이며 논문 보고값이 아니다. 실제 runtime에서 model architecture와 dtype로 다시 측정해야 한다.

## Weight memory

최종 3.1B parameter를 단순 storage로 계산하면 다음과 같다.

| Format | 이상적 raw weight storage |
| --- | ---: |
| FP32 | 12.4 GB |
| FP16/BF16 | 6.2 GB |
| INT8 | 3.1 GB |
| INT4 | 1.55 GB |

실제 W4A16 배포는 scale/zero-point, alignment, unquantized embedding/norm, FP16 vision encoder와 connector, KV cache, runtime workspace가 더해진다. 따라서 3.1B 전체를 1.55 GB에서 실행할 수 있다는 뜻이 아니다. 논문은 quantized model 크기나 peak RSS를 보고하지 않는다.

## 최소 구현 의사코드

```python
def encode_image(image):
    patch = vision_encoder(image)          # [B, M, d_vision]
    visual = connector(patch)              # [B, M, d_text]
    return visual

def build_multimodal_inputs(image, input_ids, labels=None):
    visual = encode_image(image)
    text = llm.get_input_embeddings()(input_ids)
    embeds = replace_image_placeholder(text, visual)

    if labels is not None:
        labels = insert_ignore_labels_for_visual_tokens(labels, visual.size(1))
        labels = mask_user_and_system_tokens(labels, ignore_index=-100)
    return embeds, labels

def training_step(image, input_ids, labels):
    embeds, labels = build_multimodal_inputs(image, input_ids, labels)
    return llm(inputs_embeds=embeds, labels=labels).loss

@torch.inference_mode()
def generate(image, input_ids, max_new_tokens):
    embeds, _ = build_multimodal_inputs(image, input_ids)
    return llm.generate(inputs_embeds=embeds,
                        max_new_tokens=max_new_tokens,
                        use_cache=True)
```

같은 image에 여러 질문을 할 때 vision encoder와 connector output을 cache할 수 있다. 그러나 LLM prompt가 달라지면 prefill/KV를 어디까지 재사용할 수 있는지는 conversation template와 runtime의 prefix-cache 지원에 달려 있다.

## Architecture ablation

### LLM 크기

Base recipe에서 대체로 Phi-2 variant가 TinyLlama와 StableLM-2보다 강하다. 예를 들어 CLIP 조합의 SQA-I는 Phi-2 `69.6`, StableLM-2 `62.8`, TinyLlama `60.2`다. TinyLlama는 일부 POPE 지표에서 StableLM-2보다 나아 작은 parameter가 hallucination에 항상 불리한 것은 아니다.

이 결과는 language backbone의 prior가 visual reasoning에 크게 작용한다는 뜻이다. 동시에 checkpoint마다 language pretraining이 다르므로 순수한 parameter scaling law로 해석할 수는 없다.

### CLIP 대 SigLIP

Figure 5의 base-recipe 비교에서 SigLIP은 대부분의 지표를 높인다.

| LLM | 지표 | CLIP | SigLIP |
| --- | --- | ---: | ---: |
| TinyLlama | VQAv2 | 74.0 | 75.8 |
| TinyLlama | TextVQA | 45.8 | 49.1 |
| StableLM-2 | VQAv2 | 74.9 | 78.1 |
| StableLM-2 | TextVQA | 49.5 | 54.1 |
| Phi-2 | VQAv2 | 76.6 | 79.2 |
| Phi-2 | MM-Vet | 30.7 | 32.1 |

하지만 SigLIP은 384 input과 729 token, CLIP은 336 input과 576 token이다. Accuracy-latency trade-off 표 없이 "SigLIP이 더 효율적"이라고 결론낼 수 없다.

### MLP 대 resampler

TinyLlama+CLIP에서 2-layer MLP가 resampler보다 전반적으로 우수하다. VQAv2는 `74.0` 대 `70.1`, POPE는 `86.0` 대 `83.6`, MM-Vet은 `21.6` 대 `20.6`이다. Token reduction 여부와 resampler query 수, optimization 조건이 자세한 latency 비교로 이어지지 않으므로 이 결과는 connector 선택의 최종 결론보다 preliminary exploration으로 보는 것이 맞다.

## Data와 recipe ablation

더 큰 ShareGPT4V pretraining data는 StableLM-2와 Phi-2의 많은 benchmark를 개선하지만 TinyLlama에서는 SQA-I와 POPE가 악화되는 경우가 있다. 저자는 1.1B 모델의 capacity가 더 큰 data를 충분히 흡수하지 못해 knowledge degradation이나 hallucination이 생길 수 있다고 추측한다.

Share recipe는 TinyLlama의 여러 지표를 올린다. 예를 들어 base에서 share로 바꿀 때 VQAv2 `74.1 -> 75.2`, MM-Vet `23.7 -> 25.1`이다. 반면 Phi-2의 POPE는 `87.0 -> 85.3`으로 떨어지면서 다른 지표는 좋아진다. 더 많은 parameter를 푸는 것이 평균 capability와 hallucination robustness를 동시에 보장하지 않는다.

이 ablation의 중요한 교훈은 작은 VLM도 하나의 aggregate score만 최적화해서는 안 된다는 것이다. OCR, reasoning, instruction following과 hallucination을 별도 축으로 봐야 한다.

## 최종 benchmark 비교

논문의 Table 3에서 주요 행을 정리하면 다음과 같다. `*`는 해당 benchmark image가 training data에서 관찰되었음을 논문이 표시한 항목이다.

| Model | Size | Res. | VQAv2 | GQA | SQA-I | TextVQA | MM-Vet | POPE | LLaVA-W | MME | MMB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5 | 7B | 336 | 78.5* | 62.0* | 66.8 | 58.2 | 30.5 | 85.9 | 63.4 | **1510.7** | 64.3 |
| TinyLLaVA-share-C-Phi | 3.0B | 336 | 77.7* | 61.0* | **70.1** | 53.5 | 31.7 | 86.3 | 67.1 | 1437.3 | **68.3** |
| TinyLLaVA-share-Sig-Phi | 3.1B | 384 | **79.9*** | **62.0*** | 69.1 | **59.1** | **32.0** | **86.4** | **75.8** | 1464.9 | 66.9 |

TinyLLaVA 3.1B는 LLaVA-1.5 7B보다 VQAv2, TextVQA, MM-Vet, POPE, LLaVA-W에서 높고 GQA는 같다. 반면 SQA-I, MME, MMB에서는 3.0B CLIP variant 또는 LLaVA-1.5가 더 높다. "Overall better"는 유용한 요약이지만 모든 task에서 dominance를 뜻하지 않는다.

## 장점과 기여

1. **작은 VLM 설계 변수를 같은 framework로 정리했다.** Component와 recipe를 교체하기 쉽다.
2. **Parameter 수 이외의 요인을 강조했다.** Data mixture와 freeze policy가 수십억 parameter 차이만큼 중요할 수 있음을 보여 준다.
3. **강한 3.1B baseline을 제시했다.** 7B model에 근접하거나 여러 지표에서 앞선다.
4. **Hallucination trade-off를 숨기지 않는다.** Share recipe가 POPE를 낮추는 사례를 논의한다.
5. **Vision encoder와 token 수의 중요성을 드러낸다.** 후속 mobile VLM의 visual-token 최적화 질문으로 자연스럽게 이어진다.

## 한계와 비판적 관점

### 1. On-device 효율을 직접 측정하지 않는다

Parameter, TTFT, tokens/s, peak memory, latency, energy, temperature, W4A16 결과가 없다. 3.1B는 7B보다 작지만 여전히 phone deployment에 큰 모델이다.

### 2. Vision encoder ablation이 통제되지 않았다

CLIP과 SigLIP은 objective, parameter, resolution, token 수가 모두 다르다. SigLIP의 성능 이득 중 무엇이 architecture/pretraining이고 무엇이 추가 compute인지 분리할 수 없다.

### 3. Data quality와 quantity가 섞여 있다

558k 대 1,246k pretraining sample, caption generator, SFT detailed caption이 함께 바뀐다. "Quality"만의 인과 효과라고 보기 어렵다.

### 4. LLM 크기 비교도 checkpoint 비교다

TinyLlama, StableLM-2, Phi-2는 동일 architecture를 width/depth만 바꾼 family가 아니다. Parameter 수 외의 language prior가 결과에 영향을 준다.

### 5. Evaluation contamination 주의가 필요하다

Table의 `*`는 일부 benchmark image가 training mixture에 포함됨을 뜻한다. 해당 점수는 완전히 held-out generalization과 구분해야 한다.

### 6. Connector 탐색이 제한적이다

MLP와 한 resampler 구현을 비교할 뿐, query 수별 accuracy-latency, token pruning, pooling, ROI tokenization은 분석하지 않는다.

### 7. Hallucination과 capability가 충돌한다

더 많은 data와 trainable parameter가 POPE를 낮추는 사례가 있다. 단일 평균 score로 recipe를 선택하면 안전성이 악화될 수 있다.

## 자주 헷갈리는 지점

### TinyLLaVA는 항상 3.1B 모델을 뜻하는가

아니다. 여러 LLM, vision encoder, data와 recipe를 조합하는 framework와 model family다.

### 3.1B는 LLM parameter만인가

최종 이름의 size는 vision encoder와 connector를 포함한 전체 model 규모다. Phi-2 2.7B와 SigLIP 약 0.4B가 대부분을 차지한다.

### Share recipe는 weight sharing architecture인가

이 문맥의 `share`는 ShareGPT4V data/training recipe 이름이다. Vision과 language layer가 parameter를 공유한다는 뜻이 아니다.

### SigLIP이 CLIP보다 mobile-friendly한가

논문은 accuracy 우위를 보이지만 SigLIP variant가 더 큰 입력과 더 많은 visual token을 사용한다. Mobile latency 우위는 입증하지 않았다.

### Connector만 학습하는 것이 최종 recipe인가

Base pretraining에서는 connector만 학습하지만 SFT에서는 connector와 LLM을 학습한다. Share pretraining은 vision encoder 일부와 나머지 model도 더 많이 푼다.

### Visual token은 generation이 시작되면 사라지는가

Autoregressive LLM의 context로 남아 KV cache와 attention에 영향을 준다. Runtime이 별도 compression이나 cache eviction을 구현하지 않는 한 비용이 계속된다.

## 온디바이스 관점

### 실제 병목 순서

1. LLM weight storage와 memory bandwidth
2. Visual token을 포함한 LLM prefill과 TTFT
3. KV cache
4. SigLIP/CLIP vision encoder latency
5. Connector와 layout conversion
6. Autoregressive decode tokens/s

Connector parameter는 전체에서 작지만 576/729 token을 그대로 전달한다는 정책이 system cost를 크게 만든다. Mobile VLM에서는 connector FLOPs보다 **connector가 몇 token을 남기는지**가 더 중요할 수 있다.

### 권장 배포 변형

- LLM: W4A16 또는 device가 지원하는 group-wise INT4
- Vision encoder: FP16 또는 calibration을 검증한 INT8
- Connector: FP16/INT8, projection fusion 확인
- Visual feature: 같은 image의 후속 질문에서 cache
- Context: ROI나 detector mask로 image crop 후 visual token 축소
- Training: vision/LLM 동결 후 projector 또는 LoRA만 task data로 적응

### 반드시 측정할 항목

- Vision encoder latency
- Projector latency와 output token 수
- TTFT를 vision encode와 LLM prefill로 분해
- Decode tokens/s
- Model load와 cold-start
- Peak RSS와 KV cache 증가율
- p50/p95 latency
- Energy per query, 온도, 반복 질의 throttling
- CLIP 576 token 대 SigLIP 729 token의 accuracy-TTFT trade-off

## 재현 계획

### 최소 architecture ablation

Phi-2와 동일 MLP connector를 고정하고 다음을 비교한다.

1. CLIP 336, 576 token
2. SigLIP 384, 729 token
3. SigLIP feature를 576 token으로 pooling한 변형
4. Detector ROI crop으로 196~256 token까지 줄인 변형

VQAv2/TextVQA/MM-Vet/POPE와 함께 vision latency, TTFT, peak memory를 측정하면 논문의 accuracy 결과를 온디바이스 연구 질문으로 확장할 수 있다.

### Training recipe ablation

- Connector-only pretraining
- Connector + LLM LoRA
- Connector + vision last blocks LoRA
- Share-style partial unfreeze

Trainable parameter, optimizer-state memory, wall-clock, POPE를 함께 기록한다. 더 많은 parameter를 푸는 것이 hallucination을 악화하는지 확인한다.

### 공정한 실험 조건

- 같은 image preprocessing과 crop
- 같은 visual token 수를 맞춘 비교를 별도로 수행
- 같은 SFT sample과 epoch
- 같은 decoding temperature, max tokens, prompt template
- Training overlap이 있는 benchmark를 명시
- Batch 1, 동일 device, cold/warm cache 분리

## 구현 체크리스트

- [ ] Image placeholder 위치에 visual embedding을 정확히 삽입하는가?
- [ ] Visual token label을 `ignore_index`로 mask하는가?
- [ ] User/system token이 SFT loss에 들어가지 않는가?
- [ ] CLIP/SigLIP normalization과 resolution이 checkpoint와 맞는가?
- [ ] CLS token 포함 여부와 visual token 수가 일치하는가?
- [ ] Connector input/output dimension이 vision/LLM과 맞는가?
- [ ] Base/share의 freeze 범위를 parameter name으로 검증했는가?
- [ ] Training data mixture와 benchmark overlap을 기록했는가?
- [ ] TTFT에 vision encoder와 prefill 시간을 모두 포함했는가?
- [ ] KV cache와 runtime workspace를 peak memory에 포함했는가?
- [ ] W4A16에서 정확도와 tokens/s를 함께 측정했는가?
- [ ] POPE를 capability 평균과 별도로 보고했는가?
- [ ] 반복 질의에서 vision feature cache 효과를 측정했는가?

## 로드맵에서의 위치

CLIP은 image-text representation을 만들고, MobileCLIP은 retrieval encoder 자체를 경량화한다. BLIP-2는 Q-Former로 frozen vision과 frozen LLM을 연결하며 token bottleneck을 만든다. LLaVA는 단순 projector와 visual instruction tuning의 강한 baseline이다. TinyLLaVA는 이 LLaVA형 구조를 1~3B 범위로 줄이고 data와 recipe가 model size만큼 중요하다는 점을 보여 준다.

이후 MobileVLM V2와 FastVLM을 읽을 때는 다음 질문을 이어가면 된다.

- 같은 정확도에서 vision encoder latency는 얼마인가?
- LLM으로 전달되는 visual token은 몇 개인가?
- Projector가 token을 줄이는가, dimension만 바꾸는가?
- Quantization 후 TTFT와 tokens/s는 어떻게 변하는가?

## 최종 평가

TinyLLaVA의 의미는 3.1B가 7B보다 무조건 우월하다는 데 있지 않다. **작은 multimodal model의 성능은 LLM 크기뿐 아니라 vision representation, data composition, freeze policy의 곱으로 결정된다**는 실험적 증거가 핵심이다. 특히 더 큰 data와 더 많은 trainable parameter가 hallucination 지표를 악화할 수 있다는 결과는 단일 평균 score 중심의 최적화를 경계하게 한다.

온디바이스 관점에서는 아직 출발점이다. 논문은 parameter를 줄였지만 visual token, KV cache, TTFT, quantization과 device latency를 측정하지 않는다. 실용적인 후속 실험은 576/729 visual token을 그대로 받아들이지 않고 ROI, pooling, learned resampler로 줄인 뒤 W4A16 LLM에서 accuracy-TTFT-memory 곡선을 그리는 것이다.

# 64. MobileVLM V2: Faster and Stronger Baseline for Vision Language Model

## 논문 정보

- 원본 파일: `64_MobileVLM_V2.pdf`
- 제목: **MobileVLM V2: Faster and Stronger Baseline for Vision Language Model**
- 저자: Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, Chunhua Shen
- 소속: Meituan Inc., Zhejiang University, Dalian University of Technology
- 공개: arXiv preprint, 2024
- 링크: [arXiv:2402.03766](https://arxiv.org/abs/2402.03766)
- 핵심 키워드: mobile VLM, visual token compression, LDPv2, projector, frozen vision encoder, edge inference

## 한 문장 요약

MobileVLM V2는 동결한 `CLIP ViT-L/14@336`의 576개 patch token을 저비용 projector `LDPv2`에서 144개로 줄이고 작은 언어 모델과 연결함으로써, 1.7B 및 3B급 VLM의 정확도와 Jetson Orin 생성 속도를 동시에 개선한 실용적 baseline이다.

## 논문이 해결하려는 문제

일반적인 multimodal large language model은 강한 vision encoder, projector, 수십억 parameter의 LLM을 결합한다. 이 구조를 edge에 옮기면 다음 세 비용이 겹친다.

1. vision encoder가 고해상도 이미지를 처리하는 비용
2. 수백 개 visual token이 LLM prefill sequence를 늘리는 비용
3. 긴 sequence와 큰 LLM이 만드는 activation, KV cache, memory bandwidth 비용

기존 MobileVLM은 작은 LLM과 lightweight downsample projector를 사용했지만, projector의 depthwise-pointwise block이 여전히 불필요하게 무겁고 학습 데이터 및 recipe도 충분히 정리되지 않았다. MobileVLM V2의 질문은 단순하다.

> 같은 frozen vision encoder와 작은 LLM을 유지하면서 projector, 데이터, 학습 전략만 정리해 정확도와 처리량을 얼마나 더 끌어올릴 수 있는가?

논문의 답은 `LDPv2`, 더 큰 고품질 데이터 혼합, 두 단계 모두에서 projector와 LLM을 함께 학습하는 전략이다. 중요한 점은 새로운 거대 backbone보다 **LLM에 들어가는 visual sequence 길이**를 직접 줄인다는 데 있다.

## 전체 구조

```text
image: B x 3 x 336 x 336
  -> frozen CLIP ViT-L/14
  -> 24 x 24 patch grid
  -> visual feature: B x 576 x D_v
  -> LDPv2
       pointwise projection + GELU + pointwise projection
       2 x 2 average pooling
       3 x 3 depthwise PEG + residual
  -> B x 144 x D_t
  -> concatenate with text embeddings
  -> MobileLLaMA-1.4B/2.7B or Vicuna-7B
  -> autoregressive answer
```

1.7B와 3B 모델은 각각 MobileLLaMA-1.4B-Chat과 MobileLLaMA-2.7B-Chat을 언어 모델로 사용한다. 7B 모델은 Vicuna-7B를 사용한다. 모델 이름의 1.7B와 3B는 vision encoder와 projector까지 포함한 전체 규모에 가까우므로 LLM parameter 수와 혼동하면 안 된다.

vision encoder는 두 학습 단계에서 모두 동결한다. 반면 projector와 LLM은 pretraining과 multi-task fine-tuning 모두에서 업데이트한다. 동결은 학습 memory와 optimizer state를 줄이는 선택이지, inference 때 vision encoder 계산을 없애는 선택은 아니다.

## LDPv2를 식으로 보기

vision encoder 출력 feature를 `f_v`, LLM hidden dimension에 맞춘 최종 visual representation을 `H_v`라 하자. 논문 구조는 다음처럼 정리된다.

```math
f_0 = \operatorname{PW}\!\left(
  \operatorname{GELU}\!\left(\operatorname{PW}(f_v)\right)
\right)
```

```math
f_1 = \operatorname{AvgPool}_{2\times2}(f_0)
```

```math
H_v = \operatorname{DW}_{3\times3}(f_1) + f_1
```

- `PW`: 각 spatial 위치에 동일하게 적용하는 pointwise convolution, 즉 channel projection
- `AvgPool`: 2 x 2 non-overlapping average pooling
- `DW`: channel별 3 x 3 depthwise convolution
- 마지막 residual branch: pooled token에 local positional cue를 더하는 PEG 방식

일반적으로 pooling kernel과 stride가 `k`이면 token 수는 원래의 `1/k^2`가 된다. 본 논문은 `k=2`이므로 576개가 144개가 된다.

기존 LDP는 positional enhancement를 위해 `[DW, PW]` block을 사용했다. LDPv2는 그 부분을 depthwise convolution 하나와 skip connection으로 바꾼다. 1.7B 설정에서 projector parameter 수는 다음과 같다.

| Projector | Visual token | Parameter |
| --- | ---: | ---: |
| 두 PW layer | 576 | 6.30M |
| 기존 LDP | 144 | 18.94M |
| LDPv2 | 144 | 6.32M |

논문은 PEG의 3 x 3 depthwise convolution을 `2048 x 3 x 3 = 0.02M` parameter로 계산한다. 기존 positional block의 12.64M과 비교하면 positional component만 약 630배 작다. 전체 projector 관점에서는 18.94M에서 6.32M으로 약 3배 감소한다. 서로 다른 분모를 섞어 "projector 전체가 630배 작다"고 해석하면 틀린다.

## Tensor shape 추적

### Vision encoder부터 projector까지

`CLIP ViT-L/14`에 336 x 336 이미지를 넣으면 한 축의 patch 수는 다음과 같다.

```math
\frac{336}{14}=24
```

따라서 patch token은 `24 x 24 = 576`개이다. 논문의 1.7B projector에서 LLM embedding dimension은 PEG parameter 산식으로부터 2048임을 확인할 수 있다.

| 지점 | Shape, batch=1 | FP16 element 수 | 단순 payload |
| --- | ---: | ---: | ---: |
| CLIP patch feature | `576 x D_v` | `576 D_v` | `1152 D_v` bytes |
| channel projection 뒤 | `576 x 2048` | 1,179,648 | 2.25 MiB |
| 2 x 2 pooling 뒤 | `144 x 2048` | 294,912 | 0.5625 MiB |
| LLM visual prefix | `144 x 2048` | 294,912 | 0.5625 MiB |

이 표는 tensor 하나의 raw payload만 센 reviewer 계산이다. 실제 peak memory에는 convolution workspace, layer input 보존, allocator alignment, attention buffer, model weight가 추가된다. 학습 때는 backward를 위한 activation과 optimizer state도 더 필요하다.

pooling 한 번으로 projector output activation은 정확히 4분의 1이 된다. 또한 LLM에 전달되는 visual prefix가 432 token 줄어든다.

```math
\Delta N_v = 576 - 144 = 432
```

### Text와 visual token 결합

질문 text token 수를 `N_q`라 하면 LLM prefill 길이는 대략 다음과 같다.

```math
S = N_q + N_v + N_{special}
```

special token을 잠시 무시하면 LDPv2는 `S=N_q+144`, pooling이 없으면 `S=N_q+576`이다. 예를 들어 reviewer가 `N_q=64`로 놓으면 길이는 208과 640이다. dense attention score element 수의 이론적 비는 다음과 같다.

```math
\frac{640^2}{208^2} \approx 9.47
```

이는 **attention score 항만 본 상한 성격의 예시**다. 실제 LLM layer에는 projection과 MLP 비용도 있고, FlashAttention 계열 kernel은 전체 `S x S` matrix를 HBM에 저장하지 않는다. 따라서 전체 latency가 9.47배 빨라진다는 뜻이 아니다. 그래도 visual token 축소가 prefill과 TTFT에 강하게 영향을 줄 수 있다는 방향은 분명하다.

### KV cache에 미치는 영향

decoder layer의 KV dimension을 `D_KV = n_kv_heads x d_head`, layer 수를 `L`, element byte를 `b`라 하면 visual prefix 432개를 줄여 절약하는 KV cache는 다음과 같다.

```math
\Delta M_{KV}
= 2 \times L \times 432 \times D_{KV} \times b
```

앞의 2는 key와 value를 뜻한다. 표준 MHA라서 `D_KV=D_model=2048`, FP16이라서 `b=2`라고 가정한 reviewer 예시에서는 layer당 다음을 줄인다.

```math
2 \times 432 \times 2048 \times 2
= 3,538,944\ \text{bytes}
\approx 3.375\ \text{MiB/layer}
```

실제 MobileLLaMA가 GQA 또는 MQA를 사용하면 `D_KV`가 더 작으므로 이 값을 그대로 적용하면 안 된다. 논문은 layer별 KV layout과 runtime peak memory를 보고하지 않는다. 구현 시 모델 config의 `num_key_value_heads`를 읽어 다시 계산해야 한다.

## Autoregressive 학습 목적함수

vision representation을 `H_v`, 질문 representation을 `H_q`, 정답 token sequence를 `Y_a=(y_1,...,y_T)`라 하면 conditional generation은 다음과 같다.

```math
p(Y_a\mid H_v,H_q)
= \prod_{i=1}^{T}
p(y_i\mid H_v,H_q,y_{<i})
```

학습은 정답 token의 negative log likelihood를 최소화한다.

```math
\mathcal{L}_{LM}
= -\sum_{i=1}^{T}
\log p(y_i\mid H_v,H_q,y_{<i})
```

image token과 prompt token은 context로 사용하고, 일반적으로 answer token 위치에 loss를 적용한다. 논문이 별도의 contrastive loss나 detection loss를 추가하지 않는다는 점이 구조의 단순함이다.

## 학습 데이터와 recipe

### 두 단계 데이터

- pretraining: ShareGPT4V-PT 1.2M
- multi-task fine-tuning: 총 2.4M
  - Visual Dialog 123K
  - TextVQA 35K
  - VSR 13K
  - VIGC 37K
  - IConQA 107K
  - ScienceQA 13K
  - COCO 592K
  - SBU 844K
  - ShareGPT4V 665K
- 두 단계를 합친 전체 sample 수: 약 3.6M

데이터 집합의 task 성격이 서로 다르므로 단순히 sample 수만 늘었다고 보기는 어렵다. captioning, visual question answering, OCR, reasoning data의 혼합 자체가 benchmark별 성능에 영향을 준다.

### 학습 설정

| 항목 | Pretraining | Multi-task training |
| --- | ---: | ---: |
| 입력 해상도 | 336 x 336 | 336 x 336 |
| visual token | 144 | 144 |
| global batch | 256 | 128 |
| step | 5K | 19K |
| optimizer | AdamW | AdamW |
| schedule | cosine | cosine |
| base learning rate | `2e-5` | `4e-5` |
| projector learning rate | `1e-3` | `4e-5` |
| weight decay | 0 | 0 |
| warmup ratio | 0.03 | 0.03 |
| DeepSpeed | stage 2 | stage 3 |
| hardware | 8 x A100 | 8 x A100 |
| 시간 | 약 5 h | 약 9 h |

vision encoder는 양쪽 단계 모두 frozen이고 projector와 LLM은 trainable이다. pretraining 초기에 projector learning rate를 크게 두어 frozen visual space를 LLM embedding space에 빠르게 맞춘 뒤, multi-task 단계에서는 두 모듈을 같은 작은 learning rate로 조정한다.

## Forward pseudocode

```python
def forward(image, prompt_ids, answer_ids=None):
    # image: [B, 3, 336, 336]
    with no_grad():
        patch = clip_vit_l14(image)          # [B, 576, Dv]

    x = pointwise_1(patch)
    x = gelu(x)
    x = pointwise_2(x)                      # [B, 24, 24, Dt]

    x = avg_pool_2x2(x)                     # [B, 12, 12, Dt]
    x = depthwise_conv_3x3(x) + x           # PEG
    visual = flatten_spatial(x)              # [B, 144, Dt]

    text = llm.embed_tokens(prompt_ids)      # [B, Nq, Dt]
    prefix = concat([visual, text], dim=1)   # [B, 144 + Nq, Dt]

    if answer_ids is None:
        return llm.generate(inputs_embeds=prefix)

    answer = llm.embed_tokens(answer_ids)
    hidden = concat([prefix, answer], dim=1)
    logits = llm(inputs_embeds=hidden)
    return cross_entropy_on_answer_positions(logits, answer_ids)
```

실제 구현에서는 image start/end token, tokenizer template, label masking, attention mask, padding, past KV cache가 필요하다. 또한 CLIP이 class token을 반환하는 경우 어떤 token을 projector에 넘기는지 공식 구현과 일치시켜야 한다.

## 정량 결과

논문은 GQA, ScienceQA, TextVQA, POPE, MME, MMBench를 평가한다. 아래 평균은 MME를 2000으로 나누어 다른 백분율형 지표와 함께 평균한 논문 방식이다.

| Model | GQA | SQA | TextVQA | POPE | MME | MMB | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileVLM 1.7B | 56.1 | 57.3 | 41.5 | 84.5 | 1196.2 | 53.2 | 58.7 |
| MobileVLM V2 1.7B | 59.3 | 66.7 | 52.1 | 84.3 | 1302.8 | 57.7 | 64.2 |
| MobileVLM 3B | 59.0 | 61.2 | 47.5 | 84.9 | 1288.9 | 59.6 | 62.8 |
| MobileVLM V2 3B | 61.1 | 70.0 | 57.5 | 84.7 | 1440.5 | 63.2 | 68.1 |
| MobileVLM V2 7B | 62.6 | 74.8 | 62.3 | 85.3 | 1560.7 | 69.2 | 72.1 |
| V2 7B, pooling 없음 | 64.6 | 74.8 | 66.8 | 86.1 | 1558.7 | 70.8 | 73.5 |

1.7B에서 기존 MobileVLM 대비 평균은 58.7에서 64.2로 5.5 point, 3B에서는 62.8에서 68.1로 5.3 point 상승한다. 그러나 이 차이는 projector 하나의 효과가 아니다. 데이터와 학습 전략도 동시에 바뀌었으므로 다음 ablation을 함께 봐야 한다.

7B에서 pooling을 제거해 visual token을 144에서 576으로 늘리면 평균은 72.1에서 73.5로 오른다. 특히 TextVQA가 62.3에서 66.8로 4.5 point 오른다. 작은 글자와 OCR 정보는 2 x 2 평균 pooling에서 손실되기 쉽다는 직접적인 증거다. 따라서 token 축소는 무조건적인 이득이 아니라 정확도와 latency의 조절 knob다.

## Ablation 해석

### 데이터와 학습 전략

| Data 개선 | Training 개선 | GQA | SQA | TextVQA | POPE | MME | MMB | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 아니오 | 아니오 | 56.1 | 57.3 | 41.5 | 84.5 | 1196.2 | 53.2 | 58.7 |
| 예 | 아니오 | 57.5 | 63.9 | 49.8 | 83.9 | 1157.5 | 51.6 | 60.8 |
| 예 | 예 | 58.5 | 65.4 | 50.8 | 83.4 | 1262.6 | 55.4 | 62.8 |

데이터 개선만으로 평균이 2.1 point 상승하고, 학습 전략까지 적용하면 추가로 2.0 point 상승한다. 항목별로는 모두 단조 증가하지 않는다. 예를 들어 data-only 설정에서 MME와 MMB는 baseline보다 낮다. 여러 benchmark를 하나의 평균으로 압축하면 이런 trade-off가 가려질 수 있다.

### Projector 구성

| Row | 구성 | Token | Params | GQA | SQA | TextVQA | POPE | MME | MMB | Avg. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `[PW] x 2`, 기존 학습 | 576 | 6.30M | 56.9 | 57.1 | 43.7 | 85.7 | 1137.7 | 52.8 | 58.8 |
| 2 | 기존 LDP, 기존 학습 | 144 | 18.94M | 56.1 | 57.3 | 41.5 | 84.5 | 1196.2 | 53.2 | 58.7 |
| 3 | `[PW] x 2`, D&T | 576 | 6.30M | 59.9 | 63.7 | 53.9 | 85.0 | 1271.3 | 56.0 | 63.7 |
| 4 | 기존 LDP, D&T | 144 | 18.94M | 58.5 | 65.4 | 50.8 | 83.4 | 1262.6 | 55.4 | 62.8 |
| 5 | PW + AvgPool, D&T | 144 | 6.30M | 59.3 | 65.0 | 53.1 | 84.0 | 1292.2 | 54.5 | 63.2 |
| 6 | Row 5 + learnable PE | 144 | 6.59M | 59.1 | 67.1 | 52.3 | 84.3 | 1286.7 | 55.5 | 63.7 |
| 7 | Row 5 + PEG, LDPv2 | 144 | 6.32M | 59.3 | 66.7 | 52.1 | 84.3 | 1302.8 | 57.7 | 64.2 |

`D&T`는 개선된 data와 training strategy를 뜻한다. Row 3과 4를 비교하면 576 token이 144 token보다 0.9 point 높지만, LDPv2는 144 token을 유지하면서 64.2로 Row 3의 63.7을 넘어선다. 평균 pooling은 기존 LDP보다 0.4 point, learnable position embedding은 거기서 0.5 point, PEG는 다시 0.5 point 높인다. 다만 서로 다른 row 간 작은 차이는 seed variance가 보고되지 않아 통계적 유의성을 판단하기 어렵다.

## Edge 성능: 논문이 실제로 측정한 것

### A100 처리량

논문 Figure 3은 batch 1에서 256 token을 생성한다. MobileVLM V2 1.7B와 3B는 각각 37.37 tokens/s와 28.97 tokens/s를 기록하며, 비교한 MoE 모델보다 각각 1.65배 빠르다고 보고한다. 이 수치는 datacenter GPU 결과이며 mobile latency로 간주할 수 없다.

### Jetson Orin

Jetson Orin 실험은 `llama.cpp`, 4-bit LLM, custom CUDA LDPv2를 사용하고 256 output token을 생성한다.

| Model | Eval Avg. | Speed (tokens/s) | Total time (s) |
| --- | ---: | ---: | ---: |
| ShareGPT4V-7B | 70.8 | 13.00 | 19.69 |
| LLaVA-1.5-7B | 68.8 | 12.96 | 19.75 |
| MobileVLM V2-7B | 72.1 | 15.49 | 16.53 |
| LLaVA-1.5-3.3B | 62.7 | 20.45 | 12.52 |
| MobileVLM-3B | 62.8 | 30.80 | 8.31 |
| MobileVLM V2-3B | 68.1 | 30.80 | 8.38 |
| LLaVA-1.5-1.4B | 55.7 | 43.39 | 5.90 |
| MobileVLM-1.7B | 58.7 | 49.80 | 5.14 |
| MobileVLM V2-1.7B | 64.2 | 51.63 | 4.96 |

V2-1.7B는 LLaVA-1.5-1.4B보다 평균 정확도가 8.5 point 높고 total time은 5.90초에서 4.96초로 짧다. 3B에서는 기존 MobileVLM과 표시된 tokens/s가 동일하고 total time은 오히려 8.31초와 8.38초로 거의 같다. 즉 V2의 큰 장점은 같은 속도대에서 정확도를 올린 데 있으며 모든 규모에서 wall-clock이 크게 개선된 것은 아니다.

### 보고되지 않은 지표

다음 항목은 논문에서 분리 측정되지 않았다.

- smartphone CPU, GPU, NPU latency
- TTFT
- vision encoder latency
- projector latency
- prefill time과 decode time의 분리
- p50 및 p95 latency
- peak RAM 또는 VRAM
- 전력, energy per token, 온도, throttling
- quantization에 따른 정확도 변화

Jetson Orin은 edge platform이지만 smartphone은 아니다. 또한 표의 tokens/s는 주로 autoregressive decode 처리량을 반영한다. visual token 축소가 가장 직접적으로 영향을 주는 prefill과 TTFT를 별도로 보고하지 않아, 구조적 장점을 실제 latency breakdown으로 완전히 검증하지는 못했다.

## On-device 관점의 operator 분석

LDPv2의 강점은 operator가 단순하다는 것이다.

- pointwise convolution: `1 x 1 Conv` 또는 GEMM으로 쉽게 lower 가능
- GELU: 흔히 지원되지만 일부 NPU에서는 approximation 또는 graph 분할 가능
- 2 x 2 average pooling: 대부분 accelerator가 native 지원
- 3 x 3 depthwise convolution: mobile NPU/DSP가 잘 최적화하는 대표 연산
- residual add: 저비용이지만 layout 또는 quantization scale이 다르면 requantization 발생 가능

주의할 병목은 projector보다 CLIP ViT-L/14와 LLM이다. frozen 여부와 관계없이 CLIP은 336 x 336에서 576-token global attention을 수행한다. LDPv2는 encoder 뒤에서 token을 줄이므로 vision encoder activation memory는 줄이지 못한다. end-to-end peak가 vision encoder 내부에서 발생한다면 projector 개선만으로 peak memory가 크게 내려가지 않을 수 있다.

배포 graph에서는 다음을 확인해야 한다.

1. CLIP output이 `[B,N,C]`인지 `[B,H,W,C]`인지에 따른 reshape와 transpose
2. pointwise convolution을 NHWC로 실행할지 GEMM으로 실행할지
3. average pooling과 depthwise convolution 사이 layout 변환 유무
4. PEG residual의 quantization scale 일치 여부
5. visual embedding과 text embedding concat이 복사를 만드는지 view로 처리되는지
6. prefill과 decode가 서로 다른 execution provider로 나뉘는지

## Quantization과 memory에 대한 reviewer 계산

LDPv2 6.32M parameter의 순수 weight payload는 대략 다음과 같다.

| Precision | 이론적 weight payload |
| --- | ---: |
| FP32 | 24.1 MiB |
| FP16 | 12.1 MiB |
| INT8 | 6.0 MiB |

이 값은 scale, zero point, padding, runtime packing을 제외한 단순 계산이다. 논문의 Jetson 실험은 LLM 4-bit를 명시하지만 vision encoder와 projector precision, quantization calibration, model 전체 실제 파일 크기는 상세히 보고하지 않는다.

LLM parameter를 `P`라 하면 ideal 4-bit payload는 `P/2` bytes이다. 예를 들어 1.4B parameter는 이상적으로 약 0.7 GB지만 실제로는 group scale, metadata, tokenizer, KV cache, vision encoder, projector, allocator가 추가된다. 따라서 "1.4B 4-bit이므로 전체 앱이 0.7 GB"라고 결론 내릴 수 없다.

## 강점

1. visual token 수를 구조적으로 4배 줄여 LLM prefill 비용을 직접 겨냥한다.
2. `Conv`, `AvgPool`, `DWConv`, `Add`만 사용해 mobile compiler 친화적이다.
3. 작은 1.7B와 3B 모델에서 데이터, 학습, projector 효과를 분리한 ablation을 제공한다.
4. 576-token 설정도 함께 평가해 OCR 정확도와 token budget의 trade-off를 보여 준다.
5. A100뿐 아니라 Jetson Orin에서 4-bit LLM 실측을 제시한다.

## 한계와 비판적 해석

### 1. 진짜 phone 실험이 아니다

제목과 문제 설정은 mobile VLM을 강조하지만 실제 edge 표는 Jetson Orin이다. smartphone의 unified memory, thermal envelope, NPU operator coverage와는 조건이 다르다. phone 배포 가능성은 유망하지만 논문 수치로 입증되지는 않았다.

### 2. TTFT가 없어 visual token 축소 효과를 직접 보기 어렵다

visual token은 decode 단계의 매 step 연산보다 initial prefill과 KV cache에 더 직접적으로 영향을 준다. 그런데 표는 total generation time과 tokens/s만 제공한다. encoder, projector, prefill, decode를 분리했어야 LDPv2의 시스템 효과를 정확히 설명할 수 있다.

### 3. 2 x 2 pooling은 작은 시각 정보를 잃는다

7B의 TextVQA가 576 token에서 66.8, 144 token에서 62.3이라는 결과는 명확하다. 문서, UI, 장면 문자처럼 spatial detail이 중요한 사용 사례에서는 고정 144 token이 최적이 아닐 수 있다.

### 4. Vision encoder 병목은 그대로다

token 축소가 CLIP 뒤에서 일어나므로 CLIP ViT-L/14의 576-token self-attention, weight read, activation은 그대로다. 저전력 phone에서는 LLM보다 vision encoder가 TTFT의 큰 부분을 차지할 수도 있다.

### 5. 평균 지표가 benchmark 성격을 숨긴다

MME를 2000으로 나눈 뒤 서로 다른 scale과 의미를 가진 benchmark를 단순 평균한다. 모델 선택에는 평균뿐 아니라 OCR, hallucination, reasoning을 사용 사례에 맞게 따로 봐야 한다.

### 6. 반복 실험과 분산이 없다

projector ablation의 0.4에서 0.5 point 차이를 해석하려면 seed별 평균과 표준편차가 필요하다. 논문은 단일 값만 제시한다.

## 재현 체크리스트

- [ ] 정확한 CLIP ViT-L/14 checkpoint와 336 preprocessing 확인
- [ ] resize, crop, normalization 및 RGB channel order 고정
- [ ] CLIP class token 포함 여부와 576 patch token 선택 확인
- [ ] projector PW layer의 input/output dimension과 bias 확인
- [ ] 2 x 2 AvgPool의 stride, padding, tensor layout 확인
- [ ] PEG 3 x 3 depthwise convolution의 padding과 residual 확인
- [ ] image start/end token과 conversation template 일치
- [ ] pretraining 1.2M, multi-task 2.4M 데이터 mixture 검증
- [ ] frozen vision encoder, trainable projector 및 LLM 설정 검증
- [ ] 두 단계 learning rate와 DeepSpeed stage 재현
- [ ] benchmark prompt와 evaluation script version 고정
- [ ] `llama.cpp` commit, quantization format, group size 기록
- [ ] warm-up 뒤 p50/p95 latency를 같은 output length에서 측정
- [ ] encoder, projector, prefill, decode 시간을 각각 측정
- [ ] peak RSS/VRAM, energy, 온도, throttling을 함께 기록

## 온디바이스 재현 실험 설계

### 실험 A: visual token budget

같은 checkpoint에서 pooling을 바꾸거나 제거해 `N_v={64,144,256,576}`을 비교한다.

- 정확도: GQA, TextVQA, POPE, task-specific validation
- latency: vision encode, projector, prefill, decode, total
- memory: peak RAM, KV cache, allocator high-water mark
- 품질 분석: 작은 글자, 작은 객체, dense scene별 failure case

### 실험 B: precision 분해

다음 설정을 같은 장치에서 비교한다.

1. 전 모듈 FP16
2. vision encoder INT8, projector FP16, LLM W4A16
3. vision encoder INT8, projector INT8, LLM W4A16
4. 가능한 경우 LLM W4A8 또는 NPU delegate

평균 정확도만 보지 말고 TextVQA와 POPE를 별도로 기록한다. projector INT8에서 PEG residual scale mismatch가 생기는지도 확인한다.

### 실험 C: dynamic visual token

OCR 또는 dense scene detector로 detail score를 계산하고 간단한 장면은 144 token, 문서와 UI는 576 token을 사용한다. 이때 routing 자체의 latency와 오판 비용을 포함해야 한다. 목표는 평균 token 수를 낮추면서 TextVQA 하락을 줄이는 것이다.

## 연구 확장 아이디어

1. **ROI-aware pooling**: uniform 2 x 2 average 대신 detector 또는 saliency가 지정한 영역은 높은 해상도로 유지한다.
2. **Token importance distillation**: 576-token teacher의 attention 또는 answer logit을 이용해 144-token student를 학습한다.
3. **Shared encoder**: detector와 VLM이 같은 camera frame feature를 공유해 CLIP 재실행을 줄인다.
4. **Early-exit vision encoder**: 질문 난이도나 scene 변화량에 따라 CLIP layer 수를 조절한다.
5. **Compiler-aware projector search**: FLOPs가 아니라 target NPU의 layout conversion, fusion, SRAM 사용량을 objective로 둔다.
6. **TTFT-first benchmark**: 256-token decode 처리량보다 실제 assistant UX에 가까운 16에서 64 output token과 TTFT를 중심으로 평가한다.

## 최종 평가

MobileVLM V2의 가장 중요한 기여는 복잡한 새 multimodal module이 아니라, visual token을 어디서 얼마나 줄일지에 대한 좋은 engineering answer다. 336 x 336 CLIP이 만든 576 token을 144 token으로 줄이면 LLM의 prefix가 432 token 짧아지고, prefill 연산과 KV cache가 함께 감소한다. LDPv2는 이 과정을 mobile-friendly operator로 구현하면서 1.7B 모델 평균을 58.7에서 64.2로 높였다.

다만 논문이 입증한 범위는 Jetson Orin의 4-bit LLM 생성까지다. phone의 TTFT, peak memory, 전력과 thermal stability는 보고되지 않았다. 또한 OCR benchmark가 보여 주듯 고정 pooling은 시각 detail을 잃는다. 따라서 실제 온디바이스 시스템에서는 `144 token 고정`을 결론으로 받아들이기보다, 장면과 질문에 따라 token budget을 조절하는 출발점으로 보는 것이 가장 타당하다.

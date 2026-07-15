# 67. MiniCPM-V: A GPT-4V Level MLLM on Your Phone

## 논문 정보

- 원본 파일: `67_MiniCPM_V.pdf`
- 제목: **MiniCPM-V: A GPT-4V Level MLLM on Your Phone**
- 저자: Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, Maosong Sun
- 소속: MiniCPM-V Team, OpenBMB
- 공개: arXiv preprint, 2024
- 링크: [arXiv:2408.01800](https://arxiv.org/abs/2408.01800)
- 코드: [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)
- 핵심 키워드: mobile MLLM, adaptive visual encoding, perceiver resampler, visual token compression, RLAIF-V, Q4_K_M, QNN

## 한 문장 요약

MiniCPM-Llama3-V 2.5는 SigLIP의 고해상도 slice를 slice당 96 visual token으로 압축하고 Llama3-Instruct 8B에 연결하며, 4-bit quantization과 QNN NPU, llama.cpp 최적화를 통해 Xiaomi 14 Pro에서 약 10.7초의 load-plus-prefill과 8.2 tokens/s를 실측한 end-side MLLM이다.

## 논문을 읽을 때 먼저 구분할 세 모델

MiniCPM-V는 단일 checkpoint 이름이 아니라 세 세대의 series다.

| Model | Total size | Base LLM | Resolution | Aspect ratio | Visual token/slice | Alignment |
| --- | ---: | --- | --- | --- | ---: | --- |
| MiniCPM-V 1.0 | 2.8B | MiniCPM 2B | 0.2M pixel, 448 x 448 | fixed | 64 | 없음 |
| MiniCPM-V 2.0 | 2.8B | MiniCPM 2B | 최대 1.8M pixel | arbitrary | 64 | RLHF-V |
| MiniCPM-Llama3-V 2.5 | 8.5B | Llama3-Instruct 8B | 최대 1.8M pixel | arbitrary | 96 | RLAIF-V |

논문의 architecture, benchmark, phone deployment 주장은 주로 최신 2.5 모델을 가리킨다. 2.8B와 8.5B 결과를 하나의 모델처럼 섞으면 안 된다.

## 전체 구조

```text
high-resolution image, arbitrary aspect ratio
  -> adaptive partition
       choose rows x columns
       preserve original aspect ratio
       add one global overview slice
  -> each slice resized near ViT pretraining area
  -> 2D position embedding interpolation
  -> SigLIP SoViT-400m/14
       1024 patch tokens per slice
  -> shared one-layer cross-attention resampler
       96 queries per slice for V2.5
  -> spatial special tokens and row separators
  -> concatenate visual tokens + text tokens
  -> Llama3-Instruct 8B
  -> autoregressive response
```

세 모듈은 visual encoder, shared compression layer, LLM이다. compression layer는 slice마다 별도 parameter를 두지 않고 공유되므로 slice 수가 증가해도 weight는 늘지 않는다. 그러나 vision encoder forward, intermediate activation, visual prefix는 slice 수에 거의 선형으로 증가한다.

## Adaptive visual encoding

### 1. 이상적인 slice 수

입력 image의 width와 height를 `(W_I,H_I)`, ViT pretraining resolution을 `(W_v,H_v)`라 하자. 이상적인 slice 수는 area ratio의 ceiling이다.

```math
N=\left\lceil
\frac{W_IH_I}{W_vH_v}
\right\rceil
```

단순히 square tile을 강제하면 긴 영수증이나 panorama가 심하게 왜곡된다. 그래서 `m` columns와 `n` rows를 선택할 때 slice aspect ratio가 ViT pretraining aspect ratio와 가까운지를 평가한다.

```math
S(m,n)=
-\left|
\log\frac{W_I/m}{H_I/n}
-\log\frac{W_v}{H_v}
\right|
```

```math
(m^*,n^*)=
\arg\max_{(m,n)\in\bar C}S(m,n)
```

기본 후보 `C_N`은 `m x n=N`인 integer factor pair다. `N`이 prime이면 `1 x N`과 `N x 1`밖에 없어 불리하므로 `C_(N-1)`, `C_N`, `C_(N+1)`을 합친 후보를 쓴다. 실전에서는 slice 수가 10 미만이 되도록 제한하고 최대 약 1.8M pixel, 예를 들어 1344 x 1344를 지원한다.

### 2. Slice encoding

각 slice는 aspect ratio를 유지하면서 area가 `W_v x H_v`와 비슷해지도록 resize한다. 그 뒤 pretrained 1D positional embedding을 2D grid로 복원하고 새 slice grid에 맞게 interpolation한다.

```math
P_1\in\mathbb{R}^{Q\times l},
\quad Q=q^2
```

```math
P_1\rightarrow P_2\in\mathbb{R}^{q\times q\times l}
\rightarrow \mathrm{Interp2D}(P_2,H_s,W_s)
```

local slice만 쓰면 전체 layout을 잃기 때문에 원본 image를 global overview slice로 추가한다. local detail과 holistic context를 함께 주는 대신 encoder forward가 하나 더 생긴다.

### 3. Spatial schema

각 slice token을 `<slice>`와 `</slice>`로 감싸고 서로 다른 row는 newline special token으로 구분한다. LLM이 sequence order만으로도 tile의 2D 배치를 복원하도록 돕는다. 실제 total sequence에는 96의 배수 외에 이 special token이 추가된다.

## Token compression

SigLIP은 slice 하나를 1,024 patch token으로 encoding한다. 10개 slice면 10,240개 이상이 된다. 이를 그대로 Llama3에 넣는 것은 phone에서 비현실적이다.

MiniCPM-V의 compression layer는 learnable query와 image patch 사이의 one-layer cross-attention이다.

```math
Q\in\mathbb{R}^{M\times d},
\quad
K,V\in\mathbb{R}^{1024\times d}
```

```math
H_v=\mathrm{softmax}
\left(\frac{QK^\top}{\sqrt d}\right)V
\in\mathbb{R}^{M\times d}
```

- V1.0 및 V2.0: `M=64`
- Llama3-V 2.5: `M=96`

V2.5는 image당 96에서 960 visual token을 사용한다. source patch 기준 최대 10,240에서 960으로 약 10.7배 줄인다. 이는 input slice가 많아질 때 token 수가 고정된다는 뜻은 아니다. **slice당** 96개이므로 high-resolution input은 여전히 visual prefix가 길어진다.

cross-attention score는 slice당 `96 x 1024 = 98,304` element다. 10 slice면 983,040 element로, 10,240 source token을 LLM self-attention에 직접 넣는 것보다 훨씬 작다. 다만 SigLIP 자체가 각 slice를 처리하는 비용은 그대로 남는다.

## Tensor shape와 activation 계산

batch 1, V2.5를 기준으로 핵심 shape는 다음과 같다.

| 지점 | 1 slice | 최대 10 slice |
| --- | ---: | ---: |
| SigLIP patch token | `1024 x D_v` | `10240 x D_v` |
| resampler query | `96 x D_v` | shared weight |
| compressed token | `96 x D_l` | `960 x D_l` |
| LLM prefix | `(96+N_t) x D_l` | `(960+N_t) x D_l` |

`D_v`, projection 세부와 special token은 released config에서 확인해야 한다. 다음은 Llama3-8B의 일반적 config인 `D_l=4096`, 32 layers, 32 query heads, 8 KV heads, head dimension 128을 가정한 reviewer 예시다. 논문이 측정한 memory가 아니다.

GQA의 KV dimension은 다음과 같다.

```math
D_{KV}=8\times128=1024
```

FP16 visual prefix의 KV cache는 다음과 같다.

```math
M_{KV}=2\times L\times N_v\times D_{KV}\times2\ \text{bytes}
```

| Visual token | KV cache, 32 layers, FP16 |
| ---: | ---: |
| 96 | 약 12 MiB |
| 960 | 약 120 MiB |
| 차이 | 약 108 MiB |

만약 GQA를 무시하고 `D_KV=D_model`로 계산하면 4배 과대 추정한다. quantized weight와 KV cache precision도 별개다. 논문의 Q4_K_M은 주로 weight quantization이며 KV가 자동으로 4-bit가 되는 것은 아니다.

960개 visual embedding 자체는 FP16에서 다음 raw payload를 갖는다.

```math
960\times4096\times2
=7,864,320\ \text{bytes}
\approx7.5\ \text{MiB}
```

LLM layer activation과 attention workspace를 포함하면 더 크다. phone peak memory는 5GB quantized weight만으로 결정되지 않는다.

## Autoregressive objective

compressed visual token `H_v`, prompt `H_q`, response `Y=(y_1,...,y_T)`에 대해 다음 likelihood를 학습한다.

```math
p(Y\mid H_v,H_q)
=\prod_{i=1}^{T}
p(y_i\mid H_v,H_q,y_{\lt i})
```

SFT에서는 answer position의 negative log likelihood를 최소화한다. pretraining에서는 LLM을 단계에 따라 동결해 낮은 품질의 web caption이 language capability를 흔드는 것을 막는다.

## 3-stage multimodal pretraining

### Stage 1: compression warm-up

- resolution: 224 x 224
- trainable: randomly initialized compression layer
- frozen: visual encoder와 LLM
- data: image captioning에서 random 200M

### Stage 2: visual resolution 확장

- resolution: 224에서 448 x 448로 증가
- trainable: visual encoder 전체
- frozen: compression layer와 LLM
- data: 별도 image captioning 200M

### Stage 3: adaptive high-resolution

- adaptive slice와 arbitrary aspect ratio 활성화
- trainable: visual encoder와 compression layer
- frozen: LLM
- data: caption뿐 아니라 OCR data 포함

pretraining data 표는 English caption 410M, Chinese caption 110M, English OCR/knowledge 39M, Chinese OCR 11M으로 총 약 570M source를 제시한다. 각 stage의 sampling과 중복 여부 때문에 단순 합으로 실제 seen sample을 추정하면 안 된다.

### Caption rewriting과 packing

web caption의 오류와 중복을 줄이기 위해 GPT-4 seed로 작은 LLM을 fine-tune하고 raw caption을 question-answer 형식으로 rewrite한다. 길이가 다른 sample은 fixed-length sequence에 pack하고 position id와 attention mask를 수정해 sample 간 leakage를 막는다. 논문은 packing으로 pretraining이 2-3배 빨라졌다고 보고한다.

packing mask가 잘못되면 한 image-caption sample이 다음 sample의 token을 볼 수 있다. 재현 시 block-diagonal causal mask와 label mask를 unit test해야 한다.

## Supervised fine-tuning

SFT에서는 데이터 품질이 높다고 보고 모든 parameter를 연다. Part 1은 short answer와 basic recognition, Part 2는 detailed answer와 instruction following을 강화한다.

Part 1에는 short caption 560K, VQA 1.4M, knowledge 60K, grounding 570K, reasoning 135K, math 125K, OCR 1.7M, chat 780K가 포함된다. Part 2에는 Part 1 재표본 400K, OCR 690K, instruction 1.9M, text-only data가 포함된다. V2.5는 Cauldron 2M과 36개 언어의 multilingual data 90K도 추가한다.

Part 1과 Part 2는 단순 random mixture가 아니라 순서대로 이어 학습한다. 후반 데이터가 response style을 결정한다는 가정이다.

## RLAIF-V

hallucination을 줄이기 위해 open-source AI feedback으로 preference pair를 만든다.

1. policy model이 한 instruction에 대해 high-temperature response 10개 생성
2. Llama3-8B가 각 response를 atomic claim으로 분해
3. claim을 yes/no visual question으로 변환
4. open MLLM이 각 claim의 image grounding을 판정
5. invalid claim 수의 음수를 response score로 사용
6. 더 좋은 response와 나쁜 response pair로 DPO 수행

MiniCPM-V 2.0 scoring에는 OmniLMM-12B, V2.5에는 LLaVA-NeXT-Yi-34B를 쓴다. 최종 preference data는 3K unique image에서 6K pair다.

일반 DPO objective를 쓰면 다음과 같다.

```math
\mathcal{L}_{DPO}
=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]
\right)
```

AI judge가 놓치는 subtle error와 preference bias는 그대로 student에 들어갈 수 있다. 6K pair는 전체 SFT에 비해 작지만 hallucination behavior에 집중된 후반 alignment라는 점이 중요하다.

## Inference pseudocode

```python
def encode_image(image, vit, resampler):
    n_ideal = ceil(image.width * image.height / vit.pretrain_area)
    candidates = factor_pairs(n_ideal - 1, n_ideal, n_ideal + 1)
    cols, rows = max(candidates, key=lambda mn: aspect_score(image, mn))

    local_slices = partition(image, cols=cols, rows=rows)
    slices = [make_global_overview(image)] + local_slices

    compressed = []
    for tile in slices:
        tile = resize_equal_area(tile, vit.pretrain_area)
        pos = interpolate_2d_position(vit.position_embedding, tile.shape)
        patch = vit(tile, position_embedding=pos)       # [1024, Dv]
        token = resampler(patch, num_queries=96)       # [96, Dl]
        compressed.append(token)

    return insert_slice_and_row_tokens(compressed, rows, cols)

def generate(image, prompt):
    visual = encode_image(image, vit, resampler)
    text = llm.embed_tokens(tokenize(prompt))
    return llm.generate(inputs_embeds=concat([visual, text]))
```

## Benchmark 결과

### General multimodal

| Model | Size | OpenCompass | MME | MMB en | MMB cn | MMMU | MathVista | LLaVA Bench | RealWorldQA | ObjHal Res/Men lower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniCPM-V 1.0 | 2.8B | 47.5 | 1650.2 | 64.1 | 62.6 | 38.3 | 28.9 | 51.3 | 51.2 | 21.6 / 11.5 |
| MiniCPM-V 2.0 | 2.8B | 54.5 | 1808.6 | 69.1 | 66.5 | 38.2 | 38.7 | 69.2 | 55.8 | 14.5 / 7.8 |
| MiniCPM-Llama3-V 2.5 | 8.5B | 65.1 | 2024.6 | 77.2 | 74.2 | 45.8 | 54.3 | 86.7 | 63.5 | 10.3 / 5.0 |
| GPT-4V 2023-11-06 | 미공개 | 63.5 | 1771.5 | 77.0 | 74.4 | 53.8 | 47.8 | 93.1 | 63.0 | 13.6 / 7.3 |

V2.5가 OpenCompass 평균에서는 비교한 GPT-4V보다 높지만 모든 항목에서 우위는 아니다. GPT-4V는 MMMU와 LLaVA Bench가 더 높다. 제목의 "GPT-4V level"은 특정 시점의 aggregate benchmark claim이지 전반적 capability equivalence가 아니다.

### OCR

| Model | OCRBench | TextVQA val | DocVQA test |
| --- | ---: | ---: | ---: |
| MiniCPM-V 1.0 | 366 | 60.6 | 38.2 |
| MiniCPM-V 2.0 | 605 | 74.1 | 71.9 |
| MiniCPM-Llama3-V 2.5 | 725 | 76.6 | 84.8 |
| Gemini Pro | 680 | 74.6 | 88.1 |
| GPT-4V 2023-11-06 | 645 | 78.0 | 88.4 |

V2.5는 OCRBench에서는 두 proprietary model보다 높지만 TextVQA와 DocVQA에서는 GPT-4V보다 낮다. adaptive high-resolution과 OCR data의 효과가 크지만 benchmark별 결론을 구분해야 한다.

### Alignment와 multilingual ablation

RLAIF-V를 적용하면 OpenCompass가 64.5에서 65.1, Object HalBench의 response/mention accuracy 형태 값이 86.9/93.6에서 89.7/95.0으로 개선된다. MME와 MMMU는 소폭 낮아지고 LLaVA Bench는 85.4에서 86.7로 오른다.

multilingual SFT 90K를 추가하면 Korean evaluation이 13.7에서 67.9, Japanese 13.8에서 88.0, German 22.8에서 76.5로 오른다. 언어별 향상 폭이 다르고, evaluation이 translated LLaVA Bench와 GPT-4 judge에 의존한다.

## Phone deployment

### Quantization

V2.5 FP16은 약 16-17GB memory가 필요해 12-16GB phone에 어렵다. GGML의 `Q4_K_M` 4-bit weight quantization으로 약 5GB까지 줄인다. 단순히 8.5B x 0.5 byte면 4.25GB이고, scale, metadata, vision module, runtime buffer가 더해져 약 5GB가 되는 것은 합리적이다.

### Memory optimization

ViT와 LLM을 동시에 load하면 phone memory pressure로 paging이 발생한다. 논문은 다음처럼 sequential load한다.

```text
load ViT -> image encode -> release ViT
-> load LLM -> visual/text prefill -> decode
```

Xiaomi 14 Pro에서 image processing이 45.2초에서 31.5초로 줄었다. peak memory와 swap byte는 보고하지 않았지만, weight resident set을 줄여 latency를 개선한 실제 사례다. 단점은 매 query module load 비용과 repeated question에서 vision feature cache 전략이 복잡해진다는 것이다.

### Compilation과 configuration

target device에서 직접 compile해 instruction set과 맞추면 total encoding이 50.5초에서 17.0초로 줄고 decode는 1.3에서 3.2 tokens/s로 오른다. CPU core allocation 등을 automatic search하면 decode가 8.2 tokens/s로 오른다.

### NPU

Qualcomm phone에서는 SigLIP backend를 QNN으로 바꾸고 LLM은 llama.cpp CPU에 둔다. visual encoding이 3.7초에서 1.3초로 줄지만 LLM prefill은 9.4초로 남는다. NPU가 전체 모델을 가속한 것이 아니라 vision module만 가속했다.

## Xiaomi 14 Pro optimization breakdown

Figure 6의 encoding latency는 model loading과 encoding을 포함한다. bar를 읽으면 다음과 같다.

| Setting | Image processing | LLM prefill | 합 | Decode |
| --- | ---: | ---: | ---: | ---: |
| no optimization | 45.2 s | 19.4 s | 64.6 s | 1.3 tok/s |
| + memory optimization | 31.5 s | 19.0 s | 50.5 s | 미표기 |
| + compilation | 5.1 s | 11.9 s | 17.0 s | 3.2 tok/s |
| + configuration | 3.7 s | 9.2 s | 12.9 s | 8.2 tok/s |
| + NPU vision | 1.3 s | 9.4 s | 10.7 s | 8.2 tok/s로 해석 |

본문은 initial text encoding latency를 64.2초라고 서술하지만 Figure bar 합은 64.6초다. 이 작은 불일치는 그대로 기록하는 것이 안전하다. 또한 이 10.7초는 현대적인 interactive TTFT 기준으로는 여전히 길다.

## Device별 결과

| Device | Hardware | Image processing | LLM prefill | 합 | Decode |
| --- | --- | ---: | ---: | ---: | ---: |
| Xiaomi 14 Pro | Snapdragon 8 Gen 3, QNN NPU + CPU | 1.3 s | 9.4 s | 10.7 s | 8.2 tok/s |
| vivo X100 Pro | Dimensity 9300, CPU path | 4.0 s | 13.9 s | 17.9 s | 4.9 tok/s |
| MacBook Pro | M1 | 5.7 s | 4.7 s | 10.4 s | 16.9 tok/s |

Xiaomi만 NPU를 사용한다. Mac의 vision이 5.7초로 Xiaomi NPU보다 느린 대신 LLM prefill과 decode가 훨씬 빠르다. device마다 병목 module이 다르므로 한 개 tokens/s만으로 system을 비교하면 안 된다.

## TTFT, tokens/s, peak memory 해석

- 논문의 `encoding latency`: image processing + LLM prefill, Figure 6은 model load 포함
- 일반적인 TTFT와 유사하지만 tokenizer, first decode, UI overhead 포함 여부는 불명확
- decode throughput: steady autoregressive generation의 tokens/s
- peak memory: FP16 16-17GB, quantized 약 5GB라는 model requirement만 보고
- p50/p95: 미보고
- power와 energy: 미보고
- temperature와 throttling: 미보고

실제 앱에서는 첫 질문 후 LLM을 resident로 유지하거나 image embedding을 cache하면 load 비용이 달라질 수 있다. 논문 latency는 그 execution lifecycle을 충분히 설명하지 않는다.

## Operator와 deployment 병목

### Vision path

- image slicing, resize, 2D positional interpolation
- SigLIP patch embedding과 global attention
- one-layer cross-attention resampler
- per-slice loop 또는 batch 처리

QNN graph가 dynamic aspect ratio와 slice 수를 그대로 받기 어렵다면 몇 개 static shape bucket을 compile해야 한다. CPU에서 slice를 만들고 NPU로 여러 번 submit하면 command overhead와 tensor copy가 커질 수 있다.

### LLM path

- Q4_K_M packed GEMM
- RMSNorm, RoPE, SiLU/SwiGLU
- GQA attention과 KV cache
- prefill과 decode의 서로 다른 kernel shape

논문 결과는 prefill이 9.4초로 최종 병목임을 보여 준다. NPU vision 최적화 다음 단계는 visual token 감소, LLM prefill accelerator, image feature cache다.

## 강점

1. 실제 Xiaomi와 vivo flagship phone에서 end-to-end에 가까운 load, prefill, decode 수치를 공개한다.
2. 4-bit, memory lifecycle, target compilation, core allocation, NPU를 단계별로 ablation한다.
3. 1.8M pixel arbitrary aspect ratio와 96-960 visual token을 양립시킨다.
4. OCR, multilingual, hallucination alignment를 작은 모델 안에서 통합한다.
5. source patch 1,024개를 slice당 96개로 줄여 LLM 비용을 명시적으로 제어한다.

## 한계와 비판적 해석

### 1. "GPT-4V level"은 제한된 benchmark claim이다

특정 2023 checkpoint와 OpenCompass 평균을 비교한다. GPT-4V가 더 높은 benchmark도 있으며 real-world robustness, tool use, safety가 동등하다는 증거는 아니다.

### 2. Phone TTFT는 아직 10초 이상이다

8.2 tokens/s decode는 읽을 수 있는 수준이지만 첫 token 전 대기가 길다. real-time assistant라는 표현에는 신중해야 한다.

### 3. Model loading 조건이 불명확하다

sequential loading은 memory를 줄이지만 질문마다 load하는지 process lifetime에 한 번인지가 앱 경험을 크게 바꾼다. cold와 warm latency를 나눠야 한다.

### 4. 960 token은 여전히 큰 prefix다

high-resolution OCR에서 slice가 늘면 KV cache와 prefill이 선형 이상으로 증가한다. 실제 최종 bottleneck도 LLM prefill이다.

### 5. Peak memory와 energy가 없다

5GB weight requirement 외에 allocator peak, KV cache, NPU buffer, swap, battery drain과 thermal throttling을 보고하지 않는다.

### 6. Adaptive slicing의 ablation이 제한적이다

slice candidate, global overview, 64 대 96 query, resolution cap 각각의 accuracy-latency trade-off가 충분히 분리되지 않는다.

## 재현 체크리스트

- [ ] 정확한 MiniCPM-V generation과 checkpoint 확인
- [ ] SigLIP SoViT-400m/14 preprocessing과 normalization 고정
- [ ] `C_(N-1) union C_N union C_(N+1)` partition 구현 검증
- [ ] global overview slice 포함 여부 확인
- [ ] 2D position interpolation coordinate convention 확인
- [ ] slice당 source 1024 token과 output 96 token runtime 확인
- [ ] spatial special token과 newline 배치 일치
- [ ] Q4_K_M quantizer version, group, KV dtype 기록
- [ ] llama.cpp commit과 CPU thread/core affinity 기록
- [ ] QNN SDK, DSP/NPU runtime, static shape bucket 기록
- [ ] cold load, warm image, repeated question을 분리 측정
- [ ] image, prefill, first decode, steady decode timing 분리
- [ ] peak RSS, swap, energy, 온도, throttling 기록

## 추천 온디바이스 실험

visual token budget을 `96, 192, 384, 960`으로 bucket화해 같은 image와 question에서 비교한다.

- OCRBench, DocVQA, 앱별 OCR 정확도
- image processing, prefill, TTFT, tokens/s
- KV cache와 peak RSS
- NPU submission 횟수와 CPU-NPU copy byte
- 30회 연속 query의 온도 및 p95

두 번째 실험은 module lifecycle이다.

1. 매 query ViT와 LLM sequential load
2. LLM resident, image마다 ViT swap
3. ViT와 LLM 모두 resident
4. image feature와 prompt KV cache까지 재사용

memory와 UX의 실제 trade-off를 드러낼 수 있다.

## Adaptive slicing 계산 예시

가로로 긴 `1792 x 448` image와 ViT 기준 `448 x 448`을 생각하자. area ratio는 4이므로 ideal slice 수는 4다.

```math
N=\left\lceil\frac{1792\times448}{448\times448}\right\rceil=4
```

candidate에는 `1 x 4`, `2 x 2`, `4 x 1`과 `N-1`, `N+1`의 factor pair가 들어간다. image aspect ratio가 4:1이므로 4 columns x 1 row가 square slice를 만들어 score가 가장 높다. 여기에 global overview를 더하면 구현 해석에 따라 local 4개와 overview 1개를 처리한다. V2.5 resampler output은 대략 `5 x 96 = 480` visual token과 spatial separator다.

반대로 `1344 x 1344` square image는 area ratio가 9다. 3 x 3 partition이 aspect ratio를 잘 맞추지만 global overview 포함 여부와 maximum total slice 정책을 공식 code에서 확인해야 한다. 논문은 최종 visual token range를 96-960으로 명시하므로 단순 수식 결과가 10 slice를 넘으면 runtime cap이 적용된다고 보는 것이 안전하다.

이 예시는 같은 1.8M pixel이라도 aspect ratio와 partition에 따라 slice 수, padding, NPU invocation이 달라질 수 있음을 보여 준다. benchmark에는 resolution뿐 아니라 실제 `(rows,cols,total_slices)`를 기록해야 한다.

## Resampler의 정보 병목 분석

96개 query는 1,024 patch에서 answer에 필요한 정보를 미리 선택해야 한다. query가 content-adaptive cross-attention을 사용하므로 average pooling보다 유연하지만 다음 정보 손실이 가능하다.

- 매우 작은 여러 문자에 동일 query가 attention을 분산
- 반복되는 table cell의 위치가 압축 과정에서 혼합
- local slice 안의 pixel-level coordinate가 LLM embedding에 충분히 보존되지 않음
- global overview와 local slice의 중복 정보가 token budget을 소비

재현 ablation은 query를 `32,64,96,128`로 바꾸고 source patch와 output token 사이 attention entropy를 분석할 수 있다. OCR 정확도뿐 아니라 prefill latency와 KV cache를 같이 재야 한다. query 수를 96에서 64로 줄이면 visual prefix와 KV는 3분의 1 감소하지만 품질 변화는 task마다 다를 수 있다.

## Cold, warm, repeated-query latency

논문 Figure 6은 model loading을 포함한다고 밝힌다. 실제 앱 UX를 이해하려면 세 시나리오를 분리한다.

### Cold start

process가 없고 model file도 page cache에 없는 상태다. file open, mmap/read, dequantization metadata, graph load, tokenizer까지 포함한다. 첫 실행의 p95가 중요하다.

### Warm image query

process와 LLM weight는 resident지만 새 image를 encoding한다. ViT swap 전략을 쓰면 module load가 다시 생길 수 있다. camera assistant의 일반적인 새 frame 질의와 가깝다.

### Repeated question on one image

visual embedding과 prefix KV를 cache한다. text prompt만 추가하면 image processing을 건너뛸 수 있고 TTFT가 크게 줄 수 있다. 다만 conversation이 길어지면 text KV가 증가한다.

논문 수치가 어느 lifecycle에 해당하는지 완전히 분리되어 있지 않으므로 세 값을 직접 측정해야 한다. 평균 하나만 기록하면 sequential loading의 장단점을 판단할 수 없다.

## Energy와 thermal 검증 프로토콜

phone에서 10초 prefill과 수십 token decode는 burst benchmark보다 thermal 영향을 많이 받는다. 다음 절차가 재현에 적합하다.

1. screen brightness, airplane mode, battery level, ambient temperature를 고정한다.
2. 5회 warm-up 후 같은 image와 64 output token을 30회 반복한다.
3. query별 image, prefill, decode time과 average power를 기록한다.
4. device surface 및 SoC temperature, CPU frequency, NPU utilization을 함께 기록한다.
5. 첫 5회와 마지막 5회의 p50/p95 및 tokens/s를 비교한다.

energy per query는 단순하게 다음처럼 계산할 수 있다.

```math
E_{query}=\int_{t_0}^{t_1}P(t)dt
```

NPU가 visual latency를 3.7초에서 1.3초로 줄여도 high instantaneous power 때문에 energy가 같은 비율로 줄지는 않을 수 있다. 논문에는 power trace가 없어 별도 검증이 필요하다.

## Failure analysis 기준

- slice 경계에서 문장 또는 object가 잘림
- global overview에서는 보이지만 local detail token과 연결 실패
- OCR은 맞지만 table structure를 잘못 복원
- 언어 전환 시 visual fact는 맞고 문법만 틀림
- RLAIF-V 이후 과도하게 답변을 회피
- long prompt에서 prefill이 급증
- memory pressure로 NPU graph가 CPU fallback

failure마다 raw image, chosen partition, visual token 수, backend trace를 함께 저장해야 architecture와 deployment 원인을 분리할 수 있다.

## 최종 평가

MiniCPM-V 논문의 가장 값진 부분은 model accuracy보다 phone deployment breakdown이다. Q4_K_M으로 16-17GB를 약 5GB로 줄이고, sequential load, target compilation, core configuration, QNN vision acceleration을 쌓아 Xiaomi 14 Pro의 약 64초 초기 경로를 약 10.7초까지 줄였다. decode는 8.2 tokens/s다.

architecture 측면에서는 1,024 patch를 slice당 96 token으로 압축하는 resampler가 핵심이다. 최대 960 token은 고해상도 OCR 품질을 가능하게 하지만 최종 prefill 9.4초라는 새 병목을 만든다. 따라서 이 논문은 "phone에서 완전히 해결된 VLM"이라기보다, 실제 병목이 vision에서 LLM prefill로 이동하는 과정을 수치로 보여 준 중요한 systems baseline이다.

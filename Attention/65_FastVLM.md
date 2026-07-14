# 65. FastVLM: Efficient Vision Encoding for Vision Language Models

## 논문 정보

- 원본 파일: `65_FastVLM.pdf`
- 제목: **FastVLM: Efficient Vision Encoding for Vision Language Models**
- 저자: Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari
- 소속: Apple
- 공개: arXiv preprint, 2025
- 링크: [arXiv:2412.13303](https://arxiv.org/abs/2412.13303)
- 코드: [apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)
- 핵심 키워드: FastViTHD, high-resolution VLM, hierarchical vision encoder, visual token, TTFT, Core ML, MLX

## 한 문장 요약

FastVLM은 전체 downsampling stride가 64인 5-stage hybrid encoder `FastViTHD`로 고해상도 이미지를 빠르게 처리하고 visual token 수를 해상도에 따라 직접 조절하여, 별도 token pruning 없이 vision latency와 LLM prefill을 함께 줄인 VLM이다.

## 왜 vision encoder가 VLM의 핵심 병목인가

VLM의 첫 응답 시간은 보통 다음 두 부분으로 나뉜다.

```math
T_{TTFT}=T_{vision}+T_{projector}+T_{prefill}+T_{first\ decode}
```

이 논문은 측정 편의상 projector와 first decode의 작은 비용을 별도 표기하지 않고 다음 합을 TTFT로 사용한다.

```math
T_{TTFT}^{paper}=T_{vision\ encoder}+T_{LLM\ prefill}
```

고해상도 입력은 서로 다른 방식으로 두 항을 키운다.

- vision encoder: pixel과 intermediate feature map이 커지므로 convolution과 attention 비용이 증가한다.
- LLM prefill: vision encoder가 더 많은 token을 출력하면 decoder sequence가 길어진다.

기존 ViT는 patch size가 고정되어 있어 해상도를 2배 높이면 token이 4배가 된다. 예를 들어 patch 14인 ViT에 336 x 336을 넣으면 576 token이고, 1152 x 1152를 tiling하면 수천 token이 된다. token pruning이나 perceiver resampler를 뒤에 붙일 수 있지만 encoder는 이미 많은 patch를 계산한 뒤다.

FastVLM의 핵심 발상은 **처음부터 hierarchical backbone이 적은 token을 만들도록 설계**하는 것이다. 입력 해상도 자체가 visual token budget knob가 되므로 추가 pruning module이 필요 없다.

## 전체 구조

```text
image
  -> FastViTHD vision encoder
       convolutional stem
       Stage 1-3: RepMixer blocks
       Stage 4-5: self-attention blocks
       learned pooling of multi-scale features
  -> channel-wise concatenation
  -> connector / projection
  -> visual tokens

instruction
  -> tokenizer -> text embeddings

[visual tokens ; text tokens]
  -> Qwen2-0.5B / 1.5B / 7B or Vicuna-7B
  -> autoregressive answer
```

FastVLM 자체의 generation objective는 LLaVA 계열과 같다. 구조적 기여는 LLM이 아니라 vision encoder에 있다. 따라서 논문은 decoder 크기와 입력 해상도를 교차해 같은 runtime budget에서 어떤 조합이 Pareto optimal인지 분석한다.

## FastViTHD 구조

### 5-stage hybrid encoder

FastViTHD의 stage depth와 embedding dimension은 다음과 같다.

```text
depth = [2, 12, 24, 4, 2]
width = [96, 192, 384, 768, 1536]
```

- Stage 1-3: convolution 기반 RepMixer
- Stage 4-5: multi-head self-attention
- ConvFFN expansion ratio: 4.0
- parameter: 125.1M
- stem spatial downsampling: 한 축 기준 4배
- stage 사이 patch embedding: 7 x 7 depthwise convolution, stride 2와 1 x 1 pointwise convolution
- 전체 output stride: 64

7 x 7 depthwise patch embedding에는 train-time overparameterization을 사용하지만 inference graph에서는 재매개변수화된 단순 convolution으로 배포할 수 있다. 기존 FastViT의 squeeze-excite는 고해상도 latency에 불리해 제거했다.

MCi2 계열을 단순히 넓고 깊게 만드는 naive scaling은 stage 3과 4의 비교적 큰 feature map에서 attention을 많이 수행한다. FastViTHD는 stage를 하나 더 추가하고 큰 channel의 계산을 더 작은 spatial grid로 이동한다. 같은 125M급에서 256 입력은 MCi3 계열 설계가 naive scaled MCi2보다 1.9배, 1024 입력에서는 7.1배 빠르다는 후속 MobileCLIP2 분석과도 연결된다.

### 해상도와 token 수

전체 stride가 64이므로 square input `R x R`에서 마지막 grid와 token 수는 다음과 같다.

```math
H_5=W_5=\frac{R}{64},
\qquad
N_v=\left(\frac{R}{64}\right)^2
```

| Input | Last grid | Visual token |
| ---: | ---: | ---: |
| 256 x 256 | 4 x 4 | 16 |
| 512 x 512 | 8 x 8 | 64 |
| 768 x 768 | 12 x 12 | 144 |
| 1024 x 1024 | 16 x 16 | 256 |
| 1536 x 1536 | 24 x 24 | 576 |

같은 336 입력에서 ViT-L/14는 `24 x 24 = 576` token을 만들지만 FastViTHD 식으로는 정수 grid 조건 때문에 지원 해상도를 맞춰야 한다. 논문의 표현처럼 구조적 비율만 비교하면 FastViTHD는 같은 해상도에서 ViT-L/14보다 약 16배 적은 token을 만든다.

### Multi-scale feature aggregation

hierarchical encoder의 초기 stage는 작은 물체와 문자에 필요한 fine detail을 갖고 있지만 마지막 stage는 semantic representation이 강하다. FastVLM은 여러 stage feature를 learned pooling으로 마지막 spatial 크기에 맞춘 뒤 channel 방향으로 concatenate한다.

```math
F_{multi}=\operatorname{Concat}_C(
P_2(F_2),P_3(F_3),P_4(F_4),F_5)
```

논문의 FastViT ablation에서는 multi-scale 없이 Avg-5가 62.6, average pooling은 62.7, depthwise convolution pooling은 62.9다. 개선 폭은 작지만 depthwise learned pooling이 가장 좋다. 이 모듈은 token 수를 늘리지 않고 서로 다른 scale 정보를 channel에 합친다.

## Tensor shape와 activation memory

batch 1, 1024 x 1024 입력을 예로 들면 stage별 대략적인 spatial shape은 다음과 같다. 실제 stem 내부 tensor와 connector width는 구현 config를 확인해야 한다.

| 지점 | Shape | Element | FP16 payload |
| --- | ---: | ---: | ---: |
| input | `3 x 1024 x 1024` | 3,145,728 | 6.0 MiB |
| Stage 1 | `256 x 256 x 96` | 6,291,456 | 12.0 MiB |
| Stage 2 | `128 x 128 x 192` | 3,145,728 | 6.0 MiB |
| Stage 3 | `64 x 64 x 384` | 1,572,864 | 3.0 MiB |
| Stage 4 | `32 x 32 x 768` | 786,432 | 1.5 MiB |
| Stage 5 | `16 x 16 x 1536` | 393,216 | 0.75 MiB |
| visual sequence | `256 x D_t` | `256D_t` | `512D_t` bytes |

이는 reviewer가 논문 구조로 계산한 tensor 하나의 raw payload다. residual branch, QKV, MLP expansion, multi-scale pooling, runtime workspace를 포함하지 않는다. 중요한 관찰은 마지막 token tensor보다 초기 고해상도 convolution activation이 훨씬 크다는 것이다. visual token 수만으로 vision encoder peak memory를 추정하면 틀린다.

Stage 4의 self-attention은 1024 token, Stage 5는 256 token에서 작동한다. score element 수는 head당 각각 약 1.0M과 65,536이다. FlashAttention과 accelerator-specific attention kernel은 score matrix를 전부 외부 memory에 저장하지 않을 수 있지만, Stage 4가 attention memory의 주요 후보라는 점은 변하지 않는다.

## Visual token과 LLM prefill

text prompt가 `N_t`, visual token이 `N_v`이면 prefill sequence는 대략 `S=N_t+N_v`이다. decoder의 dense attention score 항은 `S^2`, linear projection과 MLP는 `S`에 비례한다.

예를 들어 reviewer가 `N_t=64`를 가정하면 다음과 같다.

| Input | `N_v` | `S` | Head당 dense score element |
| ---: | ---: | ---: | ---: |
| 256 | 16 | 80 | 6,400 |
| 768 | 144 | 208 | 43,264 |
| 1024 | 256 | 320 | 102,400 |
| 1536 | 576 | 640 | 409,600 |

1024에서 1536으로 높이면 pixel은 2.25배지만 이 예시의 attention score 항은 4배다. 반대로 1024에서 768로 낮추면 OCR detail은 줄 수 있지만 prefill은 크게 가벼워진다. 논문이 해상도와 LLM 크기를 함께 Pareto search하는 이유다.

KV cache는 decoder layer 수 `L`, KV dimension `D_KV`, element byte `b`일 때 다음과 같다.

```math
M_{KV}=2LS D_{KV}b
```

768에서 1024로 올릴 때 visual prefix가 112 token 증가하므로 증가량은 `2 x L x 112 x D_KV x b`다. Qwen2의 GQA에서는 `D_KV`가 `D_model`보다 작을 수 있으므로 config의 `num_key_value_heads`로 계산해야 한다. 논문은 peak memory와 KV cache byte를 직접 보고하지 않는다.

## 학습 목적함수와 단계

answer token `y_i`, image representation `H_v`, instruction `H_q`에 대해 다음 likelihood를 최대화한다.

```math
p(Y\mid H_v,H_q)=\prod_i p(y_i\mid H_v,H_q,y_{<i})
```

### 2-stage ablation recipe

| 항목 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| 데이터 | LLaVA-1.5 558K | LLaVA-1.5 665K |
| learning rate | `1e-3` | `2e-5` |
| batch | 256 | 128 |
| epoch | 1 | 1 |
| trainable | projector | full model |

Stage 1은 vision encoder와 LLM을 동결하고 connector만 맞춘다. Stage 2에서는 target resolution으로 바꾸고 vision encoder, projector, LLM을 모두 fine-tune한다. 이는 고해상도 adaptation에 중요하다.

### 확장된 4-stage recipe

- Stage 1: LLaVA-558K, connector warm-up
- Stage 1.5: ReCap-CC3M + ReCap-CC12M 15M, resolution scaling
- Stage 2: 1.1M, 6.5M, 11.9M 또는 12.5M visual instruction data
- Stage 3: MammothVL에서 고른 10.6M high-quality reasoning data

모든 FastVLM은 8 x H100-80GB 한 node에서 학습한다. 1024 설정은 Stage 1.5가 77시간, Stage 2가 8시간이며, Stage 1은 Qwen2-7B에서도 약 30분이다. 논문 표에서 `15+12.5`는 pretraining 15M과 instruction tuning 12.5M을 뜻하며 15B sample이 아니다.

## Forward pseudocode

```python
def fastvlm_forward(image, prompt_ids, answer_ids=None):
    # FastViTHD
    x = conv_stem(image)                       # stride 4
    f1 = repmixer_stage1(x)
    f2 = repmixer_stage2(patch_embed(f1))
    f3 = repmixer_stage3(patch_embed(f2))
    f4 = attention_stage4(patch_embed(f3))
    f5 = attention_stage5(patch_embed(f4))     # stride 64

    multi = concat_channels([
        learned_pool(f2, target_hw=f5.hw),
        learned_pool(f3, target_hw=f5.hw),
        learned_pool(f4, target_hw=f5.hw),
        f5,
    ])
    visual = connector(flatten_hw(multi))       # [B, Nv, Dllm]
    text = llm.embed_tokens(prompt_ids)
    prefix = concat([visual, text], dim=1)

    if answer_ids is None:
        return llm.generate(inputs_embeds=prefix)
    return causal_lm_loss(llm, prefix, answer_ids)
```

## Encoder ablation과 정량 결과

### FastViT 대 ViT-L/14

| Encoder | Input | Tokens | Encoder latency | GQA | TextVQA | POPE | DocVQA | Seed | Avg-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-L/14 | 336 | 576 | 127.4 ms | 62.0 | 58.2 | 85.9 | 28.1 | 66.1 | 60.1 |
| ViT-L/14, Stage 2 fine-tune | 336 | 576 | 127.4 ms | 63.5 | 59.2 | 86.3 | 28.7 | 68.6 | 61.2 |
| FastViT | 256 | 64 | 3.0 ms | 60.2 | 51.6 | 82.9 | 15.8 | 61.5 | 54.4 |
| FastViT | 768 | 576 | 34.5 ms | 62.7 | 62.3 | 86.5 | 34.4 | 67.1 | 62.6 |

같은 576 token에서 FastViT 768은 ViT-L/14 336보다 encoder latency가 약 3.7배 짧고 TextVQA와 DocVQA도 높다. higher pixel resolution과 hierarchical feature가 text-rich task에 유리하다는 근거다. 다만 encoder pretraining과 architecture가 함께 다르므로 해상도만의 효과는 아니다.

### FastViTHD resolution sweep

| Input | Encoder latency | Tokens | GQA | TextVQA | POPE | DocVQA | Seed | Avg-5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 10.1 ms | 16 | 60.6 | 53.1 | 82.3 | 17.4 | 63.7 | 55.5 |
| 512 | 33.5 ms | 64 | 63.0 | 59.3 | 86.4 | 25.7 | 67.1 | 60.4 |
| 768 | 122.6 ms | 144 | 62.4 | 62.9 | 87.7 | 32.9 | 68.2 | 62.8 |
| 1024 | 235.6 ms | 256 | 63.1 | 64.4 | 88.1 | 35.6 | 68.5 | 63.9 |

이 표의 encoder latency는 architecture comparison 조건이고, 뒤의 Core ML breakdown 표와 값이 다르다. 서로 다른 표의 latency를 임의로 교체하면 안 된다. 정확도는 해상도가 오를수록 대체로 좋아지지만 GQA는 단조 증가하지 않는다. 768에서 1024로 112 token과 약 113ms encoder latency를 추가해 Avg-5는 1.1 point 오른다.

### Token pruning 비교

같은 LLaVA-1.5와 Vicuna-7B 조건에서 FastViTHD 256은 16 token으로 GQA 60.6, SQA 69.2, TextVQA 53.1, POPE 82.3, VQAv2 74.7, Seed 58.8을 기록한다. FastViTHD 512의 64 token은 GQA 63.0, TextVQA 59.3, POPE 86.4다. 논문 표에서 여러 ViT pruning method보다 좋은 accuracy-token trade-off를 보이지만, 일부 비교 수치는 원 논문에서 가져온 것이며 동일 runtime implementation은 아니다.

## 대표 FastVLM 결과

논문의 전체 표는 데이터 규모와 decoder가 매우 다양하다. 같은 행 안의 조건을 유지해 읽어야 한다.

| Model setting | Input | Tokens | Vision params | TTFT | GQA | SQA | TextVQA | POPE | DocVQA | MMMU | Seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2-0.5B, 15M+1.1M | 1024 | 256 | 125M | 166 ms | 61.6 | 61.4 | 57.4 | 87.4 | 61.0 | 30.9 | 65.6 |
| Qwen2-0.5B, 15M+12.5M | 1024 | 256 | 125M | 166 ms | 63.1 | 81.5 | 62.9 | 86.6 | 70.4 | 32.9 | 69.2 |
| Qwen2-1.5B, 15M+12.5M | 1024 | 256 | 125M | 233 ms | 64.3 | 89.9 | 69.0 | 87.8 | 75.6 | 39.1 | 73.1 |
| Vicuna-7B, 15M+1.1M | 768 | 144 | 125M | 387 ms | 65.0 | 78.7 | 69.4 | 87.5 | 65.5 | 37.0 | 73.7 |
| Qwen2-7B, 15M+1.1M | 768 | 144 | 125M | 446 ms | 65.6 | 85.9 | 69.5 | 87.2 | 66.9 | 43.6 | 75.3 |
| Qwen2-7B, 15M+6.5M | 1024 | 256 | 125M | 641 ms | 66.0 | 87.4 | 73.1 | 87.3 | 78.7 | 42.8 | 75.9 |

FastVLM 0.5B 설정은 같은 Qwen2-0.5B를 사용하는 LLaVA-OneVision의 최고 해상도 설정과 비교해 TTFT 166ms 대 14,124ms로 약 85배 빠르다. 그러나 두 모델의 visual token은 256 대 7,290이고 architecture 및 resolution strategy가 크게 다르다. 이는 동일 품질에서 vision encoder만 바꾼 순수 ablation이 아니라 system-level 비교다.

## On-device 측정의 정확한 의미

논문은 MacBook Pro M1 Max, 32GB에서 다음처럼 측정한다.

- vision encoder: `coremltools 7.2`로 변환, Xcode 15.4, Apple Neural Engine
- LLM: Hugging Face checkpoint를 MLX FP16으로 변환, Mac GPU
- prefill: `mlx_lm.cache_prompt`
- TTFT: vision encoder latency와 visual token 수에 해당하는 LLM prefill latency를 더한 추정값

따라서 표의 TTFT는 한 process에서 camera input부터 첫 token까지 측정한 end-to-end wall-clock trace가 아니다. 서로 다른 execution backend의 latency 합이다. 그럼에도 실제 hardware에서 두 핵심 항을 측정했다는 점은 FLOPs 기반 추정보다 강하다.

### Latency breakdown

| Setting | Input | Tokens | Vision latency | LLM prefill | 합 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2-0.5B | 1024 | 256 | 116.3 ms | 50.5 ms | 166.8 ms |
| Qwen2-1.5B | 768 | 144 | 54.8 ms | 97.1 ms | 151.9 ms |
| Qwen2-1.5B | 1024 | 256 | 116.3 ms | 116.1 ms | 232.4 ms |
| Vicuna-7B | 256 | 16 | 6.8 ms | 143.4 ms | 150.2 ms |
| Vicuna-7B | 768 | 144 | 54.8 ms | 332.1 ms | 386.9 ms |
| Vicuna-7B | 1024 | 256 | 116.3 ms | 461.1 ms | 577.4 ms |
| Qwen2-7B | 1024 | 256 | 116.3 ms | 524.5 ms | 640.8 ms |

작은 decoder에서는 vision latency 비중이 크고, 7B decoder에서는 prefill이 지배적이다. 입력 해상도가 더 높아지면 vision latency가 급격히 커져 다시 vision encoder가 지배한다. 하나의 병목만 최적화해서는 모든 operating point를 개선할 수 없다.

### Dynamic resolution

1024 tile을 2 x 2로 사용하는 2048 입력은 1,280 visual token을 만든다. 이는 4개 tile의 `4 x 256`에 global view 256을 더한 것으로 해석할 수 있다. 논문 표에서 Qwen2-1.5B의 vision latency는 581.5ms, prefill은 681.7ms, TTFT는 약 1.263초다. Qwen2-7B에서는 3.721초다. static resolution이 대체로 더 좋은 latency-accuracy trade-off이고, 1536 같은 극단적 해상도에서만 tiling이 memory bandwidth 측면에서 이점이 나타났다.

## 보고되지 않은 시스템 지표

- smartphone iPhone 실측
- decode tokens/s
- end-to-end TTFT의 single trace
- projector latency
- peak unified memory와 allocator high-water mark
- p50, p95, cold-start latency
- model load와 graph compilation 시간
- power, energy, temperature, throttling
- Core ML precision과 layer별 CPU/GPU fallback 상세

따라서 FastVLM이 TTFT를 개선했다는 결론은 강하지만, 지속 생성 속도나 thermal stability까지 빠르다고 확대 해석할 수 없다.

## Mobile operator와 배포 분석

FastViTHD는 mobile accelerator에 익숙한 연산을 많이 사용한다.

- depthwise 7 x 7 convolution
- pointwise 1 x 1 convolution
- RepMixer의 spatial mixing
- GELU와 normalization
- 후기 stage의 MHA
- learned pooling, concat, projection

주의할 지점은 다음과 같다.

1. 7 x 7 depthwise kernel이 target NPU에서 native인지 작은 kernel로 분해되는지 확인한다.
2. train-time overparameterization branch는 export 전에 반드시 fuse한다.
3. RepMixer와 attention stage 사이 NHWC/NCHW transpose가 생기는지 확인한다.
4. multi-scale pooling 뒤 channel concat이 큰 memory copy를 만드는지 측정한다.
5. Stage 4 attention의 1024-token QKV가 NPU SRAM에 맞는지 확인한다.
6. Core ML graph 일부가 CPU로 fallback하면 평균 latency와 전력이 크게 달라질 수 있다.
7. vision encoder와 LLM이 NPU 및 GPU를 순차 사용하므로 resource handoff와 synchronization 비용을 end-to-end에서 재측정한다.

## 강점

1. vision latency와 LLM prefill을 하나의 TTFT budget으로 함께 최적화한다.
2. 해상도만으로 token budget을 조절해 pruning module과 irregular gather를 피한다.
3. 16, 64, 144, 256, 576 token의 명확한 operating point를 제공한다.
4. Apple Neural Engine과 MLX에서 실제 latency를 측정하고 breakdown을 공개한다.
5. model size, resolution, LLM size, dataset scale의 상호작용을 폭넓게 비교한다.
6. text-rich task의 고해상도 이득과 tiling의 semantic break 문제를 함께 논의한다.

## 한계와 비판적 해석

### 1. M1 Max는 smartphone이 아니다

Neural Engine 계열이라는 공통점은 있지만 memory capacity, thermal budget, GPU 성능이 iPhone과 다르다. 논문 수치로 phone latency를 보장할 수 없다.

### 2. TTFT는 두 benchmark의 합이다

vision Core ML과 LLM MLX를 따로 측정한 뒤 더했다. 실제 앱에는 image decode, preprocessing, tensor copy, command submission, tokenizer, first decode가 추가된다.

### 3. 고해상도 activation peak가 충분히 분석되지 않았다

token은 적지만 1024 입력의 초기 feature map은 크다. peak memory를 보고하지 않아 12GB 이하 phone에서의 안정성을 판단하기 어렵다.

### 4. 데이터 규모가 결과에 큰 영향을 준다

같은 architecture도 instruction data 1.1M, 12.5M, 추가 Stage 3에 따라 benchmark가 크게 변한다. 다른 모델과 비교할 때 encoder 효율과 데이터 이득을 구분해야 한다.

### 5. 작은 문자에는 여전히 resolution 비용이 든다

hierarchical token은 적지만 정보가 공짜로 보존되는 것은 아니다. 256 input의 DocVQA 17.4가 1024에서 35.6으로 오르는 결과는 extreme compression만으로 OCR을 해결할 수 없음을 보여 준다.

## 재현 체크리스트

- [ ] FastViTHD depth `[2,12,24,4,2]`, width `[96,192,384,768,1536]` 확인
- [ ] stem stride 4, patch embedding stride 2, output stride 64 확인
- [ ] reparameterization branch export 전 fuse
- [ ] multi-scale learned depthwise pooling과 concat order 일치
- [ ] DataCompDR-1B 기반 vision pretraining checkpoint 확인
- [ ] Stage 1과 Stage 2 trainable module 일치
- [ ] target resolution에서 vision encoder fine-tuning
- [ ] conversation template, tokenizer, image special token 일치
- [ ] evaluation library `lmms-eval 0.2.2`와 judge version 기록
- [ ] Core ML 7.2, Xcode 15.4, compute unit 고정
- [ ] MLX FP16 conversion과 prefill command 고정
- [ ] warm-up, run count, p50/p95, cold start 분리
- [ ] visual token 수를 runtime tensor에서 직접 확인
- [ ] encoder, connector, prefill, first decode, steady decode 분리 측정
- [ ] peak memory, 전력, 온도와 10분 sustained test 기록

## 추천 온디바이스 실험

같은 FastVLM checkpoint로 `256, 512, 768, 1024`를 target phone에서 비교한다. output은 32 token으로 고정하고 다음을 기록한다.

| 축 | 지표 |
| --- | --- |
| 품질 | GQA, TextVQA, DocVQA, 앱별 validation |
| 응답성 | cold/warm TTFT, p50, p95 |
| 처리량 | decode tokens/s |
| memory | encoder peak, total peak, KV cache |
| 에너지 | joule/query, average power |
| 안정성 | 1회와 50회 latency, 온도, throttling |

추가로 Qwen2-0.5B와 1.5B를 같은 1024 input에서 비교하면 vision과 decoder budget의 교차점을 볼 수 있다. 작은 LLM에서는 116.3ms vision latency가 큰 비중이므로 encoder 최적화가 중요하고, 큰 LLM에서는 visual token과 prefill 최적화가 더 중요하다.

## Pareto operating point를 고르는 방법

논문의 결론은 항상 가장 높은 해상도를 선택하라는 것이 아니다. target 장치에서 허용하는 TTFT를 `T_max`, 최소 품질을 `A_min`이라 하고 해상도 `r`와 LLM 크기 `l`의 조합을 탐색한다.

```math
(r^*,l^*)=
\arg\max_{r,l} A(r,l)
\quad\text{s.t.}\quad
T_{vision}(r)+T_{prefill}(N_v(r),l)\le T_{max}
```

실무에서는 다음 순서가 유용하다.

1. decoder를 고정하고 resolution sweep으로 OCR/detail gain의 포화점을 찾는다.
2. resolution을 고정하고 LLM size sweep으로 reasoning gain을 찾는다.
3. dominated point를 제거해 Pareto frontier만 남긴다.
4. 평균 TTFT가 아니라 p95와 thermal steady-state를 constraint로 쓴다.
5. app query를 document, natural image, chart로 나누어 서로 다른 operating point를 허용한다.

예를 들어 논문 수치에서 Qwen2-1.5B는 768에서 vision 54.8ms와 prefill 97.1ms, 1024에서 116.3ms와 116.1ms다. 해상도 증가는 vision에 61.5ms, prefill에 19.0ms를 더한다. target UX budget이 200ms라면 768은 들어가고 1024는 넘는다. 이때 1024의 benchmark 이득이 꼭 필요한 document query에만 선택적으로 쓰는 것이 합리적이다. 수치는 M1 benchmark이므로 phone에서는 frontier를 다시 측정해야 한다.

## Peak memory를 재는 구체적 방법

FastViTHD는 visual token이 적어도 early activation이 크다. 단일 process의 total peak만 재면 어느 module이 원인인지 알기 어렵다. 다음 네 구간을 marker로 분리한다.

```text
preprocess
 -> vision encoder
 -> multi-scale connector
 -> LLM prefill
 -> autoregressive decode
```

각 경계에서 allocator current와 high-water mark를 읽고, model weight load 전후도 별도로 기록한다. unified memory 장치에서는 RSS만으로 NPU private buffer를 놓칠 수 있으므로 OS memory report와 runtime profiler를 함께 쓴다. resolution sweep에서는 다음 값이 단조롭게 움직이는지 확인한다.

- input 및 Stage 1 activation: 대략 pixel area에 비례
- Stage 4 attention: token square 항의 영향
- connector output: visual token에 선형 비례
- prefill activation과 KV cache: visual token 및 LLM config에 비례

static arena를 쓰는 mobile runtime은 여러 shape 중 최대 buffer를 미리 할당할 수 있다. 이 경우 256 inference 후에도 1024 graph용 arena가 resident라면 measured peak가 resolution에 따라 줄지 않는다. resolution별 process restart 결과와 한 process dynamic-switch 결과를 모두 기록해야 한다.

## Dynamic resolution을 쓸 때 생기는 숨은 비용

2 x 2 AnyRes는 단순히 four independent encoder calls만 뜻하지 않는다. tile crop과 resize, global view 생성, positional handling, token concat, tile separator가 추가된다. 서로 다른 tile의 중복 영역과 semantic break도 생길 수 있다. 2048 설정의 1,280 token은 1024 static의 256보다 5배이며 Qwen2-1.5B prefill이 116.1ms에서 681.7ms로 약 5.9배 증가한다. vision latency도 116.3ms에서 581.5ms로 정확히 5배다.

이는 global view를 포함한 5회 1024 encoding과 잘 맞는다. 실제 pipeline에서 tile을 batch 5로 묶으면 wall-clock과 peak memory trade-off가 달라진다. 순차 실행은 memory가 작지만 latency가 길고, batch 실행은 accelerator utilization이 좋을 수 있지만 activation peak가 커진다. 논문 표는 이 scheduling 선택을 세부적으로 보고하지 않는다.

## 실패 사례를 수집하는 기준

정량 평균 외에 다음 bucket으로 failure를 모으면 resolution과 LLM 크기의 역할을 분리할 수 있다.

- 작은 글자 자체를 못 읽음: vision resolution 또는 encoder 문제
- 글자는 읽지만 table row/column 연결 실패: spatial representation 문제
- visual fact는 맞지만 reasoning 실패: LLM size 또는 instruction data 문제
- tiling 경계에서 object가 분리됨: AnyRes semantic break
- 비슷한 숫자를 바꿈: OCR 및 hallucination 문제
- 긴 answer에서만 느림: decode throughput 문제, TTFT와 별개

이 분류를 사용하면 무조건 resolution을 높이거나 LLM을 키우는 대신 정확한 module을 개선할 수 있다.

## 최종 평가

FastVLM의 핵심 가치는 "고해상도 VLM은 큰 ViT 뒤에서 token을 잘라야 한다"는 관행을 다시 묻는 데 있다. FastViTHD는 output stride 64의 hierarchy로 1024 이미지를 256 token에 담고, FastVLM은 이 token 수와 decoder 크기를 실제 TTFT budget 위에서 고른다. M1 Max에서 Qwen2-0.5B 설정은 116.3ms vision encoding과 50.5ms prefill로 약 166ms를 기록하며, LLaVA-OneVision의 비교 설정보다 85배 짧다.

다만 이 수치는 phone end-to-end trace가 아니고 peak memory, decode throughput, energy도 없다. 실제 온디바이스 연구에서는 FastViTHD의 적은 visual token뿐 아니라 초기 convolution activation, Core ML fallback, NPU-GPU handoff까지 측정해야 한다. 그 조건을 지킨다면 FastVLM은 고해상도 VLM을 위한 매우 강한 latency-first baseline이다.

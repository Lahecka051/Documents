# 87. Efficient Deployment of Vision-Language Models on Mobile Devices

## 논문 정보

- 원본 파일: `87_OnePlus_13R_VLM_Deployment.pdf`
- 제목: **Efficient Deployment of Vision-Language Models on Mobile Devices: A Case Study on OnePlus 13R**
- 저자: Pablo Robin Guerrero, Yueyang Pan, Sanidhya Kashyap
- 소속: EPFL
- 공개: arXiv 2025, v2 2025-07-14
- 공식 링크: [arXiv:2507.08505](https://arxiv.org/abs/2507.08505)
- 주제: mobile VLM deployment, heterogeneous compute, runtime profiling, energy and thermal behavior
- 비교 대상: LLaVA-1.5 7B, MobileVLM-3B, Imp-v1.5 3B, llama.cpp, mllm, MLC-Imp

## 한눈에 보는 요약

이 논문은 VLM architecture 자체보다 **같은 모델이라도 mobile runtime이 CPU, GPU, NPU에 일을 어떻게 배치하느냐에 따라 latency, power, temperature, 사용자 반응성이 크게 달라진다**는 점을 OnePlus 13R 실험으로 보여준다.

분석하는 pipeline은 다음 네 단계다.

```text
load
  -> model/input preparation

slice-encode / image encode
  -> resize, crop/slice, vision encoder

prompt-eval
  -> text + visual token prefill, KV cache 생성

token-eval
  -> autoregressive decode, one token at a time
```

대표 관찰은 다음과 같다.

- llama.cpp와 mllm의 LLaVA-1.5 7B, llama.cpp의 MobileVLM-3B는 거의 전부 CPU-bound다.
- Aggregate CPU utilization이 500-800%까지 올라가지만 GPU는 0%, NPU는 모든 실험에서 미사용이다.
- CPU-only LLaVA는 논문 측정에서 약 10-12W, die temperature 80-95도에 도달한다.
- MLC-Imp의 Imp-v1.5 3B는 Adreno GPU를 90-100% 사용해 CPU load, power, temperature를 크게 낮춘다.
- GPU를 쓴다고 자동으로 latency가 짧아지는 것은 아니다. Imp는 image encoding에 약 18초가 걸리고 decode는 여전히 sequential하다.

가장 중요한 결론은 `작은 모델 선택`과 `양자화`만으로는 충분하지 않다는 것이다. Vision encoder, prefill GEMM, decode GEMV, KV cache, UI compositor를 서로 다른 hardware unit에 배치하고, 실제 operator support와 scheduling을 측정해야 한다.

다만 이 논문에는 재현성과 수치 해석에 영향을 주는 내부 불일치가 많다. Snapdragon 8 Gen 2와 8 Gen 3, 12GB와 16GB, 세 framework와 네 framework, stage 합계와 total wall-clock, token count와 answer length가 서로 맞지 않는다. 따라서 방향성 있는 case study로는 유용하지만, 표의 절대값을 framework 순위로 일반화하기 전에 원시 log와 코드 확인이 필요하다.

## 연구 질문

논문은 다음 질문에 답하려 한다.

1. 현재 mobile VLM runtime은 CPU, GPU, NPU를 실제로 얼마나 사용하는가?
2. Model과 runtime 조합에 따라 end-to-end latency가 어떻게 달라지는가?
3. Power와 thermal behavior가 사용자 경험에 어떤 영향을 주는가?
4. Mobile deployment에서 다음 최적화 우선순위는 무엇인가?

기존 VLM 논문은 정확도와 GPU throughput을 주로 보고한다. 하지만 smartphone에서는 다음 제약이 동시에 존재한다.

- shared memory capacity와 bandwidth
- CPU/GPU/NPU operator coverage
- DVFS와 thermal throttling
- battery power budget
- display compositor와 inference GPU의 resource contention
- app lifecycle 및 OS memory pressure

이 논문은 model FLOPs보다 실제 system trace를 중심으로 이 간극을 살핀다.

## 평가한 Model

### LLaVA-1.5 7B

강한 7B language model과 vision encoder를 연결한 범용 baseline이다. 세 모델 중 가장 크고 응답도 길다. llama.cpp와 mllm 두 runtime에서 비교한다.

### MobileVLM-3B

Mobile deployment를 겨냥한 3B급 VLM이다. llama.cpp로 실행한다. Parameter 수는 LLaVA보다 작지만 vision preprocessing과 CPU-only prefill/decode가 여전히 길다.

### Imp-v1.5 3B

LLaVA-1.5에서 distillation된 3B model로, Phi-2 language core와 SigLIP vision encoder를 사용한다. MLC-Imp runtime으로 실행하며 이 조합만 Adreno GPU를 적극 사용한다.

Model 크기 비교가 완전한 통제 실험은 아니다. Vision encoder, visual token 수, language architecture, quantization, prompt template, runtime이 동시에 다르다. `3B 대 3B` 결과도 model 효율뿐 아니라 framework와 graph lowering 차이를 포함한다.

## 평가한 Runtime

### llama.cpp

GGUF와 quantized LLM inference 생태계가 강하다. 이 실험에서는 LLaVA-1.5 7B와 MobileVLM-3B가 CPU 중심으로 실행된다. GPU/NPU offload가 실질적으로 관찰되지 않는다.

### mllm

On-device multimodal inference를 목표로 한다. LLaVA-1.5 7B에서 image encoding 자체는 매우 짧게 보고되지만 prompt evaluation과 token generation이 극단적으로 길다. Runtime 내부 throughput 표시와 wall-clock 사이에도 큰 차이가 있다.

### MLC-Imp

Imp model 전용 MLC 계열 배포 경로다. Vision과 language kernel 상당 부분을 Adreno GPU로 offload해 낮은 CPU load와 낮은 power를 보인다.

논문은 여러 곳에서 `four frameworks`라고 쓰지만 실제로 이름을 든 framework는 llama.cpp, mllm, MLC-Imp 세 개다. 네 개라고 셀 수 있는 것은 model-runtime **stack** 네 조합이다.

```text
1. LLaVA + llama.cpp
2. LLaVA + mllm
3. MobileVLM + llama.cpp
4. Imp + MLC-Imp
```

Framework 수와 deployment stack 수를 구분해야 한다.

## Device와 Runtime 환경

Methodology section이 명시한 장치는 다음과 같다.

- OnePlus 13R
- Android 15
- Qualcomm Snapdragon 8 Gen 2
- octa-core Kryo CPU
- Adreno 740 GPU
- Hexagon NPU
- 12GB LPDDR5X RAM
- 256GB UFS 4.0 storage

Root access를 통해 `/proc`, `/sys`, vendor sensor를 읽고 native Android build를 실행한다. Background process를 최소화하고 airplane mode, 고정 screen brightness 조건에서 실험한다.

그러나 같은 PDF의 limitation과 conclusion은 장치를 `Snapdragon 8 Gen 3-based handset`라고 부르고, 결과는 memory percentage를 `16GB pool` 기준으로 환산한다. 즉 SoC generation/GPU 표기와 RAM 용량이 한 논문 안에서 일치하지 않는다.

이것은 사소한 오타가 아니다.

- Gen 2와 Gen 3는 CPU/GPU/NPU 성능과 power behavior가 다르다.
- Adreno model과 driver/compiler 지원이 달라진다.
- 12GB와 16GB는 peak memory percentage를 GB로 환산할 때 33% 차이다.

재현 전에 `adb shell getprop`, `/proc/cpuinfo`, GPU driver, RAM total, exact device SKU를 raw log에서 확인해야 한다.

## 측정 방법

각 model-runtime 조합을 standard image와 prompt로 5회 반복하고 평균을 보고한다. System monitoring은 100ms interval이다.

### Metrics

| Metric | 의미 |
|---|---|
| inference latency | image encode, prompt evaluation, token generation 등 단계별 시간 |
| CPU/GPU/NPU utilization | workload가 어느 compute unit에 배치됐는지 |
| power | 순간 및 평균 소비 전력, battery drain |
| thermal | die 또는 surface temperature의 시간 변화 |
| user responsiveness | UI lag, screen freeze, shell blocking |

### Trace synchronization

Monitoring script timestamp와 screen recording을 맞추고 phase를 plot에 shade한다. Root-readable path에서 CPU, GPU, NPU utilization과 power/thermal sensor를 수집한다.

### Sampling interval의 한계

100ms sampling은 80-180초 pipeline의 큰 구간을 보기에는 충분하다. 하지만 mllm image encoding `38.5ms`처럼 sample interval보다 짧은 event는 정확히 포착할 수 없다. Short GPU burst의 peak utilization과 energy를 놓칠 수 있으며, phase boundary 정렬 오차도 상대적으로 커진다.

Power sensor는 rail current인지 battery current인지, voltage를 어떤 시점 값으로 곱했는지, package power인지 system power인지 명확히 구분해야 한다. 논문은 여러 표현을 섞어 사용해 절대값 신뢰도를 낮춘다.

## VLM의 단계별 Tensor 흐름

일반적인 single-image VLM을 다음 shape로 생각할 수 있다.

```text
image: [1, 3, H, W]
  -> vision encoder
  -> image tokens: [1, N_v, d_v]
  -> projector
  -> visual tokens: [1, N_v, d_l]

text prompt ids: [1, N_t]
  -> token embeddings: [1, N_t, d_l]

concat:
  [1, N_v + N_t, d_l]
  -> LLM prefill
  -> KV cache for every layer
  -> autoregressive decode [1 token/step]
```

Image encoding과 prompt evaluation은 많은 token을 한 번에 처리해 matrix-matrix multiplication, 즉 GEMM 성격이 강하다. GPU/NPU가 병렬성을 활용하기 쉽다.

Decode는 보통 batch 1, sequence dimension 1의 matrix-vector multiplication, 즉 GEMV 비중이 크다. Weight를 매 token 읽어야 해 arithmetic intensity가 낮고 memory bandwidth-bound가 되기 쉽다. NPU는 큰 batch/regular tensor에서 효율이 높아도 single-token dynamic decode에서는 utilization이 낮을 수 있다.

이 차이가 `prefill은 accelerator, decode는 CPU`라는 runtime 선택을 만든다. 하지만 CPU decode가 항상 최선이라는 뜻은 아니다. GPU kernel launch, weight residency, quantized GEMV, fused sampling을 함께 최적화하면 손익분기점이 바뀐다.

## Latency Table

논문 Table 2를 그대로 정리하면 다음과 같다. 단위는 ms다.

| Stage | LLaVA llama.cpp | LLaVA mllm | MobileVLM llama.cpp | Imp MLC-Imp |
|---|---:|---:|---:|---:|
| model load | 3,209.1 | 4,479.7 | 2,334.4 | 4,000.0 |
| image encoding | 2,399.0 | 38.5 | 3,053.0 | 18,000.0 |
| image decoding | 63,724.0 | 미노출 | 9,989.0 | 미노출 |
| prompt evaluation | 70,391.4 | 78,901.5 | 15,300.2 | 2,000.0 |
| token generation | 11,085.8 | 90,249.0 | 6,605.2 | 1,000.0 |
| reported total | 82,194.9 | 173,668.7 | 22,799.1 | 25,000.0 |

Table caption은 llama.cpp total을 binary가 출력한 wall-clock total이라고 설명한다. 각 stage가 동일한 exclusive 구간이라면 합과 total이 같아야 하지만 그렇지 않다.

### LLaVA llama.cpp

```math
3.209+2.399+63.724+70.391+11.086
=150.809\text{ s}
```

Reported total은 `82.195s`다. Stage 중 일부가 nested timing이거나 같은 work를 중복 계상했을 가능성이 있다. `image decoding 63.724s`와 `prompt evaluation 70.391s`가 겹치는지 명확한 설명이 필요하다.

### MobileVLM llama.cpp

```math
2.334+3.053+9.989+15.300+6.605
=37.281\text{ s}
```

Reported total은 `22.799s`다. 본문 다른 문장은 end-to-end를 약 35초라고도 한다. 따라서 Table total만으로 stage percentage를 계산하면 안 된다.

### Imp MLC-Imp

```math
4+18+2+1=25\text{ s}
```

이 행만 합과 total이 일치한다.

이 불일치는 논문의 핵심 결과에 직접 관련된다. 재현에서는 exclusive wall-clock phase, runtime 내부 counter, overlap된 asynchronous phase를 별도 column으로 저장해야 한다.

## Token Throughput 재계산

본문의 token count와 Table 2 시간을 이용해 계산한 reviewer estimate다.

### LLaVA llama.cpp decode

본문은 69 tokens, 11.8초, 171.7ms/token이라고 서술한다. Table 시간 11.0858초를 쓰면 다음과 같다.

```math
\frac{69}{11.0858}\approx6.22\text{ tokens/s}
```

본문의 171.7ms/token은 약 5.82 tokens/s다. 사용한 run이 다를 수 있다.

### LLaVA mllm decode

본문은 51 tokens, 90.249초라고 한다.

```math
\frac{51}{90.249}\approx0.565\text{ tokens/s}
```

같은 model인데 llama.cpp보다 약 11배 느리다. Runtime internal throughput `4,858 tokens/s` 표시는 wall-clock과 물리적으로 맞지 않아 counter 정의나 단위 오류를 의심해야 한다.

### MobileVLM과 Imp

Table 3의 answer length가 실제 generated token 수라고 가정하면 다음과 같다.

```math
\text{MobileVLM}: \frac{18}{6.6052}\approx2.73\text{ tokens/s}
```

```math
\text{Imp}: \frac{11}{1.0}\approx11\text{ tokens/s}
```

그러나 Table 3은 LLaVA answer length를 30 tokens로 적는 반면 latency 본문은 69 또는 51 tokens라고 적는다. Tokenizer가 다르거나 measured generation과 qualitative answer가 다른 run일 수 있다. Exact prompt, stop condition, tokenizer count를 맞추지 않으면 tokens/s 비교가 성립하지 않는다.

## TTFT와 End-to-End Latency

사용자가 느끼는 첫 응답 시간은 대략 다음이다.

```math
TTFT=T_{load}+T_{image}+T_{projector}+T_{prefill}+T_{first\ decode}
```

Model이 이미 resident면 `T_load`를 제외한 warm TTFT도 따로 측정해야 한다. 논문은 load와 prompt stage를 제공하지만 stage 중첩과 total 불일치 때문에 정확한 cold/warm TTFT를 복원하기 어렵다.

Interactive app에는 다음 세 값이 필요하다.

- cold TTFT: app 시작과 model load 포함
- warm TTFT: model resident, 새 image와 prompt
- decode throughput: 첫 token 이후 tokens/s

MobileVLM의 reported total 22.8초도 실시간 대화라고 보기 어렵고, LLaVA 82-174초는 사용자 timeout 범위다. Output이 짧아지면 total이 좋아 보일 수 있으므로 fixed output token 수와 semantic completion 조건을 모두 보고해야 한다.

## Hardware Utilization

### CPU percentage 읽는 법

Android profiler에서 100%는 보통 core 하나의 full utilization이다. 따라서 aggregate 600%는 8-core CPU 중 약 6 core가 완전히 바쁜 수준, 800%는 8 core 전체가 포화된 수준이다.

### LLaVA + llama.cpp

- CPU 약 600%
- Adreno GPU 0%
- Hexagon NPU 미사용
- die 약 88-90도 plateau
- battery current 약 3.1A, 약 10W로 환산
- memory 약 55%라고 보고

Prompt evaluation 약 80초와 decode 약 10초 동안 CPU가 지속적으로 높다.

### LLaVA + mllm

- CPU 약 650-750%
- GPU 0%, NPU 미사용
- die 약 92-95도
- 약 2.9A, 11W
- 약 180초 run

같은 model인데 더 많은 CPU time과 높은 temperature를 쓰면서 latency도 길다. Thread affinity, kernel implementation, tensor layout, scheduling overhead가 architecture만큼 중요하다는 사례다.

### MobileVLM + llama.cpp

- CPU 500-800%
- GPU 0%
- die 70-72도
- memory 32% of 16GB라고 서술, 약 5.1GB
- end-to-end 약 35초라고 trace 본문에서 서술

LLaVA보다 작아 thermal은 낮지만 accelerator를 쓰지 않는 구조는 같다. 논문은 rail current 약 57mA라는 매우 낮은 값도 적는데, 다른 부분의 MobileVLM power 약 3.5W와 쉽게 양립하지 않는다. Sensor rail이 전체 battery current가 아니거나 단위/채널이 다른지 확인해야 한다.

### Imp + MLC-Imp

- GPU image encoding 중 90-100%
- CPU 최고 약 120%
- die 약 60도
- battery current 약 0.33A, 약 1.3W
- memory 3% 미만 of 16GB라고 보고
- total 약 25초

GPU offload가 CPU, temperature, power를 크게 줄인다. 그러나 18초 image encoding 동안 GPU가 포화되고, display compositor도 같은 GPU를 쓰므로 UI responsiveness가 나빠질 수 있다.

## NPU가 사용되지 않은 이유

논문 모든 deployment에서 Hexagon NPU는 사용되지 않는다. 이것이 NPU가 VLM에 본질적으로 쓸모없다는 뜻은 아니다. 실제 장벽은 다음과 같다.

- Runtime이 Hexagon backend를 지원하지 않음
- Quantized graph 변환과 calibration 부재
- Dynamic shape, RoPE, attention, KV cache update operator 미지원
- CPU/GPU/NPU 사이 tensor copy 비용
- Decode GEMV의 낮은 arithmetic intensity
- Per-token graph launch overhead

Vision encoder와 prompt prefill은 큰 GEMM이므로 NPU에 더 적합할 수 있다. Decode는 weight-stationary INT8/INT4 kernel과 persistent graph가 없으면 CPU가 나을 수 있다. NPU utilization 0%는 hardware capability보다 software stack gap을 보여준다.

## GPU 포화와 사용자 반응성

GPU inference가 100%면 compute efficiency는 높아 보이지만 smartphone GPU는 UI rendering과 공유된다.

```text
VLM kernels ----+
                +-> Adreno scheduler -> display compositor deadline
UI rendering ---+
```

Long-running kernel이나 높은 queue occupancy가 frame deadline을 놓치면 screen freeze와 touch response 지연이 생긴다. 논문은 GPU image processing 시 device instability와 screen freezing을 관찰했다고 설명한다.

따라서 mobile accelerator 목표는 100% utilization이 아니라 다음을 만족하는 것이다.

- inference latency 최소화
- UI frame deadline 보장
- power/thermal budget 유지
- foreground/background priority 반영

GPU command queue priority, chunked kernels, periodic yield, frame-aware scheduling을 함께 설계해야 한다.

## Power와 Energy 해석

Power는 순간 소비율이고 energy는 시간 적분이다.

```math
E=\int P(t)dt
```

Reported representative power를 전체 wall-clock 평균이라고 가정한 매우 거친 reviewer estimate는 다음과 같다. 실제로는 peak인지 average인지 불명확하므로 검증용 계산일 뿐이다.

### LLaVA llama.cpp

```math
E\approx10\text{ W}\times82.2\text{ s}
=822\text{ J}\approx0.228\text{ Wh/query}
```

### LLaVA mllm

```math
E\approx11\text{ W}\times173.7\text{ s}
=1911\text{ J}\approx0.531\text{ Wh/query}
```

### Imp MLC-Imp

```math
E\approx1.3\text{ W}\times25\text{ s}
=32.5\text{ J}\approx0.0090\text{ Wh/query}
```

이 단순 계산은 Imp가 query당 energy에서도 훨씬 유리할 가능성을 보여준다. 하지만 LLaVA query가 60초보다 길어 `1분에 한 query`라는 duty-cycle 가정 자체가 충돌한다. 논문의 `CPU-only 약 8시간, GPU path 거의 이틀` battery-life 추정은 battery capacity, idle power, query overlap, peak/average power 산식을 공개하지 않아 재현하기 어렵다.

Energy report는 다음 형식이 더 낫다.

```text
idle-subtracted joules/query
joules/prefill token
joules/generated token
average and peak power
battery voltage/current sampling source
```

## Thermal Behavior

CPU-only LLaVA는 15초 안에 88-95도 plateau에 도달한다고 보고한다. 이 값이 SoC die sensor라면 surface temperature와 다르다. 사용자 안전과 comfort에는 skin temperature가 중요하고, throttling 분석에는 CPU/GPU junction temperature와 frequency trace가 필요하다.

현재 논문에는 temperature와 utilization은 있지만 다음이 부족하다.

- core별 frequency와 governor state
- thermal throttling event
- ambient/initial temperature
- case 유무
- consecutive query 수
- surface temperature 위치

한 번의 25-180초 run이 아니라 15-30분 soak test를 해야 sustained performance를 알 수 있다.

## Memory와 KV Cache

VLM runtime memory는 대략 다음으로 나뉜다.

```math
M_{peak}=M_{weights}+M_{KV}+M_{vision\ act}+M_{LLM\ workspace}+M_{runtime}
```

Batch 1 KV cache는 다음에 비례한다.

```math
M_{KV}=2LTH_{kv}d_hs
```

- `L`: decoder layers
- `T`: visual + text + generated context length
- `H_kv`: KV heads
- `d_h`: head dimension
- `s`: bytes/element
- 앞의 2: key와 value

Visual token 수가 많으면 prompt-eval latency뿐 아니라 모든 이후 decode에서 유지할 KV cache도 커진다. MobileVLM과 Imp의 visual tokenization 방식 차이를 통제하지 않으면 memory percentage 비교만으로 runtime 효율을 평가할 수 없다.

논문은 LLaVA llama.cpp 55%, mllm 40%, MobileVLM 32%, Imp 3% 미만을 보고하지만 12GB/16GB 기준이 혼재하고 측정 도구가 process RSS인지 system memory인지 명확하지 않다. `mllm 40%가 larger KV cache를 반영한다`는 본문 설명도 llama.cpp의 55%보다 낮아 직관적으로 맞지 않는다.

재현에서는 다음을 분리해야 한다.

- model mmap/file cache
- native heap
- GPU shared allocation
- KV cache
- vision intermediate
- OS/system memory

## Output Quality와 길이

같은 burger image에 대해 다음 answer length를 보고한다.

| Stack | Answer length |
|---|---:|
| LLaVA + llama.cpp | 30 tokens |
| LLaVA + mllm | 30 tokens |
| MobileVLM | 18 tokens |
| Imp | 11 tokens |

LLaVA는 세 문장, MobileVLM은 두 문장, Imp는 한 clause 수준이다. 모두 주요 object를 올바르게 분류했다고 한다.

짧은 output은 latency와 energy를 줄이지만 품질 우수성을 뜻하지 않는다. Fixed `max_new_tokens`와 stop condition, exact tokenizer, answer verbosity prompt를 고정해야 한다. 한 image의 qualitative caption만으로 VQA accuracy나 hallucination을 비교할 수 없다.

Deployment benchmark에는 최소한 다음이 필요하다.

- fixed output tokens/s test
- task-completion latency test
- VQA/caption benchmark accuracy
- hallucination 또는 object presence score
- response length-normalized energy

## 논문이 제안하는 최적화

### GPU FlashAttention

Attention과 MLP를 Adreno FP16으로 옮기면 prompt latency를 절반, power를 약 4배 줄일 수 있다고 제안한다. 이는 future opportunity이며 본 논문에서 실제 구현/검증한 결과는 아니다.

### Hexagon INT8 decoder

Final projection을 NPU로 옮겨 1-2W를 절약할 수 있다고 추정한다. 하지만 single-token decode와 전체 decoder 중 final projection 비중, transfer cost를 측정하지 않았다.

### Mixed-precision KV cache

FP8 또는 INT4 KV로 memory를 약 40% 줄일 수 있다고 제안한다. 문장이 PDF에서 `around 40`으로 끝나 percent 표기가 누락되어 있다. Quality loss와 dequantization kernel은 실험하지 않았다.

### Stage overlap

Image encoding을 GPU, prompt evaluation을 CPU에서 동시에 수행해 Mobile model에서 2-3초를 숨길 수 있다고 제안한다. 그러나 prompt evaluation은 visual embedding이 준비되어야 전체 multimodal sequence를 처리할 수 있어, text-only prefix 일부 또는 다음 request pipeline이 아니면 dependency 때문에 완전 overlap은 어렵다. 정확한 data dependency graph가 필요하다.

## 숫자와 서술의 내부 불일치

이 논문을 사용할 때 반드시 기록해야 할 항목이다.

| 항목 | 서술 A | 서술 B | 영향 |
|---|---|---|---|
| SoC | Snapdragon 8 Gen 2 | limitation/conclusion은 Gen 3 | hardware 재현 불가 |
| GPU | Adreno 740 | Gen 3 주장과 조합 불명확 | backend 성능 비교 불가 |
| RAM | methodology 12GB | results 16GB pool | memory GB 환산 오류 |
| framework 수 | four frameworks | 실제 이름은 3개 | 비교 범위 혼동 |
| LLaVA llama total | stage 합 150.8s | total 82.2s | phase breakdown 불명확 |
| MobileVLM total | stage 합 37.3s | table 22.8s, 본문 약 35s | latency ranking 불안정 |
| LLaVA output tokens | latency 본문 69/51 | answer table 30/30 | tokens/s 비교 불가 |
| MobileVLM power | rail current 약 57mA | summary 약 3.5W | sensor/단위 불명확 |

Review 관점에서 이 표는 논문의 가치를 부정하기 위한 것이 아니다. 오히려 mobile profiling에서 raw trace, measurement schema, device fingerprint가 왜 중요한지 보여준다.

## 재현 가능한 Benchmark 설계

### Phase를 exclusive하게 정의

```text
T0 app start
T1 model ready
T2 image tensor ready
T3 vision embedding ready
T4 first LLM logit ready
T5 first output token emitted
T6 final token emitted
```

```math
load=T1-T0
```

```math
preprocess=T2-T1
```

```math
vision=T3-T2
```

```math
multimodal\ prefill=T4-T3
```

```math
TTFT=T5-T0\quad\text{or warm }T5-T1
```

```math
decode=T6-T5
```

Nested runtime timer는 별도 column으로 저장하고 exclusive wall-clock과 섞지 않는다.

### Device fingerprint

```text
phone SKU:
SoC / CPU cores:
GPU / driver:
NPU / SDK:
RAM total:
storage type:
Android build:
kernel:
battery health:
ambient temperature:
```

### Model fingerprint

```text
model commit:
vision encoder:
LLM:
projector:
weight quantization and group size:
KV cache precision:
context length:
image resolution / slices:
visual tokens:
prompt tokens:
generated tokens:
```

### Runtime fingerprint

```text
framework commit:
compiler flags:
CPU thread count and affinity:
GPU layers/operators:
NPU operators:
memory map / page lock:
warm-up runs:
```

## 온디바이스 통합 Pipeline에 주는 교훈

모든 모델을 항상 실행하면 thermal budget을 넘는다. 사용자의 최종 프로젝트에는 다음 구조가 적합하다.

```text
camera frame
  -> lightweight detector at low duty cycle
  -> tracking/cache
  -> event or user query?
       no: skip heavy stages
       yes:
         selected ROI -> segmenter on demand
         ROI crop / token reduction
         vision encoder on GPU/NPU
         LLM prefill on GPU/NPU
         decode on best measured CPU/GPU path
```

### Dynamic execution interval

Detector가 scene change가 없다고 판단하면 frame rate를 낮춘다. Temperature와 battery state에 따라 resolution, model, query interval을 조절한다.

### ROI token reduction

Full-resolution image 대신 detector box/mask의 ROI만 VLM에 넣으면 vision FLOPs, visual token, prefill, KV cache가 함께 감소한다. FastVLM의 encoder latency 관점과 이 논문의 system trace를 연결하는 지점이다.

### Cache

- 동일 image의 vision feature cache
- 동일 prompt prefix의 KV cache
- detector text embedding cache
- repeated UI query의 prefix cache

Cache hit rate와 invalidation 비용을 memory budget과 함께 측정해야 한다.

### Task별 precision

Vision encoder INT8, projector FP16, LLM W4A16, KV INT8/FP8 등으로 분리한다. 단일 `model quantized` label만으로는 어느 operator가 accelerator에 배치됐는지 알 수 없다.

## 강점

1. **실제 consumer phone에서 end-to-end VLM을 측정**: desktop GPU proxy가 아니다.
2. **Latency 이외의 지표 포함**: utilization, power, thermal, UI responsiveness를 함께 본다.
3. **Phase별 trace를 시도**: image, prompt, decode 병목이 다름을 보여준다.
4. **같은 LLaVA의 runtime 차이 제시**: architecture 외 software stack의 중요성을 드러낸다.
5. **NPU 0%와 GPU idle을 명시**: 이론 peak TOPS가 실제 성능을 보장하지 않음을 보여준다.
6. **GPU offload의 power/thermal 이점과 UI contention을 함께 관찰**한다.

## 한계

### 단일 장치와 단일 image/prompt

SoC 하나, single-image English query 중심이다. Model별 accuracy benchmark와 다양한 resolution/context가 없다.

### 실험 조합이 교차되지 않음

모든 model을 모든 runtime에서 실행하지 않는다. Imp+MLC와 MobileVLM+llama 차이는 model과 runtime 효과를 분리할 수 없다.

### Quantization과 configuration 누락

Weight format, bit-width, group size, context length, visual token 수, thread affinity 같은 핵심 조건이 충분하지 않다. Parameter 수만으로 memory/latency를 해석하기 어렵다.

### 내부 수치 불일치

Device spec, total time, answer token, memory denominator가 맞지 않는다. Table의 절대값과 battery-life extrapolation은 재검증이 필요하다.

### Tail latency와 장시간 안정성 미보고

5회 평균만 제시하고 p50/p95, variance, frequency, throttling onset을 충분히 제공하지 않는다.

### 제안과 검증의 혼재

GPU FlashAttention, NPU INT8, mixed KV, stage overlap의 예상 이득은 구현 결과가 아니라 future work다. 논문이 측정한 결과와 전망을 분리해야 한다.

## 권장 추가 실험

### 실험 1: 2x2 factorial comparison

동일 3B model을 CPU-only와 GPU-offload로, 동일 runtime에서 3B와 7B를 비교해 model-size 효과와 runtime 효과를 분리한다.

### 실험 2: Visual token sweep

```text
resolution: 224, 336, 448, 672
visual tokens: 64, 144, 256, 576
```

Vision latency, prefill latency, KV memory, VQA accuracy를 함께 그린다.

### 실험 3: Fixed token protocol

모든 stack에서 같은 prompt와 `max_new_tokens=64`, 같은 stop policy를 사용하고 warm-up 후 TTFT와 decode tokens/s를 측정한다.

### 실험 4: Accelerator operator map

각 graph node가 CPU/GPU/NPU 중 어디에서 실행됐는지 trace한다.

```text
vision conv/attention:
projector GEMM:
LLM QKV:
attention score:
RoPE:
MLP:
KV update:
LM head:
sampling:
```

Fallback operator 한 개가 반복적인 device copy를 만드는지 확인한다.

### 실험 5: Thermal soak

30분 동안 1분 간격 query가 아니라 workload가 끝난 뒤 고정 idle을 포함한 명확한 duty cycle로 반복한다. Frequency, surface/die temperature, power, tokens/s 변화를 기록한다.

### 실험 6: UI-aware GPU scheduling

GPU inference queue priority와 kernel slice 길이를 바꾸며 VLM latency와 UI frame drop을 동시에 측정한다.

## 실험 기록 템플릿

```text
Device fingerprint:
Runtime commit/build flags:
Model/quantization/KV precision:
Image resolution and visual tokens:
Prompt/output tokens:

Cold load:
Warm image encode:
Warm prefill:
TTFT p50/p95:
Decode tokens/s p50/p95:
End-to-end p50/p95:

CPU utilization/frequency:
GPU utilization/frequency:
NPU utilization:
Operator fallback count:
Peak RSS/GPU shared memory/KV:

Average/peak power:
Joules/query:
Joules/generated token:
Die/surface temperature:
Throttling onset:
UI dropped frames/input delay:

Accuracy benchmark:
Response length:
Failure and crash count:
```

## 이 로드맵에서의 의미

이 논문은 앞선 architecture와 compression 논문을 실제 device에서 검증하는 마지막 단계에 해당한다.

```text
efficient backbone
  -> vision FLOPs/activation 감소

visual token reduction
  -> prefill와 KV cache 감소

quantization
  -> weight bandwidth와 model size 감소

runtime mapping
  -> CPU/GPU/NPU operator 배치

system scheduling
  -> power, thermal, UI responsiveness 관리
```

앞 단계의 논문이 `몇 GFLOPs`, `몇 GB`, `몇 bits`를 제시하더라도 실제 runtime이 GPU/NPU를 사용하지 못하면 CPU가 600-800% 포화되고 수십 초가 걸릴 수 있다. 반대로 작은 model이라도 image encoder를 비효율적으로 GPU에 올리면 18초와 100% GPU saturation이 생길 수 있다.

## 최종 평가

OnePlus 13R case study의 가장 큰 가치는 최신 phone의 peak TOPS가 아니라 **현재 open-source VLM stack이 accelerator를 거의 사용하지 못하며, framework 결정이 같은 model의 latency를 두 배 이상 바꿀 수 있다**는 현실을 수치와 trace로 보여준 데 있다. CPU-only LLaVA의 높은 temperature와 power, MLC-Imp의 GPU offload가 만든 낮은 CPU/power 대비는 runtime 공동 설계의 필요성을 분명히 한다.

동시에 이 논문은 mobile benchmark가 지켜야 할 재현성 기준도 역으로 보여준다. SoC/RAM 표기, phase timer, total wall-clock, token count, sensor channel이 일관되지 않으면 작은 표 하나로 framework를 순위화할 수 없다. 특히 GPU FlashAttention, NPU INT8, KV compression의 예상 speedup은 아직 측정 결과가 아니다.

따라서 이 논문은 최종 성능 숫자의 권위 있는 기준보다 **profiling checklist와 병목 가설을 제공하는 탐색적 case study**로 읽는 것이 적절하다. 후속 프로젝트에서는 exact device fingerprint, exclusive phase timing, fixed token protocol, operator placement, p95 latency, peak memory, joules/token, 장시간 thermal throttling을 함께 수집해야 한다.

## 체크리스트

- [ ] Device의 SoC, GPU, RAM을 실제 system log로 확인했는가?
- [ ] Framework 수와 model-runtime stack 수를 구분했는가?
- [ ] Stage timer가 exclusive인지 nested인지 확인했는가?
- [ ] Cold TTFT, warm TTFT, decode tokens/s를 분리했는가?
- [ ] 모든 model에 동일 output token/stop condition을 적용했는가?
- [ ] CPU 600%를 6 core 수준으로 올바르게 해석했는가?
- [ ] GPU/NPU operator placement와 CPU fallback을 trace했는가?
- [ ] Process RSS, GPU shared memory, KV cache를 분리했는가?
- [ ] Power sensor rail, voltage, sampling rate를 기록했는가?
- [ ] 평균뿐 아니라 p50/p95와 30분 thermal soak를 측정했는가?
- [ ] UI frame drop과 touch latency를 accelerator utilization과 함께 봤는가?
- [ ] 논문 실측과 future optimization 예상치를 구분했는가?

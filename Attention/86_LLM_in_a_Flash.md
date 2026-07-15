# 86. LLM in a Flash

## 논문 정보

- 원본 파일: `86_LLM_in_a_Flash.pdf`
- 제목: **LLM in a flash: Efficient Large Language Model Inference with Limited Memory**
- 저자: Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, S. Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar
- 소속: Apple
- 공개: arXiv 2023, v3 2024-07-30
- 공식 링크: [arXiv:2312.11514](https://arxiv.org/abs/2312.11514)
- 주제: flash-resident LLM inference, activation sparsity, selective weight loading, memory hierarchy
- 핵심 키워드: low-rank predictor, sliding window, row-column bundling, sparse FFN, DRAM budget, flash bandwidth

## 한눈에 보는 요약

이 논문은 LLM 전체가 DRAM에 들어가지 않을 때 모델 weight를 flash에 두고, 각 token 생성에 필요한 FFN neuron weight만 DRAM으로 읽어 오는 방법을 제안한다. 목표는 compute를 줄이는 것이 아니라 **flash에서 읽는 byte 수, 읽기 chunk 크기, DRAM 안에서 weight를 교체하는 비용**을 함께 최적화하는 것이다.

핵심 흐름은 다음과 같다.

```text
attention output h_t
    |
    v
small low-rank predictor
    |
    +-> 이번 token에서 active할 FFN neuron 예측
    |
    +-> 최근 k token의 active neuron cache와 비교
            |
            +-> 이미 DRAM에 있음: 재사용
            +-> 새 neuron: flash에서 bundled weight chunk 읽기
            +-> window 밖 neuron: preallocated buffer에서 제거
    |
    v
sparse FFN compute -> next layer
```

두 대표 기법은 다음과 같다.

- **Windowing**: 최근 token들이 사용한 neuron weight를 DRAM에 유지해, 새 token에서 새로 읽어야 할 weight의 차집합만 전송한다.
- **Row-column bundling**: 하나의 intermediate neuron에 함께 필요한 up-projection과 down-projection weight를 flash에 연속 배치해 작은 random read 두 번을 더 큰 read 한 번으로 만든다.

약 모델 크기의 절반만 DRAM에 둘 수 있는 실험에서 OPT-6.7B의 token당 latency는 CPU `3182 -> 669 ms`, M1 Metal `2389 -> 565 ms`, M2 Metal `2270 -> 305 ms`, RTX 4090 GPU `2218 -> 84 ms`로 줄었다. 하지만 비교 기준은 모델 전체가 DRAM에 상주하는 정상적인 dense inference가 아니라, **매 token마다 부족한 weight를 flash에서 다시 읽는 naive 또는 hybrid offload**다.

스마트폰에서 7B 4-bit 모델을 2GB 미만 DRAM으로 실행할 수 있다는 내용은 논문의 추정이다. 실제 스마트폰 구현, 4-bit sparse kernel, TTFT, 긴 prompt prefill, thermal/power 정량 평가는 제공하지 않는다.

## 문제 정의: 모델이 DRAM보다 클 때

Parameter 수가 `P`, weight precision이 `b` bit이면 weight만의 최소 저장량은 다음과 같다.

```math
M_W=P\frac{b}{8}\text{ bytes}
```

7B model은 FP16에서 대략 다음 크기다.

```math
7\times10^9\times2\text{ bytes}\approx14\text{ GB}
```

여기에 KV cache, runtime buffer, activation, OS와 다른 app의 메모리가 더해진다. 8GB 또는 12GB 통합 메모리 장치에서는 weight 전체를 DRAM에 올리는 것부터 불가능할 수 있다.

Flash는 용량이 크고 비휘발성이지만 DRAM보다 bandwidth와 latency가 나쁘다. 논문이 제시한 개념적 memory hierarchy는 다음과 같다.

```text
flash: large capacity, roughly GB/s, high first-byte latency
DRAM:  smaller capacity, roughly 100 GB/s class
CPU/GPU local cache/register: much faster, very small
```

Naive offload는 layer를 계산할 때 필요한 weight를 flash에서 읽고 다음 layer를 위해 버린다. Autoregressive decoding은 새 token마다 모든 transformer layer를 다시 통과하므로, weight를 cache하지 못하면 같은 GB 단위 데이터를 매 token마다 재전송한다.

예를 들어 13.4GB를 6.1GB/s로 읽는 이론적 하한은 다음과 같다.

```math
T_{IO}\approx\frac{13.4}{6.1}\text{ s}\approx2.20\text{ s/token}
```

실제 random read, syscall, controller, memory copy가 더해지면 더 느릴 수 있다. 따라서 이 문제에서는 FLOPs보다 `bytes/token`과 `effective flash throughput`이 우선 지표다.

## Flash의 특성: 적게 읽는 것만으로 충분하지 않다

Flash는 큰 sequential read에서는 빠르지만 작은 random read에서는 throughput이 크게 낮다. 각 read에는 다음 고정 비용이 있기 때문이다.

- system call 및 file-system 경로
- OS block layer와 driver
- interrupt 처리
- flash controller command
- latency to first byte

전체 전송 시간을 단순화하면 다음과 같다.

```math
T_{read}\approx N_{req}L_{first-byte}+\frac{V_{read}}{B_{seq}(c,p)}
```

- `N_req`: read request 수
- `L_first-byte`: 각 request의 고정 지연
- `V_read`: 읽은 총 byte
- `B_seq(c,p)`: chunk 크기 `c`, parallel stream 수 `p`에서의 유효 throughput

Sparse loading은 `V_read`를 줄이는 대신 흩어진 작은 read를 늘려 `N_req`를 키우고 `B_seq`를 낮춘다. 이 때문에 필요한 byte만 정확히 읽는 방식이 더 많은 byte를 큰 chunk로 읽는 방식보다 느릴 수도 있다.

논문은 M1 Max의 uncached 1GiB linear read가 6GiB/s를 넘지만, 작은 random read에서는 크게 낮아진다고 보고한다. 32KiB 이상 chunk와 여러 thread를 사용하면 sparse inference에 필요한 throughput에 접근할 수 있다는 것이 설계의 출발점이다.

## 비용 모델

논문은 token당 inference latency를 세 부분으로 분해한다.

```math
T_{token}=T_{IO}+T_{mem}+T_{compute}
```

### Flash I/O

```math
T_{IO}=f(V_{read},\text{chunk size},\text{request parallelism},\text{cache policy})
```

읽는 weight 양과 access pattern이 결정한다.

### DRAM memory management

```math
T_{mem}=T_{delete}+T_{insert}+T_{repack}+T_{host-device-copy}
```

새 neuron을 넣을 때 기존 matrix 전체를 재할당하거나 복사하면 flash I/O를 줄여도 이 비용이 커진다.

### Compute

```math
T_{compute}=T_{attention}+T_{predictor}+T_{sparse\ FFN}+T_{sampling}
```

논문의 주된 최적화 대상은 compute가 아니다. 다만 active FFN neuron만 계산하므로 compute도 일부 감소한다. Table 3에서 OPT CPU compute가 `986 -> 506 ms`로 내려가는 이유다.

이 분해는 실제 배포에서 유용하다. 총 latency만 보면 flash, memory copy, sparse kernel 중 어느 것이 병목인지 알 수 없다.

## 전제 1: FFN activation sparsity

Transformer FFN을 단순화하면 다음과 같다.

```math
u=W_{up}h+b_{up}
```

```math
a=\mathrm{ReLU}(u)
```

```math
y=W_{down}a+b_{down}
```

ReLU 뒤의 `a_j=0`이면 intermediate neuron `j`에 대응하는 down-projection weight는 결과에 기여하지 않는다. 논문이 사용하는 OPT-6.7B는 FFN activation의 약 97%가 0, 즉 평균 active 비율이 약 3% 수준이다. Falcon-7B는 원 모델을 ReLU로 바꾸고 fine-tune한 sparsified version을 사용한다. Llama 2는 FATReLU 기반 sparsified model을 쓴다.

중요한 점은 일반적인 GELU, SiLU, SwiGLU output이 정확히 0이 아니라는 것이다. 이 논문의 기법은 아무 dense LLM checkpoint에 즉시 적용되는 범용 paging이 아니다. 다음 중 하나가 필요하다.

- 원래 ReLU 계열로 sparse한 모델
- activation function을 바꾸고 fine-tuning한 relufied model
- threshold를 넣은 FATReLU sparse model
- sparse MoE처럼 원래 선택적 weight access가 가능한 모델

Sparsification 과정의 정확도 손실과 predictor 오차를 offload speedup과 함께 평가해야 한다.

## 왜 Predictor가 필요한가

어떤 neuron이 active인지 정확히 알려면 원래 `W_up h`를 모두 계산해야 한다. 그러나 `W_up` 전체가 flash에 있으므로 이를 계산하려면 먼저 전체 up-projection을 읽어야 하고, 선택적 loading의 목적이 사라진다.

논문은 attention output을 입력으로 받는 작은 low-rank predictor를 layer마다 학습한다.

```math
\hat z=\sigma\left(B(Ah)\right)
```

```math
A\in\mathbb{R}^{r\times d_{model}},\qquad
B\in\mathbb{R}^{d_{ff}\times r}
```

```math
\hat m_j=\mathbf{1}[\hat z_j\gt 0.5]
```

`r`이 작으므로 원래 up-projection보다 predictor parameter와 compute가 작다. Positive와 negative sample 수가 크게 다르기 때문에 balanced loss를 쓴다.

논문은 C4 training sample 10,000개, 2 epochs로 predictor를 학습하며 A100에서 각 predictor 학습에 약 4시간이 걸렸다고 서술한다. OPT-6.7B에서는 앞 28 layers에 rank 128, 마지막 4 layers에 rank 1024를 사용한다. 평균 false negative 5%, false positive 7%를 보고한다.

- False positive: 불필요한 weight를 읽어 latency와 memory가 증가하지만 출력은 보존된다.
- False negative: 실제 active neuron을 빠뜨려 모델 출력과 정확도에 직접 영향을 준다.

논문 Figure 3은 false negative가 주로 0에 가까운 작은 preactivation에 몰린다고 설명한다. OPT zero-shot 결과는 predictor 적용 전후 거의 같다.

| Task | OPT-6.7B | Predictor 적용 |
|---|---:|---:|
| ARC Easy | 66.1 | 66.2 |
| ARC Challenge | 30.6 | 30.6 |
| HellaSwag | 50.3 | 49.8 |

하지만 이 세 지표만으로 long-form generation, factuality, rare token, safety behavior가 보존됐다고 단정할 수는 없다.

## Selective Persistence

모든 weight를 flash에 둘 필요는 없다. 논문은 다음을 DRAM에 지속적으로 둔다.

- token embedding
- attention projection weight
- low-rank predictor
- 최근 window에서 사용한 FFN neuron weight

Attention weight는 모델 크기의 대략 1/3이므로 이를 resident로 두면 매 token의 dense attention weight I/O를 없앨 수 있다. Dynamic하게 읽는 부분은 FFN이다.

이 설계는 memory budget이 attention resident set보다 작으면 그대로 적용할 수 없다. KV cache가 커지는 긴 context에서도 FFN cache에 쓸 수 있는 DRAM이 줄어든다. 논문 실험은 batch 1이며 KV cache에 일정 부분을 남긴 뒤 model weight에 약 절반 크기의 memory를 배정한다.

## Sliding Window

### 핵심 아이디어

연속 token은 문맥이 비슷하므로 active neuron set도 겹친다는 관찰을 이용한다. Token `t`에서 predictor가 고른 set을 `S_t`라 하자. 최근 `k`개 token의 union을 cache하면 resident FFN set은 다음과 같다.

```math
C_t=\bigcup_{i=t-k+1}^{t}S_i
```

다음 token에서 새로 읽는 neuron은 다음 차집합뿐이다.

```math
L_{t+1}=S_{t+1}\setminus C_t
```

Window를 한 칸 이동하면서 더 이상 필요하지 않은 neuron은 제거한다.

```math
D_{t+1}=C_t\setminus\bigcup_{i=t-k+2}^{t+1}S_i
```

논문 notation으로 `s_agg(k)`는 `k` token 동안의 aggregated neuron usage다. Window를 늘리면 DRAM의 union set은 커지지만 token당 incremental transfer `s_agg(k+1)-s_agg(k)`는 감소한다.

### OPT-6.7B의 구체적 결과

논문은 OPT에 window size `k=4`를 사용하고 최근 5 token을 위한 공간이라고도 표현한다. 현재 token 포함 여부에 따른 count 차이이므로 구현에서는 정의를 명확히 해야 한다.

- Predictor active neuron을 매번 새로 읽으면 FFN weight의 약 10% DRAM access가 필요
- Window reuse 후 새 token당 새로 읽는 비율은 FFN의 약 2.4%
- Window cache 자체는 FFN의 약 24%를 DRAM에 유지

전체 모델 크기 대비 DRAM 구성은 다음과 같다.

| 구성 | 모델 크기 대비 비율 |
|---|---:|
| embedding | 3.0% |
| attention | 32.3% |
| predictor | 1.25% |
| cached FFN | 15.5% = 24% x 64.62% |
| 합계 | 52.1% |

즉 약 48% 모델 weight를 상주시킬 수 없는 조건에서도 실행 가능하게 만든다.

### Memory-latency trade-off

Window가 클수록 cache hit가 늘어 flash traffic은 줄지만 DRAM 사용량은 커진다.

```text
small k -> less DRAM, more flash reads, higher latency
large k -> more DRAM, fewer flash reads, lower latency
```

이것은 고정 hyperparameter라기보다 runtime resource controller 문제다. 다른 app이 memory pressure를 만들거나 KV cache가 커지면 `k`를 줄이고, 여유가 생기면 늘릴 수 있다.

## Row-Column Bundling

Intermediate neuron `j`가 active하면 up-projection에서 그 neuron을 만드는 weight vector와 down-projection에서 그 neuron을 소비하는 weight vector가 함께 필요하다. Matrix 저장 convention에 따라 하나는 row, 다른 하나는 column으로 표현된다.

논문은 두 vector를 flash에 연속 배치한다.

```text
neuron j chunk:
[up-projection vector for j | down-projection vector for j]
```

각 element가 `num_bytes`, vector 길이가 `d_model`이면 chunk는 다음 크기다.

```math
C_{bundle}=2d_{model}\times\text{num\_bytes}
```

OPT의 32-bit CPU 예에서 `d_model=4096`이면 다음과 같다.

```math
2\times4096\times4=32,768\text{ bytes}=32\text{ KiB}
```

두 개의 16KiB random read 대신 한 개의 32KiB read가 되어 flash throughput이 좋아진다. Table 2에서 같은 0.2GB를 읽으면서 throughput이 `1.25 -> 2.25 GB/s`, I/O latency가 `164 -> 87 ms`로 감소한다.

Bundling은 FLOPs나 읽는 유효 weight 수를 줄이지 않는다. Storage layout을 hardware read granularity에 맞추는 system optimization이다.

## Co-activation Bundling의 실패

함께 활성화되는 neuron들을 한 bundle로 묶으면 request 수를 더 줄일 수 있을 것처럼 보인다. 논문은 C4에서 각 neuron의 가장 자주 함께 활성화되는 `closest friend`를 찾았다. Co-activation은 power-law 형태이며 가까운 neuron pair는 높은 확률로 함께 켜졌다.

그러나 이 기법은 실패했다. 매우 자주 활성화되는 몇몇 neuron이 많은 neuron의 closest friend가 되어 여러 bundle에 중복 저장되거나 중복 loading되었기 때문이다.

이 negative result의 교훈은 `pairwise co-activation`만 최대화하면 안 된다는 것이다. 실제 bundling objective는 다음을 함께 최소화해야 한다.

```math
\text{cost}=
\alpha\cdot\text{bytes read}
+\beta\cdot\text{requests}
+\gamma\cdot\text{duplicate loads}
+\delta\cdot\text{cache pollution}
```

Deployment corpus가 바뀌면 co-activation graph도 바뀔 수 있으므로 static clustering의 일반화도 검증해야 한다.

## DRAM Data Structure와 Memory Management

새 neuron이 들어올 때마다 sparse matrix를 재할당하고 기존 rows를 복사하면 `T_mem`이 커진다. 논문은 layer마다 최대 cache 크기를 미리 할당한다.

```text
matrix       : [Req_i, 2*d_model]
pointer      : original neuron id -> current compact row
scalar/bias  : per-neuron metadata
num_used     : currently occupied rows
last_k_active: recent token activity metadata
```

삭제할 row는 buffer 끝의 row와 swap해 occupied prefix를 연속으로 유지한다. 새 bundled row는 `matrix[num_used:]` 뒤에 붙인다.

```python
def update_cache(cache, predicted_active, flash):
    keep = neurons_used_in_recent_window(cache)
    to_delete = cache.active - keep - predicted_active
    to_load = predicted_active - cache.active

    for neuron in to_delete:
        cache.swap_remove(neuron)

    chunks = flash.parallel_read_bundles(to_load)
    cache.append(chunks)
    cache.update_recency(predicted_active)
```

FFN 계산에서는 compact buffer의 앞 절반을 up-projection, 뒤 절반을 transpose해 down-projection으로 사용한다. Intermediate neuron 순서를 함께 permutation하면 합산 결과가 같기 때문에 original neuron 순서로 매번 정렬할 필요가 없다.

삭제 `c`개에는 대략 `O(c d_model)` memory rewrite가 필요하다. Preallocation은 allocation과 전체 matrix copy를 없애지만 swap 및 metadata update 비용까지 없애지는 않는다.

## OS Cache를 끈 이유

파일을 읽으면 OS page cache가 flash block을 DRAM에 저장할 수 있다. 그러면 benchmark는 빠르게 보이지만 model cache 외의 숨은 DRAM을 소비한다. Memory가 모델보다 작은 조건에서는 page cache 때문에 다른 process와 경쟁하거나, layer를 순서대로 돌면서 앞 layer block이 뒤 layer 때문에 축출되어 hit rate가 거의 0이 될 수도 있다.

논문은 보수적이고 반복 가능한 flash 측정을 위해 다음을 사용한다.

- macOS/iOS: `fcntl()`의 `F_NOCACHE`
- Linux: Direct I/O
- macOS 측정 전 resident buffer purge
- 32-thread parallel reads

실서비스는 OS cache를 완전히 끌 필요가 없지만, report할 때는 page cache 포함 여부와 실제 process RSS, system-wide memory pressure를 함께 측정해야 한다.

## 실험 설정

### Models

- OPT-6.7B
- sparsified Falcon-7B
- Persimmon-8B
- relufied Phi-2 2.7B
- FATReLU sparsified Llama 2-7B

### Data와 generation

C4 validation subset의 각 sample에서 첫 128 token을 prompt로 사용하고 256 new tokens를 생성한다. Batch는 1이다.

### Hardware

- Apple M1 Max, 1TB SSD: CPU float32 또는 Metal GPU float16
- Apple M2 Ultra, 2TB SSD: Metal GPU
- Linux, NVIDIA RTX 4090 24GB: bfloat16 GPU

모든 환경에서 model computation에 전체 model 크기의 약 절반 정도 memory만 쓸 수 있다고 가정한다. Phi-2는 sparsity가 낮아 약 65% memory를 허용한다.

### Baselines

- Naive: 필요한 model weight를 forward마다 flash에서 읽음
- Hybrid: model 절반은 resident, 나머지 절반은 매 token loading, sparsity 사용 안 함
- All: predictor + windowing + bundling + preallocated management

Hybrid의 절반-model read는 sparsity나 weight sharing을 쓰지 않을 때 가능한 이론적 I/O 하한으로 설정한다. 실제 구현은 더 느릴 수 있어 baseline에 유리한 비교다.

## OPT I/O Ablation

OPT-6.7B, model의 절반 정도 memory가 가능한 조건의 Table 2는 각 기법의 효과를 분리한다.

| Hybrid | Predictor | Window | Bundle | DRAM | Flash->DRAM/token | Throughput | I/O latency |
|---|---|---|---|---:|---:|---:|---:|
| X | X | X | X | 0GB | 13.4GB | 6.10GB/s | 2196ms |
| O | X | X | X | 6.7GB | 6.7GB | 6.10GB/s | 1090ms |
| O | O | X | X | 4.8GB | 0.9GB | 1.25GB/s | 738ms |
| O | O | O | X | 6.5GB | 0.2GB | 1.25GB/s | 164ms |
| O | O | O | O | 6.5GB | 0.2GB | 2.25GB/s | 87ms |

핵심은 sparse read의 throughput이 dense sequential read보다 낮다는 점이다. Predictor가 byte를 `6.7 -> 0.9GB`로 줄여도 throughput이 `6.10 -> 1.25GB/s`로 떨어진다. Windowing이 transfer를 `0.2GB`까지 낮추고, bundling이 throughput을 일부 회복해야 최종 `87ms`가 된다.

## End-to-End Latency

논문 Table 3의 token당 평균 latency는 다음과 같다.

| Model | Method | Backend | I/O | Memory | Compute | Total |
|---|---|---|---:|---:|---:|---:|
| OPT-6.7B | Naive | CPU | 2196 | 0 | 986 | 3182ms |
| OPT-6.7B | All | CPU | 105 | 58 | 506 | 669ms |
| OPT-6.7B | Naive | Metal M1 | 2196 | 0 | 193 | 2389ms |
| OPT-6.7B | All | Metal M1 | 92 | 35 | 438 | 565ms |
| OPT-6.7B | Naive | Metal M2 | 2145 | 0 | 125 | 2270ms |
| OPT-6.7B | All | Metal M2 | 26 | 8 | 271 | 305ms |
| OPT-6.7B | Naive | RTX GPU | 2196 | 0 | 22 | 2218ms |
| OPT-6.7B | All | RTX GPU | 30 | 34 | 20 | 84ms |
| OPT-6.7B | Speculative | RTX GPU | 38.5 | 9.5 | 12 | 60ms |
| Falcon-7B | Naive | CPU | 2295 | 0 | 800 | 3095ms |
| Falcon-7B | Hybrid | CPU | 1147 | 0 | 800 | 1947ms |
| Falcon-7B | All | CPU | 161 | 92 | 453 | 706ms |
| Persimmon-8B | All | CPU | 283 | 98 | 660 | 1041ms |
| Phi-2 2.7B | All | CPU | 211 | 76 | 259 | 546ms |
| Llama 2-7B | All | CPU | 279 | 152 | 563 | 994ms |

Appendix Table 5는 반복 측정의 standard deviation도 제공한다. OPT All은 CPU `669.20 +/- 39.74ms`, GPU `84.64 +/- 6.16ms`, speculative GPU `60.16 +/- 13.4ms`다.

RTX GPU에서 naive 대비 약 26배지만, dense model이 24GB VRAM/DRAM에 완전히 상주할 수 있는 환경의 latency와 비교한 것이 아니다. `84ms/token`은 약 `11.9 tokens/s`, `60ms/token`은 약 `16.7 tokens/s`다. 반면 CPU `669ms/token`은 약 `1.49 tokens/s`다.

## Speculative Decoding과의 결합

Draft model이 `lambda=4` tokens를 제안하고 큰 model이 검증한다. Window cache에는 검증 후 살아남을 token의 active neuron을 유지해야 한다. Acceptance ratio를 `alpha`라 하면 논문은 대략 `alpha(lambda+1)`번째 지점 근처를 기준으로 최근 `k` token을 보존하는 전략을 사용한다.

OPT GPU에서 `84 -> 60ms/token`, 약 `1.4x` 추가 speedup을 얻는다. 원래 speculative decoding의 `1.58x`에 가깝다. Flash loading과 speculative decoding은 직교적인 최적화가 될 수 있음을 보여준다.

다만 draft model memory도 필요하다. 제한 DRAM 조건에서 draft model, main predictor, FFN cache, KV cache의 합을 명시해야 한다.

## 긴 Generation과 Thermal

논문은 OPT GPU에서 1000 tokens까지 생성하며 flash latency가 뒤로 갈수록 증가하지 않는다고 보고한다. 초기 몇 token은 empty cache를 채워야 해 오히려 flash latency가 더 높다. Greedy와 nucleus sampling도 장기 경향이 비슷했다.

그러나 SSD thermal throttling이 없었다는 이 실험을 smartphone UFS와 동일하게 볼 수 없다. 장치의 enclosure, ambient temperature, flash controller, 동시 write workload, battery state가 다르다.

전력에 대해서도 중요한 부정적 결과가 있다.

- Sparse model은 순간 power가 dense model보다 낮았다.
- 그러나 token 생성 시간이 더 길어 total energy는 더 높았다.
- 정량적인 power/energy 분석은 future work로 남겼다.

즉 memory capacity 문제를 해결했다고 energy efficiency까지 해결된 것은 아니다.

## 정확도와 Sparsification 비용

Predictor 자체의 영향은 작아 보여도, dense model을 sparse activation model로 바꾸는 과정에서 정확도가 떨어질 수 있다.

### Llama 2

- Original Llama 2 MMLU: 41.8
- FATReLU sparsified model: 38.96
- Predictor 추가 후: 38.63

Predictor 추가 손실은 작지만 원 모델 대비 sparsification 손실은 약 2.8 points다.

### Phi-2

Relufication과 distillation 후 MMLU가 `57 -> 54.3`으로 낮아진다. Layer별 sparsity가 달라 predictor rank도 그룹별로 160, 480, 800 등 다르게 설정한다. 마지막 group에는 predictor를 쓰지 않는다.

### Quantization과 sparsity

Appendix는 OPT quantization 전후 active 비율이 거의 같다고 보고한다. 평균 `3.30% -> 3.27%`다. 이를 근거로 4-bit weight에도 같은 selective loading을 적용할 수 있다고 제안한다. 하지만 실제 4-bit random gather, unpack, sparse GEMM kernel은 구현하지 않았다.

## 스마트폰에 대한 주장과 실제 범위

논문은 7B 4-bit model이 baseline에서 약 3.5GB weight를 필요로 하고, 제안 방식을 쓰면 2GB 미만 DRAM으로 동작할 수 있다고 추정한다. 이는 모델 크기의 약 절반 resident라는 계산을 4-bit에 적용한 것이다.

논문이 실제로 증명한 것:

- Mac과 RTX 시스템에서 batch-1 decode
- flash cache를 통제한 selective loading
- sparse FFN model의 token당 latency
- 약 50-65% model-size memory budget

논문이 증명하지 않은 것:

- Android/iPhone의 실제 UFS/NVMe 및 NPU 배포
- 4-bit packed sparse compute kernel
- 긴 prompt prefill latency와 TTFT
- mobile OS memory pressure와 app coexistence
- p50/p95 tail latency
- battery energy/token과 표면 온도
- flash endurance와 장기 반복 read 영향

따라서 이 논문을 `스마트폰에서 7B가 빠르게 돈다`로 요약하면 과장이다. 정확한 결론은 `sparse weight working set을 예측하고 flash layout을 바꾸면 DRAM보다 큰 모델의 decode가 naive paging보다 훨씬 빨라질 수 있다`이다.

## Tensor와 Memory 손계산

### 7B weight 크기

| Precision | 계산 | 대략적 크기 |
|---|---:|---:|
| FP16 | 7B x 2 bytes | 14GB |
| INT8 | 7B x 1 byte | 7GB |
| INT4 | 7B x 0.5 byte | 3.5GB |

Scale, zero-point, group metadata, alignment padding은 제외한 이론값이다.

### Bundled neuron 크기

`d_model=4096`일 때:

| Precision | `2*d_model*bytes` | Chunk |
|---|---:|---:|
| FP32 | 2 x 4096 x 4 | 32KiB |
| FP16 | 2 x 4096 x 2 | 16KiB |
| INT8 | 2 x 4096 x 1 | 8KiB |
| INT4 packed | 2 x 4096 x 0.5 | 4KiB |

정밀도를 낮추면 byte는 줄지만 chunk도 작아져 random read throughput이 나빠질 수 있다. INT4에서 여러 neuron을 한 physical chunk로 추가 bundling해야 32KiB 이상 read를 유지할 수 있다. 이는 quantization과 flash layout을 공동 설계해야 한다는 뜻이다.

### KV cache와의 경쟁

Layer 수 `L`, KV heads `H_kv`, head dimension `d_h`, context length `T`, byte 수 `s`라 하면 batch 1 KV cache는 대략 다음과 같다.

```math
M_{KV}=2LTH_{kv}d_hs
```

Context가 길어지면 KV cache가 선형 증가해 FFN window cache 예산이 줄어든다. 논문 실험은 prompt 128 + generation 256 중심이므로 8K, 32K context에서 같은 52% resident budget을 보장하지 않는다.

## 구현 시 핵심 Data Path

```python
def decode_one_token(token, state):
    h = embed(token)

    for layer_id, layer in enumerate(layers):
        h = layer.attention(h, state.kv[layer_id])

        active = layer.predictor(h) > layer.threshold
        cached = state.ffn_cache[layer_id].active_neurons()
        missing = active & ~cached

        # Physical file offsets must already follow bundled layout.
        chunks = parallel_direct_read(
            file=layer.flash_file,
            neuron_ids=missing,
            no_os_cache=True,
        )

        state.ffn_cache[layer_id].evict_outside_window(active)
        state.ffn_cache[layer_id].insert_compact(chunks)

        h = layer.sparse_ffn(h, state.ffn_cache[layer_id], active)
        state.ffn_cache[layer_id].record(active)

    return sample(lm_head(h)), state
```

실제 구현에서는 I/O와 compute를 overlap하고, read completion queue, packed quantization metadata, GPU upload, failure handling을 추가해야 한다. Layer `l`을 계산하는 동안 `l+1` predictor 결과를 아직 알 수 없으므로 prefetch scheduling도 단순하지 않다.

## 강점

1. **문제를 memory hierarchy 관점에서 정확히 분해**: FLOPs가 아니라 byte, chunk, memory movement를 주 지표로 삼는다.
2. **Algorithm과 storage layout을 공동 설계**: activation sparsity만 쓰지 않고 flash access granularity에 맞춘다.
3. **I/O, memory management, compute를 따로 보고**: bottleneck 이동을 확인할 수 있다.
4. **Ablation이 명확**: hybrid, predictor, windowing, bundling을 순차적으로 추가한다.
5. **여러 model과 backend에서 검증**: OPT, Falcon, Persimmon, Phi-2, Llama 2 및 CPU/Metal/RTX를 포함한다.
6. **Negative result 공개**: co-activation bundling이 왜 실패했는지 제시한다.
7. **OS cache를 통제**: 숨은 DRAM 사용으로 결과가 왜곡되는 것을 줄인다.

## 한계

### Sparse model이 필요

Dense SwiGLU LLM에 그대로 적용할 수 없다. Relufication/FATReLU와 predictor training이 필요하고, model별/layer별 rank와 threshold를 조정한다.

### Prefill과 multi-batch 미평가

논문은 batch 1 autoregressive decode가 중심이다. Prompt processing은 여러 token을 matrix 형태로 처리하므로 neuron working set과 I/O 패턴이 다르다. TTFT가 중요한 chat/VLM에서는 이 공백이 크다.

### 비교 baseline의 해석

큰 speedup은 DRAM 부족으로 매 token GB 단위 weight를 읽는 naive/hybrid 대비다. Model이 압축되어 DRAM에 완전히 상주하는 실용 baseline과의 latency, energy, quality 비교가 없다.

### Hardware-specific engineering

Direct I/O alignment, flash controller, thread 수, chunk size, unified/discrete GPU 구조에 따라 최적점이 달라진다. M1/M2 결과를 Android UFS에 그대로 이식할 수 없다.

### Accuracy 평가 범위

몇 개 zero-shot task와 MMLU, perplexity를 보고하지만 long-form generation 품질과 safety regression은 충분히 다루지 않는다.

### Power와 thermal

Sparse 방식의 total energy가 더 높을 수 있음을 인정하지만 정량 분석이 없다. Flash를 지속적으로 읽는 workload의 thermal 및 battery 영향도 빠져 있다.

## 온디바이스 VLM에 적용할 때

VLM은 vision encoder, projector, LLM으로 나뉜다. LLM in a Flash는 주로 autoregressive LLM의 FFN weight를 다룬다.

```text
image
 -> vision encoder resident or on-demand
 -> visual tokens
 -> LLM prefill: large prompt batch, paper가 충분히 평가하지 않음
 -> LLM decode: selective flash loading 적용 가능
```

FastVLM처럼 visual token 수를 줄이면 두 효과가 있다.

- Prefill compute와 TTFT 감소
- KV cache 감소로 FFN window cache에 더 많은 DRAM 사용 가능

따라서 ROI 기반 visual token compression과 flash-resident LLM은 서로 보완적이다. 다만 vision encoder weight까지 같은 flash scheduler에 넣으면 camera pipeline과 I/O contention이 생길 수 있다.

실용적인 선택 순서는 다음이 합리적이다.

1. 먼저 AWQ/GPTQ 등으로 model이 DRAM에 들어가는지 확인한다.
2. 들어가면 DRAM-resident INT4가 대개 latency와 energy에서 유리하다.
3. 그래도 들어가지 않을 때 sparse flash loading을 고려한다.
4. Vision token, KV cache, draft model까지 포함한 전체 memory budget을 다시 계산한다.

## 권장 재현 실험

### 실험 1: Chunk size와 thread sweep

```text
chunk: 4, 8, 16, 32, 64, 128 KiB
threads: 1, 2, 4, 8, 16, 32
cache: buffered vs direct/no-cache
```

Sequential GB/s뿐 아니라 random p50/p95 latency와 CPU overhead를 기록한다.

### 실험 2: Window Pareto curve

`k={0,1,2,4,8,16}`에 대해 다음을 측정한다.

- cached FFN bytes
- new bytes/token
- I/O latency
- memory management latency
- tokens/s
- predictor miss rate

Memory pressure가 생길 때 dynamic `k` controller도 비교한다.

### 실험 3: INT4 physical layout

한 neuron bundle이 4KiB에 불과한 7B INT4 조건에서 8개 neuron을 32KiB physical block으로 묶는다. Over-read와 throughput의 trade-off를 측정한다.

### 실험 4: Prefill과 decode 분리

Prompt length `128, 512, 2048, 8192`, generation `128, 512, 1024`로 다음을 분리한다.

- model open/load time
- TTFT
- prefill tokens/s
- decode tokens/s
- steady-state flash GB/s
- KV cache peak

### 실험 5: 실제 모바일 soak test

30분 반복 generation에서 배터리 power, energy/token, surface temperature, flash latency drift, OS memory kill 여부를 기록한다.

## 실험 기록 템플릿

```text
Model and sparsification method:
Parameter count / weight bits:
Predictor rank by layer:
Predictor threshold:
False positive / false negative rate:
Window definition and size:

Device / OS / flash type:
Direct I/O or page cache:
Chunk size / read threads:
DRAM budget:
Resident attention bytes:
FFN cache bytes:
KV cache bytes:

Prompt length / generation length / batch:
TTFT:
Prefill tokens/s:
Decode tokens/s:
I/O p50/p95:
Memory-management p50/p95:
Peak RSS / peak GPU memory:
Average power / energy per token:
Temperature / throttling:

Quality baseline:
Perplexity / MMLU / task accuracy:
Long-generation regression cases:
```

## 이 로드맵에서의 의미

앞 단계의 quantization은 model byte 수를 줄이고, token pruning은 activation과 KV cache를 줄인다. LLM in a Flash는 그래도 DRAM에 들어가지 않는 model을 위해 storage를 memory hierarchy의 일부로 사용한다.

```text
quantization -> total model bytes 감소
activation sparsity -> token별 필요한 FFN weight 감소
windowing -> 시간축 weight reuse 증가
bundling -> flash request 효율 증가
preallocation -> DRAM 내부 copy 감소
```

이 논문의 가장 중요한 교훈은 `압축률`만으로 배포 가능성을 판단하지 말라는 것이다. 같은 3.5GB INT4 model도 physical layout, read granularity, KV cache, OS cache, sparse kernel 지원에 따라 latency와 peak memory가 크게 달라진다.

## 최종 평가

LLM in a Flash는 DRAM보다 큰 LLM을 실행하는 문제를 단순한 weight offload가 아니라 **sparse working set prediction + temporal cache + flash-friendly layout + compact DRAM management** 문제로 재구성했다. OPT-6.7B에서 token당 flash traffic을 13.4GB에서 0.2GB 수준으로 줄이고, bundling으로 sparse read throughput을 높인 결과는 hardware-aware algorithm의 좋은 사례다.

하지만 이 결과는 `모든 스마트폰에서 7B가 실시간 동작한다`는 증거가 아니다. Sparse model 준비, predictor 학습, 약 절반 model-size DRAM, batch-1 decode, Mac/RTX 중심 측정이라는 조건이 있다. INT4 smartphone 수치는 projection이며 4-bit sparse kernel과 실제 mobile flash 검증이 빠져 있다.

온디바이스 연구에서 이 논문은 최종 recipe보다 사고방식이 더 중요하다. 모델이 안 들어갈 때 먼저 byte를 줄이고, 다음으로 필요한 byte만 선택하며, 그 다음 access를 큰 연속 chunk로 만들고, 마지막으로 DRAM 내부 movement까지 측정해야 한다. 최종 평가는 TTFT와 tokens/s뿐 아니라 peak memory, p95 I/O, energy/token, temperature, 장시간 throttling까지 포함해야 한다.

## 체크리스트

- [ ] 대상 LLM이 실제로 exact activation sparsity를 갖는가?
- [ ] Dense 원본 대비 sparsification 정확도 손실을 분리했는가?
- [ ] Predictor false negative가 long generation 품질에 미치는 영향을 측정했는가?
- [ ] OS page cache를 포함했는지 명시했는가?
- [ ] Flash chunk size와 parallel read 수를 장치에서 sweep했는가?
- [ ] Weight, predictor, FFN window, KV cache의 합이 DRAM budget 안에 드는가?
- [ ] Prefill/TTFT와 decode를 분리했는가?
- [ ] INT4 bundle이 너무 작아지는 문제를 physical layout에서 해결했는가?
- [ ] p50/p95 latency와 tokens/s를 함께 기록했는가?
- [ ] 순간 power가 아니라 energy/token과 thermal throttling을 측정했는가?

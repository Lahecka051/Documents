# 46. RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer

## 논문 정보

- 제목: **RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer**
- 저자: Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu
- 공개: arXiv:2407.17140v1, 2024-07-24, Technical Report
- 논문: [arXiv](https://arxiv.org/abs/2407.17140) / [PDF](https://arxiv.org/pdf/2407.17140)
- 원본 파일: `46_RT_DETRv2.pdf`
- 공식 구현: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- 핵심 키워드: real-time DETR, deformable attention, multi-scale sampling, discrete sampling, deployment, bag-of-freebies

## 한눈에 보는 요약

RT-DETRv2는 새 detector를 처음부터 설계한 논문이라기보다 **RT-DETR의 decoder sampling과 training recipe를 배포 친화적으로 다듬은 4쪽짜리 기술 보고서**다. 큰 방향은 세 가지다.

```text
RT-DETR
  + scale마다 서로 다른 deformable-attention sampling point 수
  + grid_sample을 대체할 수 있는 discrete_sample
  + 동적 augmentation 종료 + model scale별 backbone learning rate
  = RT-DETRv2
```

첫째, 서로 해상도와 의미 수준이 다른 feature scale에 같은 수의 sampling point를 강제하지 않는다. 둘째, 일부 inference runtime에서 지원하기 까다로운 `grid_sample`의 bilinear interpolation을 좌표 반올림과 discrete gather로 바꿀 수 있게 한다. 셋째, 마지막 2 epoch에는 강한 augmentation을 끄고, ResNet18부터 ResNet101까지 backbone 크기에 따라 learning rate를 다르게 준다.

논문 보고값으로 RT-DETRv2-S는 COCO `AP 47.9`, `AP50 64.9`, T4 TensorRT FP16 batch 1에서 `217 FPS`를 기록한다. 같은 표의 RT-DETR-S보다 AP가 `+1.4` 높고 FPS는 동일하다. 그러나 이 속도 표는 discrete sampling 자체의 latency 개선을 측정한 표가 아니며, NMS 포함 여부, peak memory, p95 latency, 전력은 보고하지 않는다.

## 이 논문이 해결하려는 문제

### RT-DETR의 장점과 남은 배포 마찰

원래 DETR은 object query와 bipartite matching으로 anchor와 NMS 의존성을 줄였지만, 긴 multi-scale sequence와 decoder 때문에 실시간화가 쉽지 않았다. RT-DETR은 efficient hybrid encoder, uncertainty-minimal query selection, deformable decoder를 결합해 실시간 영역으로 내려왔다.

RT-DETRv2가 지적하는 남은 문제는 다음과 같다.

1. 모든 scale에 동일한 sampling point 수를 주면 scale별 정보 밀도 차이를 반영하지 못한다.
2. `grid_sample`은 YOLO 계열에 흔하지 않은 연산이며 runtime에 따라 변환, plugin, CPU fallback이 필요할 수 있다.
3. detector 크기가 크게 다른데도 같은 backbone learning rate를 쓰면 scale별 최적화가 덜 될 수 있다.
4. 학습 말기까지 강한 augmentation을 유지하면 실제 평가 domain에 적응할 시간이 부족할 수 있다.

중요하게도 RT-DETRv2는 backbone, hybrid encoder, query selection, set-prediction loss를 전면 교체하지 않는다. 논문은 "framework remains the same"이라고 명시하며 decoder의 deformable attention과 학습 전략만 수정한다.

## 전체 pipeline과 tensor 흐름

논문은 RT-DETRv2 전체 architecture figure와 feature channel을 다시 제시하지 않는다. 따라서 아래는 **RT-DETR에서 상속되는 구조를 설명한 개념 shape**이며, v2 보고서가 직접 명시한 고정 숫자가 아니다.

```text
input image [B, 3, H, W]
 -> ResNet backbone
 -> multi-scale maps {C3, C4, C5}
 -> hybrid encoder / channel projection
 -> encoded multi-scale features {F_l}
 -> query selection + learned content/position
 -> deformable decoder x D
 -> per-query class logits + bounding box
 -> set predictions
```

Scale `l`의 feature를 다음처럼 둔다.

```math
F_l\in\mathbb{R}^{B\times C\times H_l\times W_l},
\qquad
X_l\in\mathbb{R}^{B\times N_l\times C},
\qquad
N_l=H_lW_l
```

Decoder query는 보통 다음 shape로 생각할 수 있다.

```math
Q\in\mathbb{R}^{B\times N_q\times C}
```

각 query는 모든 spatial token에 dense attention을 만들지 않고, reference point 주변의 제한된 위치만 scale별로 sampling한다. 이것이 고해상도 feature pyramid를 다루면서 DETR decoder 비용을 제한하는 핵심이다.

## Multi-scale deformable attention

Head `m`, feature level `l`, sampling point `k`에 대해 출력을 단순화하면 다음과 같다.

```math
y_q=
\sum_{m=1}^{M}W_m
\left[
\sum_{l=1}^{L}\sum_{k=1}^{K_l}
A_{qmlk}\,
W'_mF_l\!\left(\phi_l(\hat p_q)+\Delta p_{qmlk}\right)
\right]
```

- `q`: object query index
- `M`: attention head 수
- `L`: feature level 수
- `K_l`: level `l`에서 query-head당 sampling point 수
- `A_qmlk`: sampling weight, 보통 해당 query-head 안에서 정규화
- `p_hat_q`: normalized reference point
- `phi_l`: normalized 좌표를 level `l`의 feature 좌표로 바꾸는 함수
- `Delta p_qmlk`: query에서 예측한 offset

기존 설정을 단순화하면 모든 level에 같은 `K`를 써서 `K_l=K`로 둔다. RT-DETRv2의 변경은 `K_1, K_2, ..., K_L`을 서로 다르게 둘 수 있게 하는 것이다.

```math
K_1=K_2=\cdots=K_L
\quad\longrightarrow\quad
(K_1,K_2,\ldots,K_L)\text{를 scale별로 지정}
```

논문은 어떤 scale에 몇 개를 배정했는지 구체적인 vector를 공개하지 않는다. 따라서 "fine scale에 항상 더 많이 준다" 같은 규칙은 원문에서 확인할 수 없다. 논문이 보여주는 것은 총 sampling point를 줄여도 성능 저하가 완만하다는 결과다.

## `grid_sample`과 `discrete_sample`의 차이

### Bilinear `grid_sample`

연속 좌표 `(u,v)`에서 feature를 읽을 때 주변 네 pixel을 보간한다.

```math
F(u,v)=\sum_{i\in\lbrace\lfloor u\rfloor,\lceil u\rceil\rbrace}
\sum_{j\in\lbrace\lfloor v\rfloor,\lceil v\rceil\rbrace}
w_{ij}(u,v)F[i,j]
```

좌표가 조금 움직이면 보간 weight가 연속적으로 바뀌므로 sampling offset predictor까지 gradient가 전달된다. 반면 gather, weight 계산, boundary 처리까지 포함한 fused `grid_sample` 지원 여부가 runtime마다 다르다.

### Discrete sampling

RT-DETRv2는 좌표를 가까운 integer 위치로 반올림하고 해당 feature를 직접 읽는다.

```math
F_{\mathrm{disc}}(u,v)=F[\mathrm{round}(u),\mathrm{round}(v)]
```

이 방식은 bilinear interpolation을 제거하고 일반적인 rounding, index calculation, gather에 가깝게 바꾼다. 하지만 `round`는 거의 모든 점에서 좌표에 대한 gradient가 0이고 경계에서는 미분 불가능하다.

논문의 학습 절차는 이를 다음처럼 우회한다.

```text
6x pretraining: grid_sample 사용, offset predictor까지 정상 학습
1x fine-tuning: discrete_sample로 교체, offset 예측 parameter gradient 차단
deployment: discrete_sample 사용
```

즉 처음부터 discrete operation만으로 offset을 학습하는 것이 아니다. 이미 학습된 sampling geometry를 고정하고, 나머지 weight가 양자화된 위치 읽기에 적응하도록 fine-tuning하는 방식이다.

## Sampling point 수와 계산 복잡도

논문은 전체 decoder에서 사용하는 sampling point 수를 다음처럼 센다.

```math
N_{\mathrm{sample}}
=N_{\mathrm{head}}
\left(\sum_lK_l\right)
N_{\mathrm{query}}
N_{\mathrm{decoder}}
```

RT-DETRv2-S의 ablation은 `86,400 -> 64,800 -> 43,200 -> 21,600` point를 비교한다. 이는 **전체 decoder layer에 걸친 연산 횟수 지표**이지 한 시점에 모두 보존되는 tensor 크기가 아니다.

Dense cross-attention이 query마다 모든 multi-scale token을 본다면 score 원소 수는 대략 다음과 같다.

```math
O\!\left(BN_q\sum_lH_lW_l\right)
```

Deformable attention은 이를 다음으로 바꾼다.

```math
O\!\left(BN_qM\sum_lK_l\right)
```

단, theoretical point 감소가 latency와 정확히 비례하지는 않는다. 작은 gather가 memory-bound일 수 있고, 좌표 계산, kernel launch, feature layout 변환이 고정 비용으로 남기 때문이다. 논문도 point별 실제 latency ablation은 제공하지 않는다.

## Batch 1 shape와 activation memory 계산 예

다음은 v2 보고서가 shape를 공개하지 않은 부분을 보충하기 위한 **reviewer 계산 예시**다. `640×640`, stride `{8,16,32}`, projected channel `C=256`, `B=1`을 가정한다.

| Level | Shape `B×C×H×W` | Element 수 | FP16 payload |
| --- | ---: | ---: | ---: |
| P3 | `1×256×80×80` | 1,638,400 | 3.125 MiB |
| P4 | `1×256×40×40` | 409,600 | 0.781 MiB |
| P5 | `1×256×20×20` | 102,400 | 0.195 MiB |
| 합계 | 8,400 spatial tokens | 2,150,400 | **4.102 MiB** |

이 값은 세 projected feature payload만 더한 것이다. Backbone activation, encoder intermediate, Q/K/V projection, residual, workspace, allocator fragmentation은 포함하지 않으므로 peak memory가 아니다.

`N_q=300`, `C=256`을 추가로 가정하면 query state는 다음과 같다.

```math
300\times256=76{,}800\text{ elements}
```

FP16 payload는 약 `0.146 MiB`다. Table 3의 86,400 sampling weight를 전 layer에 걸쳐 단순 합산해도 FP16 `0.165 MiB`지만, 실제 구현에는 2D offset, index, sampled value, backward buffer가 추가된다. 또한 decoder layer를 순차 실행하면 전체 layer 합계가 곧 peak tensor 크기가 되지 않는다.

이 계산의 핵심은 parameter보다 **multi-scale feature와 sampled-value temporary**가 activation memory를 지배할 수 있다는 점이다.

## Parameter와 latency를 읽는 법

Table 2의 논문 보고값과 FPS를 latency로 환산한 reviewer 계산은 다음과 같다.

| Model | Backbone | Params | FPS, 논문 | `1000/FPS`, 계산 | AP | AP50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RT-DETRv2-S | ResNet18 | 20M | 217 | 4.61 ms | 47.9 | 64.9 |
| RT-DETRv2-M | ResNet34 | 31M | 161 | 6.21 ms | 49.9 | 67.5 |
| RT-DETRv2-M* | ResNet50 | 36M | 145 | 6.90 ms | 51.9 | 69.9 |
| RT-DETRv2-L | ResNet50 | 42M | 108 | 9.26 ms | 53.4 | 71.6 |
| RT-DETRv2-X | ResNet101 | 76M | 74 | 13.51 ms | 54.3 | 72.8 |

측정 환경은 T4 GPU, TensorRT FP16, batch 1, fixed `640×640`이다. `1000/FPS`는 reciprocal throughput일 뿐 paper가 직접 보고한 end-to-end latency가 아니다. 전처리, device transfer, post-processing 포함 여부가 명시되지 않아 camera pipeline latency로 그대로 사용하면 안 된다.

20M parameter인 S model의 순수 weight payload를 계산하면 FP32 약 `76.3 MiB`, FP16 약 `38.1 MiB`, INT8 약 `19.1 MiB`다. 이는 serialization metadata, quantization scale, alignment, TensorRT engine workspace를 제외한 하한이다. 논문은 MACs/FLOPs, activation peak, engine size를 보고하지 않는다.

## Training scheme

### Dynamic data augmentation

초반에는 RT-DETR의 강한 augmentation을 유지하고 마지막 2 epoch에는 다음을 끈다.

- `RandomPhotometricDistort`
- `RandomZoomOut`
- `RandomIoUCrop`
- `MultiScaleInput`

학습 말기에 evaluation domain과 가까운 분포로 적응시키려는 설계다. 다만 이 항목만 단독으로 on/off한 AP ablation은 보고하지 않는다.

### Scale-adaptive backbone learning rate

Detector learning rate는 모두 `1e-4`로 두고 backbone만 model scale에 따라 바꾼다.

| Model | Backbone | `lr_backbone` | `lr_det` |
| --- | --- | ---: | ---: |
| S | ResNet18 | `1e-4` | `1e-4` |
| M | ResNet34 | `5e-5` | `1e-4` |
| L | ResNet50 | `1e-5` | `1e-4` |
| X | ResNet101 | `1e-6` | `1e-4` |

저자들의 설명은 작은 pretrained backbone은 feature quality가 낮으므로 더 크게 갱신하고, 큰 backbone은 더 보수적으로 갱신한다는 것이다. 이 learning-rate schedule 역시 독립 ablation이 없다.

공통 학습 조건은 COCO train2017, batch 16, AdamW, EMA decay `0.9999`다. `6x`와 `1x`가 정확히 몇 epoch인지 이 보고서 본문은 풀어 쓰지 않는다.

## Loss와 matching

RT-DETRv2 보고서는 새로운 loss 식을 제안하지 않는다. Class loss, box regression loss, decoder auxiliary loss, matching cost의 계수도 재기재하지 않는다. 따라서 다음과 같이 구분해야 한다.

- **v2에서 직접 바뀐 것**: sampling 방식과 training schedule
- **RT-DETR에서 상속된 것**: query-based set prediction과 학습 loss
- **이 PDF만으로 확정할 수 없는 것**: loss별 정확한 coefficient와 구현 세부

재현 시에는 v2 기술 보고서만 보고 loss를 새로 조합하지 말고, 논문이 링크한 공식 RT-DETR config의 version과 checkpoint를 고정해야 한다.

## 구현 pseudocode

```python
def deformable_sample(features, ref, offsets, attn, points_per_level,
                      mode="bilinear"):
    # features[l]: [B, C, H_l, W_l]
    # ref:         [B, Q, L, 2]
    # offsets:     [B, Q, heads, sum(K_l), 2]
    # attn:        [B, Q, heads, sum(K_l)]
    outputs = []
    cursor = 0

    for level, (feat, k_l) in enumerate(zip(features, points_per_level)):
        coord = ref[:, :, level] + offsets[..., cursor:cursor + k_l, :]

        if mode == "bilinear":
            sampled = grid_sample_bilinear(feat, coord)
        elif mode == "discrete":
            index = round_and_clip(coord, feat.shape[-2:])
            sampled = gather_2d(feat, index)
        else:
            raise ValueError(mode)

        weight = attn[..., cursor:cursor + k_l]
        outputs.append(weighted_sum(sampled, weight))
        cursor += k_l

    return sum(outputs)
```

Discrete fine-tuning에서는 offset predictor가 더 이상 바뀌지 않도록 명시적으로 막아야 한다.

```python
offsets = offset_head(query)
if sampling_mode == "discrete":
    offsets = offsets.detach()
```

실제 구현에서는 `detach()` 범위가 offset parameter에만 적용되는지 확인해야 한다. Query 전체를 detach하면 decoder 학습까지 차단할 수 있다.

## 핵심 실험 결과

### RT-DETR 대비

| Scale | RT-DETR AP | v2 AP | 변화 | RT-DETR AP50 | v2 AP50 | 변화 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S | 46.5 | 47.9 | +1.4 | 63.8 | 64.9 | +1.1 |
| M | 48.9 | 49.9 | +1.0 | 66.8 | 67.5 | +0.7 |
| M* | 51.3 | 51.9 | +0.6 | 69.6 | 69.9 | +0.3 |
| L | 53.1 | 53.4 | +0.3 | 71.3 | 71.6 | +0.3 |
| X | 54.3 | 54.3 | +0.0 | 72.7 | 72.8 | +0.1 |

작은 model일수록 gain이 크고 X에서는 AP gain이 없다. 따라서 recipe의 효과가 model scale에 독립적으로 일정하다고 볼 수 없다.

### Sampling point 수 ablation

| 총 point | AP | AP50 | 86,400 대비 |
| ---: | ---: | ---: | ---: |
| 86,400 | 47.9 | 64.9 | baseline |
| 64,800 | 47.8 | 64.8 | AP -0.1 |
| 43,200 | 47.7 | 64.7 | AP -0.2 |
| 21,600 | 47.3 | 64.3 | AP -0.6 |

Point를 75% 줄인 `86,400 -> 21,600`에서도 AP 감소는 0.6이다. 그러나 latency, bandwidth, memory가 얼마나 줄었는지는 측정하지 않았다. 이 표만으로 "4배 빨라진다"고 결론내릴 수 없다.

### Discrete sampling ablation

| Model | Grid AP / AP50 | Discrete AP / AP50 | reviewer 계산 변화 |
| --- | ---: | ---: | ---: |
| S | 47.9 / 64.9 | 47.4 / 64.8 | AP -0.5, AP50 -0.1 |
| M | 49.9 / 67.5 | 49.2 / 67.1 | AP -0.7, AP50 -0.4 |
| M* | 51.9 / 69.9 | 51.4 / 69.7 | AP -0.5, AP50 -0.2 |
| L | 53.4 / 71.6 | 52.9 / 71.3 | AP -0.5, AP50 -0.3 |

원문은 `AP50` 감소가 작다는 점을 강조한다. 반면 stricter IoU까지 평균한 AP는 `0.5-0.7` 감소한다. Bilinear interpolation을 없애면서 box localization precision이 일부 손실됐을 가능성이 있지만, 논문은 AP75나 object-size별 AP를 보고하지 않아 원인을 확정할 수 없다.

## 장점

- 새 architecture를 크게 바꾸지 않고 실무 배포를 가로막는 operator를 직접 다룬다.
- Scale별 point 수를 분리해 deformable attention의 불필요한 uniform constraint를 제거한다.
- Point 수 감소와 discrete replacement에 대한 정확도 ablation을 제공한다.
- S부터 X까지 같은 방향의 recipe를 평가해 model scale별 gain 차이를 보여준다.
- TensorRT FP16, batch 1, fixed resolution을 명시해 throughput 표의 조건이 비교적 분명하다.

## 한계와 헷갈리기 쉬운 부분

### 1. 짧은 기술 보고서라 재현 정보가 부족하다

Scale별 실제 `K_l`, decoder/query/head 수, MACs, loss coefficient, exact epoch 수가 본문에 없다. 공식 config 없이는 Table 3의 setting을 완전히 복원하기 어렵다.

### 2. Bag-of-freebies별 독립 효과가 분리되지 않는다

Dynamic augmentation, scale-adaptive learning rate, distinct point allocation을 각각 단독으로 켠 ablation이 없다. Table 2의 gain을 어느 요소가 만들었는지 알 수 없다.

### 3. Deployment 주장을 latency로 증명하지 않는다

`grid_sample` 지원 제약을 없앤다는 engineering 이점은 타당하지만, discrete version의 TensorRT/ONNX/NCNN latency와 peak memory를 보고하지 않는다. "지원된다"와 "더 빠르다"는 다른 주장이다.

### 4. 평균 FPS만 있고 tail latency가 없다

실시간 시스템에는 p50/p95, warm-up, engine build mode, host-device copy, preprocessing이 중요하다. 논문은 이 정보를 제공하지 않는다.

### 5. AP50만 보면 localization 손실을 놓칠 수 있다

Discrete sampling은 AP50 감소가 작지만 전체 AP 감소는 더 크다. Small-object AP와 AP75가 없어 integer sampling이 세밀한 위치 추정에 미치는 영향을 판단하기 어렵다.

### 6. Offset gradient 차단의 안정성 분석이 없다

6x grid pretraining 뒤 1x discrete fine-tuning이라는 recipe만 제시한다. Fine-tuning 길이, straight-through estimator, offset 일부 재학습 같은 대안 비교는 없다.

## 온디바이스 관점

RT-DETRv2에서 가장 중요한 온디바이스 기여는 parameter 감소가 아니라 **operator surface 축소**다. NPU compiler가 `grid_sample`을 지원하지 않으면 해당 layer만 CPU/GPU로 fallback되어 device synchronization과 tensor copy가 전체 latency를 지배할 수 있다. Discrete gather가 target backend에서 native로 fuse된다면 theoretical FLOPs 변화보다 큰 end-to-end 이득이 날 수 있다.

반대로 다음 위험도 있다.

- Dynamic gather와 integer index가 NPU에서 여전히 비효율적일 수 있다.
- `round`, `clip`, layout transpose가 각각 별도 kernel로 분리될 수 있다.
- Offset/weight tensor가 NHWC와 NCHW 사이를 왕복하면 bandwidth 비용이 커진다.
- INT8 feature에서 bilinear를 제거해도 attention weight와 box head가 FP16 fallback될 수 있다.
- Camera input의 aspect ratio가 바뀌면 dynamic shape 때문에 compiler optimization이 약해질 수 있다.

따라서 모바일 검증은 다음 네 variant를 동일 device에서 비교해야 한다.

| Variant | 확인 목적 |
| --- | --- |
| grid, original points | 정확도/latency 기준선 |
| grid, reduced points | point 수 자체의 효과 |
| discrete, original points | operator replacement 효과 |
| discrete, reduced points | 최종 배포 조합 |

각 variant에 대해 `AP/AP50/AP75/APS`, p50/p95 latency, peak RSS/device memory, energy per frame, thermal steady state를 기록해야 한다. NMS-free라도 top-k selection과 box decoding이 완전히 비용 0인 것은 아니므로 end-to-end profiler 범위에 포함한다.

## 재현 체크리스트

- [ ] arXiv v1과 사용한 official commit/config hash를 기록했는가?
- [ ] COCO train2017/val2017와 `640×640` preprocessing을 고정했는가?
- [ ] S/M/M*/L/X의 backbone과 learning rate가 Table 1과 일치하는가?
- [ ] AdamW, batch 16, EMA decay `0.9999`를 확인했는가?
- [ ] 마지막 2 epoch에 네 augmentation을 정확히 끄는가?
- [ ] `points_per_level` vector와 총 point 계산을 log로 남겼는가?
- [ ] Grid 6x 뒤 discrete 1x 순서로 fine-tuning했는가?
- [ ] Discrete 단계에서 offset predictor만 gradient를 차단했는가?
- [ ] 좌표 rounding과 out-of-bound clipping 규칙이 export 후에도 동일한가?
- [ ] ONNX/TensorRT/NPU graph에 CPU fallback이 없는가?
- [ ] FPS와 reciprocal latency를 혼동하지 않고 실제 p50/p95를 측정했는가?
- [ ] Pre/post-processing과 device transfer를 포함한 latency도 별도로 기록했는가?
- [ ] AP50뿐 아니라 AP, AP75, small-object AP를 비교했는가?
- [ ] Peak activation과 workspace를 target runtime에서 측정했는가?

## 로드맵에서의 연결

RT-DETRv2는 Faster R-CNN/FPN에서 시작해 FCOS, DETR, Deformable DETR을 거친 뒤 읽을 때 의미가 분명해진다.

```text
FPN: multi-scale feature hierarchy
 -> FCOS/YOLOX/RTMDet: dense real-time one-stage detection
 -> DETR: query와 set prediction, NMS-free
 -> Deformable DETR: sparse multi-scale sampling
 -> RT-DETR: encoder/decoder를 실시간화
 -> RT-DETRv2: sampling operator와 training recipe를 배포 친화적으로 수정
```

로드맵의 동일 데이터, 해상도, 장치 비교에서는 RTMDet 또는 YOLOX와 나란히 두고 다음 질문을 검증하기 좋다.

- NMS 제거가 실제 end-to-end p95 latency를 줄이는가?
- Decoder layer 수와 sampling point 수를 줄일 때 AP와 latency가 어떻게 변하는가?
- Small object에서 discrete sampling 손실이 커지는가?
- Unsupported operator 하나가 전체 NPU partition을 얼마나 깨뜨리는가?

## 최종 평가

RT-DETRv2의 핵심은 새로운 detection paradigm이 아니라 **좋은 real-time model도 deployment operator와 학습 recipe까지 맞춰야 실제 장치에서 쓸 수 있다**는 점이다. Scale별 sampling point, discrete sampling, late-stage augmentation 조정은 모두 실용적인 방향이며 S model의 `+1.4 AP`는 의미가 있다.

다만 보고서가 매우 짧아 각 개선의 독립 효과, discrete version의 실제 latency, memory와 tail latency가 빠져 있다. 따라서 이 논문을 "RT-DETRv2가 모든 장치에서 더 빠르다는 증명"으로 읽기보다, **실험해야 할 배포 가설과 공식 baseline을 제공한 기술 보고서**로 읽는 것이 정확하다. 온디바이스 연구에서는 Table 2의 FPS를 재인용하는 데서 끝내지 말고 grid/discrete와 point 수를 교차한 4-way ablation을 target device에서 다시 측정해야 한다.

# 56. EdgeSAM: Prompt-In-the-Loop Distillation for SAM

## 논문 정보

- 제목: **EdgeSAM: Prompt-In-the-Loop Distillation for SAM**
- 저자: Chong Zhou, Xiangtai Li, Chen Change Loy, Bo Dai
- 소속: S-Lab, Nanyang Technological University, The University of Hong Kong
- 게재: International Journal of Computer Vision, 2025년 8월 4일 accepted
- arXiv: [2312.06660](https://arxiv.org/abs/2312.06660)
- 검토 버전: arXiv v3, 2025년 9월 7일
- 원본 파일: `56_EdgeSAM.pdf`
- 핵심 키워드: SAM acceleration, CNN encoder, knowledge distillation, interactive prompts, on-device inference

## 1. 한 문장 요약

EdgeSAM은 SAM의 ViT-H 이미지 인코더를 RepViT-M1 기반 CNN과 작은 FPN으로 교체하되, 단순 feature matching에 그치지 않고 teacher와 student의 마스크 불일치 영역에서 교정 점을 반복 샘플링하는 prompt-in-the-loop distillation을 사용해, 1024 입력에서 iPhone 14 기준 38.7 FPS를 달성한 경량 prompt segmenter다.

## 2. 문제 정의

원본 SAM은 같은 이미지에 여러 프롬프트를 넣을 때 디코더가 빠르지만, 첫 image embedding을 만드는 ViT-H가 매우 무겁다. 이 논문이 인용하는 원본 SAM의 규모는 다음과 같다.

- 641.1M parameters
- `1024 x 1024` 입력에서 2734.8 GFLOPs
- NVIDIA 2080 Ti에서 4.3 FPS
- iPhone 14에서는 메모리와 계산량 때문에 OOM

가장 단순한 해결은 SAM decoder는 유지하고 encoder만 작은 네트워크로 바꾸어 teacher feature를 MSE로 맞추는 것이다. MobileSAM이 이 방식을 취한다. 하지만 EdgeSAM 저자들은 다음 두 문제가 남는다고 본다.

1. teacher의 최종 feature를 닮는 것만으로는 프롬프트가 마스크로 변환되는 과제 지식을 충분히 전달하지 못한다.
2. SA-1B는 instance와 part가 섞인 multi-grained 데이터이므로 점 하나처럼 모호한 프롬프트에서 원하는 granularity가 데이터셋마다 달라진다.

따라서 EdgeSAM은 세 단계로 문제를 푼다.

```text
stage 1: encoder-only feature distillation
stage 2: prompt-in-the-loop mask distillation
stage 3: optional RPN for dataset-specific granularity prior
```

## 3. 원본 SAM에서 무엇을 유지하고 무엇을 바꾸는가

원본과 EdgeSAM의 공통 인터페이스는 다음과 같다.

```text
input image:       [B, 3, 1024, 1024]
image embedding:  [B, 256, 64, 64]
sparse prompts:   positive/negative points and one box
mask decoder:     two-way Transformer + dynamic mask head
outputs:          four internal mask candidates and IoU scores
```

EdgeSAM이 바꾸는 주된 부분은 image encoder다.

| 구성요소 | 원본 SAM | EdgeSAM |
|---|---|---|
| image encoder | ViT-H 계열 | RepViT-M1 CNN + tiny FPN |
| prompt encoder | SAM pretrained module | 구조와 표현 유지 |
| mask decoder | lightweight two-way Transformer | 구조/초기 weight 상속 후 distillation |
| 입력 해상도 | 1024 x 1024 | 1024 x 1024 |
| image embedding | 256 x 64 x 64 | 동일 |
| 파라미터 | 641.1M | 9.6M |
| 계산량 | 2734.8 GFLOPs | 22.1 GFLOPs |

decoder가 원본 SAM 전체 파라미터의 0.6%에 불과하므로 이를 새로 설계해 얻을 이득보다 pretrained behavior를 유지하는 이득이 크다는 판단이다. 핵심 연구 질문은 작은 encoder가 원본 decoder와 잘 맞는 feature를 어떻게 만들게 할 것인가다.

## 4. CNN encoder와 resolution alignment

### 4.1 왜 RepViT-M1인가

저자들은 세 종류의 경량 backbone을 비교한다.

- ViT 계열: TinyViT-5M
- hybrid 계열: EfficientViT-B1
- CNN 계열: RepViT-M1

desktop FLOPs나 FPS만 보면 차이가 작아 보일 수 있지만, Apple Neural Engine처럼 convolution 경로가 성숙한 모바일 가속기에서는 순수 CNN이 훨씬 유리했다. FPN alignment 설정에서 iPhone 14의 encoder latency는 RepViT-M1이 14 ms이며 TinyViT-5M보다 14배, EfficientViT-B1보다 4배 빠르다고 논문은 보고한다.

이는 `경량 모델 = 적은 FLOPs`만으로 선택하면 안 된다는 좋은 사례다. 같은 1024 입력에서도 attention, reshape, transpose, LayerNorm이 가속기 밖으로 fallback되면 작은 ViT가 CNN보다 훨씬 느릴 수 있다.

### 4.2 tiny FPN

MobileSAM은 student의 마지막 두 stage downsampling을 제거해 SAM의 `64 x 64` 출력에 맞춘다. EdgeSAM은 원래 backbone의 downsampling을 유지하고 마지막 두 stage feature를 FPN으로 결합한다.

```text
late feature F4 ---- projection to 256 ---- upsample --+
                                                      +--> add --> 256x64x64
earlier feature F3 -- projection to 256 --------------+
```

논문은 RepViT-M1 각 stage의 정확한 channel/shape 표를 다시 제시하지 않는다. 따라서 위 흐름에서 `F3`, `F4`의 세부 channel 수를 EdgeSAM 본문만으로 임의 확정해서는 안 된다. 재현 시 사용한 RepViT 구현과 checkpoint config를 함께 고정해야 한다.

FPN은 두 가지 목적을 갖는다.

- SAM decoder가 기대하는 `C=256, H=W=64` 인터페이스 보존
- 고해상도 stage의 공간 정보와 저해상도 semantic feature 결합

## 5. 1단계: Encoder-only knowledge distillation

teacher SAM encoder를 `T_enc`, student EdgeSAM encoder를 `S_enc`, 입력을 `I`라 하면 feature loss는 단순 MSE다.

```text
L_p = MSE(T_enc(I), S_enc(I))
```

두 출력은 projection과 FPN을 거쳐 같은 shape가 된다.

```text
T_enc(I), S_enc(I) in R^[B, 256, 64, 64]
```

이 단계는 teacher가 만든 image embedding의 pixel-wise 값에 student를 맞춘다. 그러나 다음 시도들은 뚜렷한 해결책이 아니었다.

- encoder-only 일정을 10 epochs에서 20 epochs로 연장
- structured distillation loss 추가
- channel-wise distillation loss 추가

SA-1K의 box/center 결과는 baseline `82.0/64.6`, 20 epochs `81.9/65.9`, structured `81.6/65.2`, channel-wise `81.8/64.9`였다. point는 일부 개선되지만 box가 떨어지고 전체 상한이 남는다. feature가 비슷하다는 조건이 prompt-conditioned decision boundary까지 같다는 조건은 아니기 때문이다.

## 6. 2단계: Prompt-in-the-loop distillation

### 6.1 decoder까지 학습 신호에 포함하는 이유

SAM decoder에는 다음 두 stream이 들어간다.

```text
image stream:  f in R^[B, 256, 64, 64]
token stream:  prompt embeddings p
             + four mask tokens m
             + one IoU token c
```

teacher/student의 attention map, token, refined image feature, IoU score 등 여러 중간 표현을 맞출 수 있다. 하지만 ablation에서 가장 효과적인 것은 teacher가 만든 최종 mask를 직접 student supervision으로 쓰는 단순한 방법이었다.

```text
L_d = L_mask(
        binary_threshold(T_dec(f_t, p, m, c)),
        S_dec(f_s, p, m, c)
      )

L_mask = L_Dice + L_BCE
```

teacher mask는 threshold해 pseudo ground truth로 쓰고, student mask logits에는 Dice와 BCE를 적용한다. teacher와 student는 같은 prompt embedding, mask token, IoU token을 사용한다. teacher는 고정하고 gradient는 student decoder뿐 아니라 student encoder까지 흐른다.

논문의 표현상 `p, m, c`는 공유되고 frozen이다. 실제 구현을 재현할 때는 prompt encoder와 token parameters 중 무엇이 optimizer에 들어가는지 checkpoint/config로 다시 확인해야 한다. 반면 student decoder는 SAM weight로 초기화한 뒤 기본 설정에서 encoder와 함께 fine-tuning한다.

### 6.2 동적 prompt sampling

각 학습 sample의 초기 프롬프트는 동일 확률로 box 또는 point다. teacher와 student가 같은 프롬프트로 마스크를 만든 뒤 둘이 다른 영역에서 교정 점을 뽑는다.

teacher의 선택 마스크를 `M_t`, 대응하는 student 마스크를 `M_s`라고 하면:

```text
FN = M_t AND NOT M_s
FP = M_s AND NOT M_t

sample from FN -> positive point
sample from FP -> negative point
```

새 점을 기존 프롬프트에 append하고 두 decoder를 다시 실행한다. 불일치는 teacher의 predicted IoU가 가장 높은 mask와 그에 대응하는 student mask 사이에서 계산한다. 박스는 객체당 하나만 지원하므로 correction loop에서는 새 박스를 뽑지 않고 점만 추가한다.

이 설계는 세 가지 효과를 갖는다.

1. box, point, box+point, 여러 point의 다양한 조합을 동적으로 만든다.
2. student가 실제로 틀린 영역에 학습을 집중하는 hard-example mining 역할을 한다.
3. 첫 점에서 모호했던 teacher도 추가 점을 받아 더 명확한 마스크를 supervision으로 제공한다.

### 6.3 학습 의사코드

```python
for image, initial_prompts in loader:
    with no_grad():
        teacher_feature = teacher_encoder(image)

    student_feature = student_encoder(image)
    prompt = random_choice(initial_prompts.point, initial_prompts.box)

    loss = 0
    for step in range(1 + correction_loops):
        with no_grad():
            teacher_masks, teacher_scores = teacher_decoder(
                teacher_feature, prompt
            )

        student_masks, _ = student_decoder(student_feature, prompt)
        k = argmax(teacher_scores)
        loss += dice_bce(student_masks[k], threshold(teacher_masks[k]))

        if step < correction_loops:
            new_point = sample_disagreement(
                teacher_masks[k], student_masks[k]
            )
            prompt = append(prompt, new_point)

    loss.backward()            # student encoder and decoder
    optimizer.step()
```

기본 correction loop 수는 1이다. 0 loop는 decoder를 한 번 학습하되 추가 교정점을 쓰지 않는 설정이며 encoder-only KD와는 다르다.

## 7. 3단계: Optional RPN granularity prior

### 7.1 왜 point prompt에 dataset prior가 필요한가

SA-1B에는 object instance뿐 아니라 part와 subpart가 섞여 있다. 사람 몸 위 center point 하나가 어떤 데이터셋에서는 사람 전체 정답이고, SA-1B의 다른 마스크에서는 옷이나 신체 일부일 수 있다. 실제로 원본 SAM조차 COCO center point에서 mIoU 53.6에 그친다.

box는 공간 범위를 제공해 granularity가 비교적 명확하지만, single point는 그렇지 않다. EdgeSAM은 COCO 같은 특정 배포 도메인의 prior를 명시적으로 넣는 선택형 RPN을 추가한다.

### 7.2 구조와 동작

encoder는 freeze하고 EfficientDet식 FPN과 shared detection head를 얹는다. 특정 데이터셋의 class-agnostic ground-truth box로 focal loss와 Huber loss를 사용해 학습한다.

추론 시에는 다음 과정을 거친다.

```text
point prompt
  -> RPN proposals
  -> point와 중심이 가까운 K proposals 선택
  -> confidence-weighted box merge
  -> merged box + original point
  -> SAM mask decoder
```

논문 본문은 `K nearest neighbors`라고만 쓰고 K의 수치를 명시하지 않는다. 이는 RPN 재현성의 빈틈이다. 구현 config를 확인하고 K, proposal threshold, merge 식, NMS 설정을 실험 기록에 남겨야 한다.

RPN은 필요할 때만 켤 수 있다. COCO center-point mIoU는 48.0에서 54.3으로 6.3 상승하지만, 특정 데이터셋의 granularity convention을 주입하므로 범용 behavior가 필요한 상황에서는 끄는 것이 맞다.

## 8. 학습 설정

### 8.1 Stage 1

- 학습 데이터: SA-1B 이미지의 1%, 약 110k 규모
- objective: encoder feature MSE
- epochs: 10
- batch size: 64
- optimizer: AdamW
- learning rate: `1.25e-2`에서 cosine decay로 `6.25e-7`

### 8.2 Stage 2

- 초기화: stage 1 encoder + 원본 SAM decoder weights
- 학습 데이터: 동일한 1% SA-1B
- objective: prompt-in-the-loop Dice + BCE mask distillation
- epochs: 5
- batch size: 16
- learning rate: `1e-4`에서 `1e-5`
- 이미지당 최대 instances: 16
- correction loops: 1

한 SA-1B 이미지에 평균 약 100 instances가 있어 모두 사용하면 VRAM이 넘친다. 8, 16, 32 prompts ablation에서 16이 기본 trade-off로 선택됐다.

### 8.3 Stage 3, optional

- 나머지 module freeze
- COCO class-agnostic boxes로 RPN 학습
- MMDetection 1x schedule
- focal loss + Huber loss

저자들은 teacher encoder embedding을 disk에 cache해 stage 1과 stage 2의 반복 teacher 계산을 줄였다. 학습 속도에는 유리하지만 저장 공간 요구가 크다.

리뷰어 계산으로 1% SA-1B를 110k 이미지, embedding을 `256 x 64 x 64`라 두면:

```text
per image FP16 embedding = 2 MiB
110,000 images x 2 MiB ~= 214.8 GiB
```

FP32라면 약 429.7 GiB다. 실제 저장 dtype, compression, sample 수에 따라 달라지며 논문은 cache disk 사용량을 보고하지 않는다. 재현 계획에 이 비용을 반드시 포함해야 한다.

## 9. Batch 1 tensor shape와 메모리

아래 값은 `1024 x 1024`, batch 1에 대한 **리뷰어 계산**이며 논문 보고 peak memory가 아니다.

| tensor | shape | FP16 크기 |
|---|---:|---:|
| 입력 이미지 | `1 x 3 x 1024 x 1024` | 6.00 MiB |
| student 최종 embedding | `1 x 256 x 64 x 64` | 2.00 MiB |
| teacher 최종 embedding | `1 x 256 x 64 x 64` | 2.00 MiB |
| prompt decoder용 펼친 image tokens | `1 x 4096 x 256` | 2.00 MiB |
| SAM upsampled mask feature | `1 x 32 x 256 x 256` | 4.00 MiB |
| 4 low-resolution masks | `1 x 4 x 256 x 256` | 0.50 MiB |

EdgeSAM weight 저장 하한은 다음과 같다.

```text
9.6M params, FP32 ~= 36.6 MiB
9.6M params, FP16 ~= 18.3 MiB
9.6M params, INT8 ~= 9.2 MiB
```

비교용 원본 SAM 641.1M의 FP16 weight만 약 1.19 GiB다. 따라서 파라미터 수는 약 `641.1 / 9.6 = 66.8`배 줄었다.

학습 batch 16에서 teacher와 student 최종 embedding만 각각 32 MiB다. 이미지당 16 prompt를 처리할 때 embedding을 prompt마다 물리 복제하면 student 한쪽만 512 MiB가 되므로, broadcast/view 또는 decoder 내부 batching을 사용해야 한다. 실제 peak memory는 RepViT 중간 activation, FPN, decoder workspace, autograd 저장 방식에 좌우되며 논문은 수치를 보고하지 않는다.

## 10. 효율 결과

### 10.1 논문 보고값

모든 GFLOPs는 `1024 x 1024` 입력 기준이다.

| 모델 | 학습 데이터 | iPhone 14 FPS | 2080 Ti FPS | Params | GFLOPs |
|---|---|---:|---:|---:|---:|
| SAM | SA-1B | OOM | 4.3 | 641.1M | 2734.8 |
| MobileSAM | 1% SA-1B | 4.9 | 103.5 | 9.8M | 38.2 |
| EfficientSAM-Ti | SA-1B + ImageNet | 4.9 | 103.5 | 9.8M | 38.2 |
| EdgeSAM | 1% SA-1B | 38.7 | 164.3 | 9.6M | 22.1 |
| EdgeSAM-RPN | 1% SA-1B | 34.1 | 123.9 | 9.8M | 22.2 |

2080 Ti에서는 ONNX로 compile하고 GPU 100% payload를 확인했다. iPhone 14에서는 coremltools로 변환하고 Xcode benchmark 도구로 측정했다.

### 10.2 FPS를 latency로 바꾼 리뷰어 계산

```text
EdgeSAM iPhone 14:      1000 / 38.7 = 25.84 ms
EdgeSAM-RPN iPhone 14:  1000 / 34.1 = 29.33 ms
MobileSAM iPhone 14:    1000 / 4.9  = 204.08 ms

EdgeSAM 2080 Ti:        1000 / 164.3 = 6.09 ms
SAM 2080 Ti:            1000 / 4.3   = 232.56 ms
```

논문 초록의 iPhone image encoding 26 ms는 전체 표의 38.7 FPS 역수와 거의 같다. 별도 backbone ablation에서는 RepViT-M1 encoder 자체가 14 ms라고 보고하므로, 14 ms는 encoder-only 경로, 약 26 ms는 EdgeSAM 전체 benchmark로 구분해야 한다.

GFLOPs는 원본 대비 약 124배 감소했지만 2080 Ti 속도는 약 38배 증가했다. 반대로 MobileSAM 대비 GFLOPs 차이는 1.73배인데 iPhone 속도는 약 7.9배다. 이 비선형성은 operator/backend 적합성이 FLOPs보다 중요하다는 논문의 핵심 근거다.

RPN은 파라미터와 GFLOPs를 각각 0.2M, 0.1만 늘리지만 iPhone FPS는 38.7에서 34.1로, 2080 Ti는 164.3에서 123.9로 떨어진다. proposal decoding, selection, memory access 같은 FLOPs에 잘 드러나지 않는 작업의 비용을 보여준다.

## 11. 정확도 결과

아래는 논문이 보고한 수치다.

### 11.1 Ground-truth box prompts

| 모델 | SA-1K Box | COCO Box | COCO +1 pt | COCO +2 pt | LVIS Box |
|---|---:|---:|---:|---:|---:|
| SAM | 86.7 | 77.3 | 77.7 | 78.1 | 77.8 |
| MobileSAM | 82.0 | 74.4 | 74.8 | 75.1 | 73.1 |
| EfficientSAM-Ti | - | 75.2 | 76.0 | 76.6 | 74.6 |
| EdgeSAM | 83.0 | 76.7 | 78.1 | 79.0 | 76.2 |

EdgeSAM은 첫 box에서 원본 SAM보다 조금 낮지만 COCO에 교정점을 하나 또는 둘 추가하면 78.1, 79.0으로 원본의 77.7, 78.1을 넘는다. distillation loop가 반복 refinement behavior를 직접 학습한 효과와 일치한다.

LVIS에서도 EdgeSAM은 box 76.2로 MobileSAM 73.1, EfficientSAM-Ti 74.6보다 높고, 원본 SAM 77.8에 가깝다.

### 11.2 Center-point prompts

| 모델 | SA-1K Center | SA-1K +2 pt | COCO Center | COCO +2 pt | LVIS Center |
|---|---:|---:|---:|---:|---:|
| SAM | 76.5 | 85.1 | 53.6 | 71.7 | 60.5 |
| MobileSAM | 64.6 | 76.2 | 50.9 | 66.8 | 52.1 |
| EfficientSAM-Ti | - | - | 49.8 | 65.7 | 56.4 |
| EdgeSAM | 67.5 | 79.0 | 48.0 | 68.7 | 53.7 |
| EdgeSAM-RPN | - | - | 54.3 | - | - |

point-only에서는 EdgeSAM이 MobileSAM보다 refinement 후 좋지만 COCO 첫 점에서는 48.0으로 더 낮다. 이는 prompt-aware distillation이 teacher의 SA-1B granularity behavior까지 더 충실히 모방한 부작용으로 해석된다. COCO 전용 RPN이 이를 54.3으로 보정한다.

### 11.3 External detector boxes, COCO

ViTDet-H box prompt를 쓸 때:

| 모델 | mask AP | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|
| SAM | 46.1 | 33.6 | 51.9 | 57.7 |
| FastSAM | 37.9 | 23.9 | 43.4 | 50.0 |
| MobileSAM | 39.4 | 26.9 | 44.4 | 52.2 |
| EfficientSAM-Ti | 42.3 | 26.7 | 46.2 | 57.4 |
| EdgeSAM | 42.2 | 29.6 | 47.6 | 53.9 |

EdgeSAM은 MobileSAM보다 명확히 높고 EfficientSAM-Ti와 전체 AP가 비슷하지만 원본 SAM과는 3.9 AP 차이가 난다. 저자들은 capacity 한계뿐 아니라 학습에는 ground-truth box를 쓰고 추론에는 detector의 부정확한 box를 쓰는 mismatch도 원인일 수 있다고 지적한다.

Detic box를 쓸 때 EdgeSAM은 AP 35.2, boundary IoU 22.5이고 MobileSAM은 AP 33.1, boundary IoU 20.2다. 이 결과는 경계 품질 개선도 보여주지만 detector가 달라지면 결과도 달라지므로 box source를 반드시 함께 보고해야 한다.

## 12. 핵심 ablation 해석

### 12.1 Prompt-in-the-loop KD

SA-1K에서 encoder-only 대비 prompt-KD는 다음처럼 개선된다.

| 설정 | Box | Box +2 pt | Center | Center +2 pt |
|---|---:|---:|---:|---:|
| encoder-only KD | 82.0 | 82.8 | 64.6 | 76.2 |
| prompt-in-the-loop KD | 83.0 | 84.1 | 67.5 | 79.0 |

초기 box보다 교정점 이후의 개선폭이 더 크다. 즉, 단순 마스크 imitation뿐 아니라 correction interaction을 학습한 것이 핵심이다.

### 12.2 Backbone 일반성

prompt-KD는 inference cost를 바꾸지 않으면서 세 backbone 모두 개선했다.

- TinyViT-5M: box/center `81.6/63.7 -> 83.5/68.0`
- EfficientViT-B1: `80.7/60.9 -> 82.7/65.8`
- RepViT-M1: `82.0/64.6 -> 83.0/67.5`

따라서 기여는 RepViT 자체가 아니라 prompt selection을 포함한 distillation 방법이다. RepViT는 모바일 backend에서 가장 빠른 구현 선택이다.

### 12.3 Decoder target

| distillation target | SA-1K Box | Center | COCO AP | Boundary IoU |
|---|---:|---:|---:|---:|
| mask only | 83.1 | 67.4 | 35.1 | 22.4 |
| mask + attention | 82.8 | 67.0 | 35.0 | 22.3 |
| mask + IoU | 82.5 | 67.1 | 34.9 | 22.2 |
| refined image feature | 71.6 | 55.0 | 30.2 | 17.2 |

복잡한 중간 정렬이 더 좋지 않았다. teacher가 최종적으로 수행해야 할 task output을 직접 제공하고, 올바른 prompt 조합을 만드는 것이 더 중요했다.

### 12.4 학습 순서와 loop 수

- encoder KD와 prompt KD를 joint train: box/center `81.2/63.2`
- 두 단계 순차 학습: `83.1/67.4`
- correction loop 0: center +2 pt 72.4
- loop 1: 79.0
- loop 2: 79.2

두 번째 loop의 추가 이득이 0.2에 불과하므로 기본값 1이 타당하다. stage별 최적화 조건이 달라 joint training보다 sequential training이 낫다.

### 12.5 데이터 크기

ViTDet-H box 기준 COCO AP는 1% SA-1B에서 42.2, 3%에서 42.7, 10%에서 43.0으로 증가한다. 더 많은 데이터가 도움은 되지만 10배 데이터에 +0.8 AP라 수익이 빠르게 감소한다.

비슷한 약 110k 이미지 수로 SA-1B, COCO, LVIS를 비교하면 box prompt는 SA-1B 학습이 세 test set에서 가장 잘 전이된다. center point는 train/test domain이 같을 때 유리해 granularity prior 의존성을 다시 확인한다.

## 13. 온디바이스 배포 관점

### 13.1 이 논문이 실제로 보여준 것

- 실제 iPhone 14에서 Core ML/Xcode benchmark 수행
- 1024 입력에서 EdgeSAM 38.7 FPS
- RepViT-M1 encoder-only 14 ms 보고
- MobileSAM/EfficientSAM-Ti와 동일 장치 비교
- CNN이 ANE에 더 잘 mapping된다는 실측 근거 제공

이는 GFLOPs와 desktop GPU latency만 제시하는 경량화 논문보다 훨씬 강한 시스템 증거다.

### 13.2 여전히 미보고인 것

- iPhone 14 peak resident memory와 ANE memory
- p50/p95 또는 반복 횟수별 latency 분포
- cold model load와 첫 inference 시간
- 입력 전처리, 좌표 변환, mask upsample/overlay 포함 end-to-end 앱 latency
- ANE/GPU/CPU별 operator 배치와 fallback 목록
- 전력, 온도, 장시간 throttling
- FP16, INT8, mixed precision 비교
- prompt 수에 따른 cached decoder latency

논문 표의 FPS는 중요한 실제 측정값이지만, 사용자 경험 전체를 의미하지는 않는다. 특히 demo device는 본문에서 iPhone 13/iOS 17.1로 설명되고 정량 표는 iPhone 14이므로 두 결과를 혼합하지 않아야 한다.

### 13.3 권장 실행 구조

```text
new image or selected video frame
  -> EdgeSAM CNN encoder once
  -> cache [1,256,64,64]

each click / box
  -> optional RPN only when domain prior is desired
  -> frozen prompt encoder
  -> SAM-compatible lightweight decoder
  -> low-res mask, then selective upsample
```

RPN은 단일 COCO식 point intent가 중요할 때만 켜고, box prompt나 범용 도메인에서는 끄는 정책이 좋다. detector가 이미 box를 주는 파이프라인에서는 RPN을 중복 실행할 이유가 거의 없다.

### 13.4 Everything mode는 피해야 한다

SAM식 `32 x 32` point grid는 decoder를 1,024번 호출한다. EdgeSAM 논문도 속도 민감 장치에서 everything mode를 권하지 않는다. CNN encoder가 26 ms 수준이어도 decoder 1,024회와 NMS가 붙으면 real-time 장점이 사라진다.

온디바이스에서는 다음 대안이 낫다.

- 외부 경량 detector의 box만 prompt로 사용
- 사용자가 선택한 ROI 내부에서만 소수 point grid 사용
- coarse proposal 뒤 uncertainty가 높은 영역만 추가 prompt
- 여러 point를 backend-friendly batch로 묶기

## 14. 실패 사례와 한계

### 14.1 원본 SAM과의 정확도 차이

box prompt에서는 가깝지만 external detector box의 COCO AP는 원본보다 낮다. 9.6M 모델이 641M teacher의 모든 zero-shot behavior를 보존한 것은 아니다.

### 14.2 Hole과 세부 경계

정성 결과에서 도넛 내부 hole을 잘못 채우는 사례가 제시된다. 저자들은 학습 데이터 부족 가능성을 언급하고 hole 영역에 negative point를 추가하면 보정할 수 있다고 설명한다. 이는 자동 실행보다 interactive correction에 더 적합한 모델임을 뜻한다.

### 14.3 Teacher 오류의 전이

teacher mask를 threshold해 pseudo target으로 쓰므로 teacher가 part를 선택하거나 경계를 틀리면 student도 이를 학습한다. 교정점은 teacher와 student의 불일치만 찾으며 teacher와 실제 정답의 불일치는 직접 찾지 않는다.

### 14.4 RPN의 도메인 종속성

COCO용 granularity prior는 COCO metric을 높이지만, 다른 도메인의 유효한 part-level behavior를 억제할 수 있다. RPN은 범용성 개선 모듈이 아니라 특정 application convention을 주입하는 adapter로 이해해야 한다.

### 14.5 양자화와 장기 실행 미검증

논문은 quantization, pruning, mixed precision, 추가 on-device optimization을 미래 과제로 남긴다. 38.7 FPS가 thermal steady state에서도 유지되는지는 알 수 없다.

### 14.6 데이터와 augmentation

기본 학습은 SA-1B의 1%만 사용하고 augmentation을 적용하지 않았다. 효율적이지만 극단 조명, motion blur, 카메라 noise에 대한 모바일 robustness는 별도 평가가 필요하다.

## 15. 강점

1. **과제 지향 distillation**: feature를 닮게 하는 데서 끝나지 않고 prompt와 mask interaction을 직접 증류한다.
2. **동적 hard example 생성**: student가 틀린 위치가 다음 학습 prompt가 되어 별도 mining pipeline이 필요 없다.
3. **SAM decoder 호환성**: 원본의 prompt interface와 decoder weight를 재사용한다.
4. **모바일 실측**: iPhone 14 Core ML 결과로 CNN/ViT의 backend 차이를 입증한다.
5. **데이터 효율**: 1% SA-1B와 짧은 2-stage training으로 경쟁 모델과 비슷하거나 나은 성능을 낸다.
6. **광범위한 ablation**: backbone, loss target, freezing, LoRA, loop 수, prompt 수, 데이터 크기와 source를 분리해 검증한다.

## 16. 재현 체크리스트

### 구조

- [ ] RepViT-M1의 정확한 variant와 stage config를 고정했는가?
- [ ] 원래 downsampling을 유지하고 마지막 두 stage를 tiny FPN으로 합쳤는가?
- [ ] 최종 embedding이 정확히 `B x 256 x 64 x 64`인가?
- [ ] SAM prompt encoder와 two-way decoder checkpoint를 올바르게 상속했는가?
- [ ] mask token 4개와 IoU token의 순서를 유지했는가?

### Stage 1

- [ ] teacher/student feature의 resize와 channel projection이 동일한가?
- [ ] 1% SA-1B split과 SA-1K test 1,000 images가 겹치지 않는가?
- [ ] teacher embedding cache dtype과 disk 크기를 기록했는가?
- [ ] MSE reduction 방식과 normalization을 기록했는가?

### Stage 2

- [ ] 초기 box/point를 동일 확률로 선택하는가?
- [ ] teacher highest-IoU mask와 대응 student mask를 비교하는가?
- [ ] false negative에서는 positive, false positive에서는 negative point를 뽑는가?
- [ ] correction loop 기본값이 1인가?
- [ ] teacher mask threshold와 Dice/BCE 가중치를 기록했는가?
- [ ] 이미지당 최대 16 instances를 적용했는가?
- [ ] student encoder와 decoder 모두 optimizer에 포함했는가?

### RPN

- [ ] backbone과 나머지 module을 freeze했는가?
- [ ] class-agnostic boxes, focal/Huber loss를 사용했는가?
- [ ] K-nearest K, proposal threshold, merge와 NMS 규칙을 config에 명시했는가?
- [ ] RPN off 결과도 함께 평가했는가?

### 평가

- [ ] GT box, center point, external detector box를 구분했는가?
- [ ] detector 이름과 box AP를 함께 기록했는가?
- [ ] mIoU, AP, boundary IoU를 혼동하지 않았는가?
- [ ] 첫 prompt와 +1/+2 refinement point를 모두 측정했는가?
- [ ] iPhone 모델, OS, Core ML compute unit, precision을 기록했는가?
- [ ] p50/p95, peak memory, power, temperature를 추가 측정했는가?

## 17. 로드맵에서의 의미

SAM이 `heavy encoder once + cheap prompts many times`라는 구조를 제안했다면, EdgeSAM은 첫 번째 항을 실제 모바일 CNN으로 바꾸는 방법을 보여준다. 특히 다음 교훈이 중요하다.

```text
feature-only KD
  -> prompt behavior가 충분히 보존되지 않음

prompt-in-the-loop KD
  -> teacher/student disagreement가 다음 supervision을 생성
  -> box와 point 조합, iterative correction behavior 보존

on-device deployment
  -> FLOPs보다 CNN operator support와 compiler mapping이 중요
```

통합 프로젝트에서는 detector가 매 프레임 실행되고 EdgeSAM은 이벤트 또는 사용자 요청에만 실행되는 구성이 적합하다. detector box를 prompt로 넘기면 single-point granularity 문제와 RPN 비용을 동시에 피할 수 있다. 생성된 ROI mask로 VLM visual token을 줄이는 실험까지 연결할 수 있다.

SAM 2 이후에는 이 아이디어가 video memory까지 포함한 EdgeTAM류로 확장된다. 이미지 encoder만 줄여도 memory attention과 memory bank가 새 병목이 되므로, EdgeSAM은 이미지 기반 경량화의 출발점이지 영상 문제의 완결된 해답은 아니다.

## 18. 최종 정리

- EdgeSAM의 핵심 기여는 RepViT 교체 자체보다 **prompt를 학습 loop 안에 넣은 distillation**이다.
- teacher와 student가 다른 영역에서 교정점을 만들고, 한 번의 추가 loop로 iterative refinement를 효과적으로 전달한다.
- 기본 구조는 9.6M parameters, 22.1 GFLOPs이며 원본 SAM의 `256 x 64 x 64` decoder interface를 유지한다.
- iPhone 14에서 38.7 FPS, 약 25.84 ms를 보고하며 MobileSAM의 4.9 FPS보다 약 7.9배 빠르다.
- 모바일 속도 차이는 FLOPs만이 아니라 CNN이 ANE에 잘 mapping되는 데서 크게 나온다.
- point granularity를 위한 RPN은 COCO mIoU를 48.0에서 54.3으로 높이지만 domain-specific adapter이며 필요할 때만 켜야 한다.
- everything mode의 1,024 decoder calls는 여전히 비현실적이므로 detector box, ROI, 사용자 prompt 기반 요청형 실행이 적합하다.
- peak memory, p95, 전력, 온도, 양자화 결과는 논문 미보고이므로 실제 장치 연구에서 반드시 보완해야 한다.

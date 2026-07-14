# 74. Knowledge Distillation

## 논문 정보

- 원본 파일: `74_Knowledge_Distillation.pdf`
- 제목: **Distilling the Knowledge in a Neural Network**
- 저자: Geoffrey Hinton, Oriol Vinyals, Jeff Dean
- 공개: arXiv:1503.02531, v1, 2015-03-09
- 공식 링크: [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- 태스크: model compression, ensemble compression, classification, speech recognition, large-scale specialist ensemble
- 핵심 키워드: soft targets, temperature, dark knowledge, teacher-student, transfer set, specialist model, regularization

## 한눈에 보는 요약

이 논문은 큰 model이나 ensemble이 학습한 함수를 작은 단일 model로 옮기는 방법을 정립한 고전이다. 핵심 관찰은 one-hot label이 정답 class 하나만 알려 주는 반면, teacher의 전체 class probability에는 class 사이의 유사성이 들어 있다는 것이다.

```text
hard target:  [0, 0, 1, 0, 0]

teacher soft target at T > 1:
              [0.03, 0.12, 0.70, 0.10, 0.05]
                    ^       ^
        which wrong classes look plausible
```

Student는 두 목표를 함께 학습한다.

```text
teacher logits --softmax(T)-- soft targets --+
                                             +-> student update
ground-truth label ----------- hard target ---+
```

Temperature `T`를 높이면 teacher distribution이 부드러워져 작은 wrong-class probability의 상대 관계가 드러난다. Student도 같은 `T`에서 soft target을 맞추고, hard label은 `T=1`에서 학습한다. Soft loss gradient가 대략 `1/T^2`로 작아지므로 논문은 soft term에 `T^2`를 곱하는 이유를 유도한다.

MNIST에서 2개 hidden layer, layer당 1,200 unit인 teacher는 67 errors이고, regularization 없는 800-unit student는 146 errors다. Distillation한 800-unit student는 74 errors까지 회복한다. Speech recognition에서는 10-model ensemble의 WER `10.7%`를 같은 크기의 distilled single model도 `10.7%`로 재현한다.

이 리뷰는 2015년 논문의 원 기여와 이후의 현대적 사용을 분리한다. Feature-map matching, attention transfer, self-distillation, detector box loss distillation, LLM token distillation은 이 논문이 직접 제안하거나 실험한 방법이 아니다.

## 논문의 문제 설정

### Knowledge는 parameter가 아니라 function에 있다

논문은 model의 지식을 parameter 값 자체보다 입력에서 출력으로 가는 learned mapping으로 본다. 같은 함수를 표현하는 architecture와 parameterization은 여러 가지일 수 있기 때문이다.

```math
f_T:x\mapsto p_T(y\mid x)
```

Teacher가 큰 ensemble이어도 배포에 필요한 것은 ensemble 구조 자체가 아니라 이 mapping의 일반화 특성이다. Student `f_S`가 같은 mapping을 근사하면 훨씬 작은 추론 graph로 배포할 수 있다.

### Hard label이 버리는 정보

Class가 `car`인 이미지에 one-hot label은 다음만 말한다.

```text
car = correct
all others = wrong
```

Teacher distribution은 `truck`이 `carrot`보다 더 그럴듯한 오답이라는 정보를 줄 수 있다. 논문은 이런 wrong-class probability의 상대 구조를 지식의 중요한 부분으로 본다. 후대 문헌에서 흔히 `dark knowledge`라고 부르는 내용이다.

이 정보는 sample 하나당 여러 class 관계를 제공하므로 gradient variance를 줄이고 더 적은 데이터에서도 일반화 regularizer처럼 작동할 수 있다.

## Temperature softmax

Teacher 또는 student의 class `i` logit을 `z_i`라 하자. Temperature `T`의 softmax는 다음과 같다.

```math
q_i(T)
=
\frac{\exp(z_i/T)}
{\sum_j\exp(z_j/T)}
```

- `T=1`: 일반 softmax
- `T>1`: distribution이 부드러워짐
- 매우 큰 `T`: class probability가 uniform에 가까워짐
- 작은 `T`: top class에 mass가 집중됨

Logit 차이가 `z_a-z_b`일 때 probability ratio는 다음과 같다.

```math
\frac{q_a}{q_b}
=
\exp\left(\frac{z_a-z_b}{T}\right)
```

따라서 `T`가 클수록 같은 logit 차이가 완만하게 표현된다. 예를 들어 두 logit 차이가 4이면 ratio는 `T=1`에서 `e^4`, `T=4`에서 `e^1`이다. Top class에 가려진 나머지 class 관계를 loss가 읽을 수 있게 된다.

### 너무 큰 temperature의 위험

무조건 큰 `T`가 좋은 것은 아니다. 모든 class가 거의 같은 확률이 되면 meaningful ranking과 noise를 구분하기 어렵다. 논문의 MNIST 30-unit student에서는 `T=2.5`에서 `4`가 더 좋았고, 300 unit 이상에서는 `T>8`의 여러 값이 비슷했다. 최적 temperature는 student capacity와 class 구조에 따라 달라진다.

## Distillation objective

Teacher logit `v`, student logit `z`, hard label `y`를 두자. 논문의 설명을 현대적인 notation으로 정리하면 다음 혼합 objective다.

```math
\mathcal L
=
\lambda_{soft}T^2
H\left(p_T^{(T)},p_S^{(T)}\right)
+
\lambda_{hard}
H\left(y,p_S^{(1)}\right)
```

여기서:

- `p_T^(T)`: teacher logits에 temperature `T`를 적용한 target
- `p_S^(T)`: student logits에 같은 `T`를 적용한 prediction
- `p_S^(1)`: 실제 추론 temperature의 student prediction
- `H`: cross-entropy

논문은 hard-target cross-entropy에 considerably lower weight를 두는 방식을 설명한다. 오늘날 많이 쓰는 `alpha` 형태의 정확한 mixing convention은 구현마다 다르다. 원문 실험의 상대 weight와 현대 library parameter를 그대로 동일시하면 안 된다.

### 왜 hard target도 섞는가

Teacher가 항상 맞는 것은 아니다. Student가 soft target을 정확히 맞추지 못할 때 ground-truth 방향으로 오차를 허용하는 것이 도움이 될 수 있다. Hard loss는 teacher bias에 대한 안전장치이자 실제 task label을 직접 최적화하는 신호다.

Transfer set이 unlabeled라면 soft loss만 쓸 수 있고, labeled training set이면 두 loss를 함께 쓸 수 있다. 논문은 original training set을 transfer set으로 쓰는 것도 잘 작동한다고 보고한다.

## Gradient와 T 제곱 보정

Teacher probability를 `p_i`, student probability를 `q_i`라 하고, 같은 temperature에서 cross-entropy를 쓴다. Student logit에 대한 gradient는:

```math
\frac{\partial C}{\partial z_i}
=
\frac{1}{T}(q_i-p_i)
```

Temperature가 커지면 probability difference 자체도 대략 `1/T`로 줄어든다. 따라서 전체 gradient scale은 대략 `1/T^2`가 된다.

논문은 logits의 평균을 0으로 두고 high-temperature approximation을 전개한다. Class 수를 `N`, teacher logit을 `v_i`, student logit을 `z_i`라 하면:

```math
\frac{\partial C}{\partial z_i}
\approx
\frac{1}{NT^2}(z_i-v_i)
```

따라서 soft-target loss의 contribution을 temperature가 바뀌어도 비슷하게 유지하려면 `T^2`를 곱한다.

```text
without correction: T grows -> soft gradient rapidly shrinks
with T^2 correction: relative soft-loss influence stays comparable
```

### Logit matching이 special case인 이유

위 근사에서 gradient는 student와 teacher logit의 제곱오차 gradient와 같은 형태다. 즉 매우 높은 temperature, zero-mean logits 조건에서는 probability matching이 logit matching에 접근한다. 논문은 기존 logit regression을 distillation의 특수한 경우로 설명한다.

하지만 실제 구현에서 무한히 높은 `T`를 쓰거나 mean centering을 생략하고 MSE만 적용하면 원문의 조건과 다르다. Softmax는 additive logit shift에 불변이지만 raw logit MSE는 그렇지 않다는 차이도 있다.

## 전체 학습 절차

```text
1. Train one large regularized model or an ensemble.
2. Choose a transfer set, labeled or unlabeled.
3. Run teacher once and store logits or soft targets.
4. Train student to match teacher at temperature T.
5. If labels exist, add hard-target loss at T=1.
6. Deploy student alone at T=1.
```

Teacher가 ensemble이면 soft target은 member prediction의 arithmetic 또는 geometric average로 만들 수 있다. 어떤 공간에서 평균하는지에 따라 결과가 다르므로 probability 평균인지 logit 평균인지 기록해야 한다.

### Offline logit cache의 시스템 이점

Teacher를 student training 매 step마다 실행할 필요는 없다. Transfer set이 고정이면 logits를 미리 저장한다.

```text
teacher inference stage:
  x -> teacher -> logits -> disk cache

student training stage:
  x + cached logits + optional labels -> student update
```

Cache 저장량은 sample 수 `M`, class 수 `C`, precision에 비례한다.

```math
\text{bytes}=M\times C\times\text{bytes per logit}
```

예를 들어 1M samples, 1,000 classes, FP16이면 약 `1.86 GiB`다. Detection처럼 region/token 차원이 더 붙으면 full logit cache가 매우 커져 top-k 또는 on-the-fly teacher가 필요할 수 있다. 이 계산은 현대적 적용을 위한 리뷰어 분석이며 원 논문의 실험 저장 형식을 뜻하지 않는다.

## MNIST 실험

### 설정

Teacher는 hidden layer 2개, 각 `1,200` rectified linear units를 사용하고 dropout과 입력 translation jitter로 regularize한다. Student baseline은 hidden layer 2개, 각 `800` units이며 regularization을 쓰지 않는다.

| 모델 | 구조/학습 | test errors |
|---|---|---:|
| Teacher | 2 x 1,200, dropout + jitter | 67 |
| Small baseline | 2 x 800, no regularization | 146 |
| Distilled small | 2 x 800, T=20 | 74 |

Student가 architecture를 줄였는데도 teacher에 가까운 error를 얻는다. 특히 transfer set에는 translated image가 없어도 teacher soft target이 augmentation에서 배운 invariance를 전달한다는 해석을 제시한다.

### 극단적으로 작은 student

Hidden layer당 30 unit으로 줄인 경우 optimal temperature가 `2.5`에서 `4` 범위였다. Student capacity가 작으면 매우 미세한 teacher relation을 모두 맞추려는 것이 오히려 방해가 될 수 있다. Capacity mismatch를 temperature로 조절해야 함을 보여 준다.

### 한 class를 transfer set에서 제거

Transfer set에서 digit `3` example을 모두 제거한다. Student 관점에서는 학습 중 실제 `3`을 본 적이 없다.

- 초기 결과: 전체 206 errors
- 그중 test set의 1,010개 `3`에서 133 errors
- Class 3 bias를 `+3.5` 조정: 전체 109 errors
- `3`에서 14 errors, 즉 `98.6%` correct

Teacher가 다른 digit example에 부여한 class-3 probability만으로도 `3`에 대한 상당한 지식이 전달된다. 다만 bias를 test performance에 맞춰 조정한 결과이므로 완전한 blind generalization으로 과장하면 안 된다.

논문은 `7`과 `8`만 transfer set에 남긴 더 극단적인 실험도 설명한다. 그대로는 test error `47.3%`지만 해당 bias를 조정하면 `13.2%`까지 낮아진다. Soft target이 보지 않은 class structure를 일부 담지만 class prior calibration은 별도 문제라는 뜻이다.

## Speech recognition 실험

### Acoustic model

- Hidden layers: 8
- Units per layer: 2,560 ReLU
- Output labels: 14,000 HMM states
- Input: 26 frames x 40 Mel filter-bank coefficients
- Parameters: 약 85M
- Training data: 약 2,000 hours, 700M examples
- Baseline frame accuracy: 58.9%
- Baseline WER: 10.9%

10개 model은 architecture와 procedure가 같고 random initialization만 다르다. 별도 data subset으로 diversity를 늘리는 시도는 결과를 유의미하게 바꾸지 않아 단순한 independent initialization을 사용한다.

Distillation temperature 후보는 `[1,2,5,10]`이고, hard-target cross-entropy relative weight는 `0.5`다.

### 결과

| System | test frame accuracy | WER |
|---|---:|---:|
| Baseline | 58.9% | 10.9% |
| 10-model ensemble | 61.1% | 10.7% |
| Distilled single model | 60.8% | 10.7% |

Frame accuracy에서 ensemble 개선의 80% 이상을 single model이 전달받고, 최종 WER은 ensemble과 동일하다. 중요한 점은 student가 baseline보다 더 작은 architecture가 아니라 **같은 크기의 single model**이라는 것이다. 이 실험의 목적은 크기 축소보다 ensemble inference cost 제거다.

WER 개선이 frame accuracy 개선보다 작은 것은 학습 objective와 최종 sequence decoding metric의 mismatch 때문이다. 이는 현대 detector에서 class-logit KD만으로 mAP가 충분히 오르지 않을 수 있는 이유와도 연결된다.

## Specialist ensemble

### 배경: JFT 100M/15K

JFT는 100M labeled images와 15,000 labels를 가진 내부 dataset이다. Generalist model 하나를 약 6개월 학습했다고 논문은 설명한다. 같은 full model 여러 개로 ensemble을 만들면 정확도는 오르지만 training compute가 과도하다.

### Specialist 구조

각 specialist는 서로 혼동하기 쉬운 class subset에 집중한다. 예를 들어 특정 차종, 다리 종류, 행사 종류가 한 cluster가 된다. 관심 없는 나머지 class는 하나의 `dustbin` class로 합친다.

```text
15,000 classes
  -> generalist covers all
  -> specialist m covers about 300 confusing classes
  -> all other classes -> one dustbin class
```

Specialist는 generalist weight로 initialize한다. Training batch의 절반은 special subset, 절반은 나머지에서 random sampling한다. Special class가 oversampling된 prior bias는 dustbin logit에 oversampling proportion의 log를 더해 보정한다.

### Class grouping

True-label confusion matrix 대신 generalist prediction covariance matrix의 column을 online K-means로 clustering한다. 함께 높은 probability를 받는 class를 같은 specialist에 배정한다.

```text
generalist predictions over data
  -> covariance between class outputs
  -> cluster covariance columns
  -> confusable subsets S_m
```

Label 오류나 class imbalance에 덜 직접적으로 의존하지만, generalist가 한 번도 함께 혼동하지 않은 관계는 cluster에 드러나지 않는다.

### Conditional inference

Test image마다 모든 specialist를 실행하지 않는다.

1. Generalist의 top-n class set `k`를 구한다. 실험은 `n=1`이다.
2. `S_m`과 `k`가 교차하는 specialist만 active set `A_k`에 넣는다.
3. Generalist와 active specialist distribution에 가까운 full distribution `q`를 구한다.

```math
q^*
=
\arg\min_q
\left[
KL(p_g,q)
+\sum_{m\in A_k}KL(p_m,q)
\right]
```

Specialist의 dustbin probability는 `q`에서 해당 specialist가 다루지 않는 모든 class probability를 합한 값과 비교한다. 일반 closed form이 없어서 `q=softmax(z)`로 parameterize하고 image마다 logit `z`를 gradient descent로 최적화한다.

이 per-image optimization은 오늘날 배포 관점에서는 비싸다. 논문도 specialist ensemble 자체를 최종 endpoint로 고집하기보다 지식을 다시 single model로 distill하는 방향을 목표로 한다.

### JFT 결과

61개 specialist를 사용하고 각 specialist는 300 special classes와 dustbin을 가진다.

| System | conditional test accuracy | overall test accuracy |
|---|---:|---:|
| Baseline | 43.1% | 25.0% |
| + 61 specialists | 45.9% | 26.1% |

Overall relative improvement는 `4.4%`다. Correct class를 더 많은 specialist가 cover할수록 대체로 개선율이 커진다. 예를 들어 specialist 1개 coverage에서 relative change `+3.4%`, 9개에서 `+16.6%`다.

중요한 한계가 있다. 논문은 specialist ensemble의 향상을 다시 single large network로 distill하는 실험을 완료하지 못했다고 명시한다. Specialist section을 최종 distillation 성공 결과로 읽으면 안 된다.

## Soft target의 regularization 효과

85M-parameter speech model에 전체 training data의 3%만 사용한 실험이다.

| System/training set | train frame accuracy | test frame accuracy |
|---|---:|---:|
| Baseline, 100% | 63.4% | 58.9% |
| Hard targets, 3% | 67.3% | 44.5% |
| Soft targets, 3% | 65.4% | 57.0% |

Hard target model은 train accuracy가 더 높은데 test는 크게 나빠진다. Soft target model은 3% data만으로 full-data baseline보다 약 2 percentage points 낮은 test accuracy까지 회복하고 early stopping 없이 57%에 수렴한다.

Soft target은 단순 label smoothing과 다르다.

```text
label smoothing:
  every example receives nearly the same non-target mass pattern

distillation:
  each example receives teacher-specific wrong-class structure
```

Teacher가 배운 class similarity와 input-dependent uncertainty가 regularizer 역할을 한다.

## 구현 의사코드

```python
def kd_loss(student_logits, teacher_logits, labels,
            temperature=4.0, hard_weight=0.1):
    T = temperature
    teacher_prob = softmax(teacher_logits / T, dim=-1).detach()
    student_log_prob_T = log_softmax(student_logits / T, dim=-1)

    soft = -(teacher_prob * student_log_prob_T).sum(dim=-1).mean()
    soft = soft * (T * T)

    hard = cross_entropy(student_logits, labels)
    return soft + hard_weight * hard
```

Teacher는 반드시 stop-gradient한다. Loss library가 `KLDivLoss`를 쓰면 reduction과 constant term이 달라질 수 있지만 student gradient는 target cross-entropy와 같은 방향으로 만들 수 있다.

### Logit cache

```python
with no_grad():
    for sample_id, x in transfer_loader:
        logits = teacher_ensemble_logits(x)
        save_fp16(sample_id, logits)

for sample_id, x, y in student_loader:
    teacher_logits = load_fp16(sample_id)
    student_logits = student(x)
    loss = kd_loss(student_logits, teacher_logits, y)
    loss.backward()
```

FP16 cache가 soft target ranking을 충분히 보존하는지 일부 sample에서 FP32와 비교해야 한다.

## 원 논문의 기여와 현대적 해석 분리

### 원 논문이 직접 제안하고 검증한 것

- Temperature softmax를 이용한 output probability distillation
- Soft loss와 hard label loss의 결합
- `T^2` gradient scaling의 근거
- Logit matching을 high-temperature special case로 해석
- MNIST와 speech ensemble compression
- Soft targets의 data-efficient regularization 효과
- JFT generalist + confusable-class specialists 구성과 conditional ensemble

### 원 논문이 직접 다루지 않은 것

- Intermediate feature-map matching
- Attention map transfer
- Relational knowledge distillation
- Self-distillation 또는 born-again network
- Teacher 없는 label smoothing
- Detection box regression, NMS, mask, VLM token별 distillation
- LLM sequence-level preference distillation
- Quantization-aware distillation

아래의 온디바이스 detection/VLM 적용은 논문 원문 결과가 아니라 이 원리를 현대 task에 확장한 리뷰어 제안이다.

## 온디바이스 vision 적용

### Detector distillation 설계

Teacher가 Grounding DINO, SAM, Florence-2이고 student가 YOLOE/SegFormer라면 output space가 일치하지 않는다. Class soft target 외에 assignment가 필요하다.

```text
teacher boxes/masks/text scores
  -> geometric matching with student candidates
  -> class soft logits KD
  -> box distribution or IoU target
  -> mask probability target
```

하지만 이것은 2015 논문의 직접 방법이 아니다. Temperature KD를 class alignment component로 사용하고 detection-specific loss는 별도 명시해야 한다.

Open-vocabulary에서는 teacher와 student vocabulary embedding이 다를 수 있다. 동일 prompt set을 두 모델에 주고 `[region,prompt]` logit을 맞추거나, teacher top-k phrase만 transfer set target으로 저장할 수 있다.

### VLM distillation

Teacher token distribution `[L,V]` 전체를 저장하면 vocabulary `V`가 커서 cache가 크다. Top-k logit과 residual mass를 저장하는 방법을 고려할 수 있다.

```text
full logits: M x L x V
top-k cache: M x L x (indices + k logits + residual)
```

Location token과 natural-language token은 entropy가 다르므로 temperature와 loss weight를 분리하는 것이 합리적이다. 이 역시 현대적 확장이다.

### 배포 이득 계산

Teacher ensemble `K`개를 매 요청 실행하던 시스템을 student 하나로 바꾸면 이상적 compute는 대략 `1/K`에 가까워질 수 있지만 student architecture, batching, memory bandwidth에 따라 실제 speedup은 다르다.

Weight memory는 parameter 비에 직접 비례한다.

```math
\text{FP16 weight MiB}
=
\frac{P\times2}{2^{20}}
```

예를 들어 500M teacher에서 50M student로 옮기면 FP16 raw weight는 약 953.7 MiB에서 95.4 MiB로 줄어든다. 그러나 distillation은 student의 activation memory나 unsupported operator를 자동으로 해결하지 않는다. 실제 latency는 target device에서 측정해야 한다.

## 강점

1. Model compression을 parameter 복제가 아니라 function transfer로 명확히 재정의했다.
2. Wrong-class probability가 전달하는 class structure를 직관과 수식으로 설명했다.
3. Temperature와 gradient scale의 관계를 유도했다.
4. 작은 MNIST뿐 아니라 85M speech model과 10-model ensemble에서 검증했다.
5. Soft target이 강한 regularizer가 될 수 있음을 3% data 실험으로 보였다.
6. 거대 class space에서 specialist를 조건부 실행하는 아이디어를 제시했다.

## 한계

### 비교 범위

MNIST와 speech 실험은 강력하지만 현대 detection, segmentation, generative model의 structured output을 다루지 않는다. 오늘날 모든 KD variant의 효능을 이 논문 결과로 보증할 수 없다.

### Teacher 오류 전달

Soft target은 teacher의 useful similarity뿐 아니라 bias와 calibration error도 전달한다. Hard loss가 이를 일부 완화하지만 unlabeled transfer set에서는 ground-truth 안전장치가 없다.

### Temperature와 weight tuning

최적 `T`와 loss weight가 student capacity와 task에 민감하다. 논문은 일반적인 자동 선택 규칙을 제공하지 않는다.

### Ensemble 비용은 training에 남는다

Deployment는 student 하나로 줄지만 teacher ensemble을 학습하고 transfer set logits를 생성하는 비용은 필요하다. Training energy와 storage까지 포함하면 압축의 총비용은 무료가 아니다.

### Specialist distillation 미완료

JFT specialist ensemble의 정확도 개선은 보여 주지만 그 지식을 single model로 distill하는 최종 단계는 보여 주지 못했다.

## 재현 체크리스트

### Teacher와 target

- [ ] Teacher checkpoint와 ensemble averaging 방식을 기록한다.
- [ ] Probability 평균인지 logit 평균인지 명시한다.
- [ ] Teacher는 eval mode, student loss에서 detach한다.
- [ ] Transfer set이 labeled인지 unlabeled인지 구분한다.
- [ ] Teacher가 transfer set augmentation을 어떻게 보는지 고정한다.

### Loss

- [ ] Teacher/student에 동일 temperature를 적용한다.
- [ ] Soft term의 `T^2` scaling 여부를 기록한다.
- [ ] Hard loss relative weight의 convention을 명시한다.
- [ ] KL reduction이 batchmean인지 mean인지 확인한다.
- [ ] `T=1`, 여러 `T`, hard-only, soft-only ablation을 수행한다.

### 평가

- [ ] Teacher, ensemble, baseline student, distilled student를 모두 비교한다.
- [ ] Parameter, MACs, model size, latency, peak memory를 함께 보고한다.
- [ ] Calibration과 per-class accuracy를 확인한다.
- [ ] Teacher가 틀린 sample에서 student behavior를 분석한다.
- [ ] 동일 target device에서 batch=1 p50/p95를 측정한다.

## 추천 ablation

| 축 | 설정 |
|---|---|
| Temperature | 1, 2, 4, 8, 16 |
| Hard weight | 0, 0.1, 0.5, 1.0 |
| Transfer data | labeled train, unlabeled in-domain, out-of-domain |
| Teacher | single large, homogeneous ensemble, heterogeneous ensemble |
| Student size | 1/2, 1/4, 1/10 parameters |
| Cache | FP32, FP16, top-k logits |

성능뿐 아니라 teacher inference cost와 cache size를 포함해 total training cost를 비교해야 한다.

## 최종 평가

이 논문의 오래가는 핵심은 `큰 모델의 정답을 복사한다`가 아니라 `큰 모델이 입력마다 형성한 전체 확률 구조를 작은 모델이 배우게 한다`는 관점이다. Temperature는 top class에 가려진 구조를 드러내고, hard target은 실제 label과 teacher prediction 사이의 균형을 잡으며, `T^2` 보정은 그 soft signal의 gradient scale을 유지한다.

온디바이스 연구에서는 teacher를 배포하지 않아도 된다는 점이 가장 중요하다. Grounding DINO, SAM, Florence-2 같은 강한 model로 offline pseudo target을 만들고, YOLOE나 경량 segmenter가 이를 흡수하도록 하면 runtime graph는 student 하나로 남는다. 다만 structured vision task에 추가하는 box, mask, feature loss는 이 논문의 원 기여가 아니라 후속 설계임을 명확히 해야 한다.

좋은 distillation 실험은 teacher accuracy만 좇지 않는다. Student가 실제 장치에서 얼마나 작고 빠른지, peak activation과 p95가 줄었는지, teacher bias와 rare class 성능이 어떻게 변했는지를 함께 측정해야 한다.

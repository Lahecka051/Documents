# 55. Segment Anything

## 논문 정보

- 제목: **Segment Anything**
- 저자: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick
- 소속: Meta AI Research, FAIR
- 공개: arXiv 2023
- arXiv: [2304.02643](https://arxiv.org/abs/2304.02643)
- 원본 파일: `55_Segment_Anything_SAM.pdf`
- 핵심 산출물: promptable segmentation task, Segment Anything Model(SAM), SA-1B 데이터셋

## 1. 한 문장 요약

SAM은 무거운 이미지 인코더의 출력을 한 번 캐시한 뒤 점, 박스, 마스크 같은 다양한 프롬프트를 경량 양방향 Transformer 디코더로 처리하고, 모호한 프롬프트에는 여러 개의 유효한 마스크를 내도록 학습한 범용 프롬프트 기반 분할 모델이다.

## 2. 이 논문이 해결하려는 문제

기존 분할 모델은 대개 semantic, instance, interactive segmentation처럼 정해진 과제와 레이블 체계에 맞춰 학습된다. 새로운 객체 종류, 새로운 영상 도메인, 새로운 사용자 상호작용으로 옮기려면 다시 학습하거나 별도 모델을 만들어야 한다. 저자들은 자연어 모델의 prompting과 유사한 인터페이스를 분할에 도입하고자 한다.

논문이 묶어서 푸는 문제는 세 가지다.

1. 어떤 사전학습 과제가 새로운 분할 과제로 zero-shot 전이될 수 있는가?
2. 여러 프롬프트를 빠르게 처리하면서 모호성까지 표현할 모델은 어떻게 설계해야 하는가?
3. 웹에 자연스럽게 존재하지 않는 대규모 마스크 레이블은 어떻게 수집할 것인가?

이에 대한 답이 각각 다음 세 요소다.

- 과제: **promptable segmentation**
- 모델: **Segment Anything Model(SAM)**
- 데이터: 모델을 주석 도구 안에 넣어 반복 개선한 **data engine**과 11M 이미지, 1.1B 마스크의 **SA-1B**

이 논문의 중요한 관점은 SAM을 완결된 semantic segmenter로 보는 것이 아니라, detector, gaze tracker, VLM, 사용자의 클릭과 조합 가능한 마스크 생성 모듈로 보는 것이다.

## 3. Promptable segmentation task

입력 이미지 `I`와 프롬프트 `p`가 주어졌을 때 모델은 유효한 마스크를 반환한다.

```text
M = SAM(I, p)
```

프롬프트는 다음과 같은 형태일 수 있다.

- foreground/background 점
- 거친 bounding box
- 이전 단계의 저해상도 mask logits
- 개념적으로는 텍스트 등 무엇을 분할할지 나타내는 정보

여기서 핵심 표현은 정답 마스크가 아니라 **valid mask**다. 예를 들어 사람의 셔츠 위 한 점은 셔츠, 상체, 사람 전체를 모두 가리킬 수 있다. 데이터셋에는 보통 그중 하나만 정답으로 기록되어 있지만, 실제로는 여러 해석이 유효하다. SAM은 한 점에서 단일 평균 마스크를 억지로 만들지 않고 여러 후보를 예측한다.

### 3.1 일반 interactive segmentation과의 차이

일반적인 interactive segmentation은 클릭이 누적되면 특정 정답 마스크에 수렴하는 것이 목표다. SAM의 사전학습 목표는 첫 프롬프트가 모호한 시점에도 적어도 하나의 합리적인 마스크를 내는 것이다. 따라서 다음 두 능력을 함께 요구한다.

- 적은 프롬프트에서의 범용성과 모호성 처리
- 추가 프롬프트와 이전 마스크를 이용한 반복 보정

여러 클릭이 주어지는 고 IoU 영역만 최적화한 모델과 목표가 다르므로, SAM이 모든 interactive benchmark에서 최상이라는 뜻은 아니다.

## 4. 전체 구조

SAM은 세 부분으로 나뉜다.

```text
image 1024x1024
  -> heavy image encoder, once per image
  -> image embedding [B, 256, 64, 64], cacheable

points / box / mask
  -> prompt encoder
  -> sparse tokens and/or dense prompt embedding

cached image embedding + prompt embeddings
  -> two-layer lightweight mask decoder
  -> mask logits + estimated IoU scores
```

이 분리는 단순한 모듈화가 아니라 지연시간 설계의 핵심이다. 같은 이미지에 여러 번 클릭할 때 이미지 인코더는 한 번만 실행하고, 프롬프트 인코더와 마스크 디코더만 반복한다. 논문이 보고하는 웹 브라우저 CPU의 약 50 ms는 **이미지 임베딩이 이미 계산된 이후의 prompt-to-mask 경로**다. ViT-H 이미지 인코더까지 포함한 end-to-end 지연시간이 아니다.

## 5. 이미지 인코더

### 5.1 ViT-H/16과 고해상도 적응

기본 모델은 MAE로 사전학습한 ViT-H/16을 사용한다. 입력 이미지는 긴 변을 기준으로 리사이즈하고 짧은 변을 패딩해 `1024 x 1024`로 만든다. patch size가 16이므로 토큰 격자는 다음과 같다.

```text
H_p = W_p = 1024 / 16 = 64
N = 64 x 64 = 4096 tokens
```

ViTDet 방식에 따라 대부분의 블록은 `14 x 14` window attention을 사용하고, 깊이에 균등하게 배치한 네 블록만 global attention을 사용한다. 이 선택은 모든 32개 블록에서 `4096 x 4096` attention을 수행하는 비용을 피하면서 전역 정보 교환 경로를 남긴다.

ViT 출력은 neck을 통과한다.

```text
ViT feature
  -> 1x1 conv, 256 channels -> LayerNorm
  -> 3x3 conv, 256 channels -> LayerNorm
  -> [B, 256, 64, 64]
```

이 `64 x 64` 임베딩은 입력보다 공간 해상도가 16배 작으며, 대화형 세션 동안 캐시할 수 있다.

### 5.2 global attention이 남기는 메모리 문제

window attention을 쓴다고 해서 이미지 인코더 전체가 모바일 친화적인 것은 아니다. 네 개의 global attention 블록은 여전히 모든 4096 토큰을 연결한다. 단순 구현이 attention score를 명시적으로 만들면 head 수를 `h=16`으로 둘 때 원소 수는 다음과 같다.

```text
h x N x N = 16 x 4096 x 4096
            = 268,435,456 elements
```

FP16 한 장만 약 `512 MiB`다. softmax 출력, backward 저장, Q/K/V를 포함하면 학습 메모리는 더 커진다. FlashAttention류의 fused kernel은 이 행렬의 materialization을 줄일 수 있지만, 모바일 NPU가 해당 attention 패턴과 window padding을 효율적으로 지원하는지는 별도 문제다.

## 6. 프롬프트 인코더

프롬프트는 sparse와 dense로 나뉜다.

### 6.1 점 프롬프트

점 좌표 `x_i`에는 위치 인코딩 `PE(x_i)`를 적용하고, foreground인지 background인지 나타내는 학습 임베딩을 더한다.

```text
e_point_i = PE(x_i) + e_fg_or_bg
e_point_i in R^256
```

좌표만 넣는 것이 아니라 점의 부호를 별도 임베딩으로 구분하므로, 같은 위치의 positive click과 negative click이 다른 토큰이 된다.

### 6.2 박스 프롬프트

박스는 하나의 pooled 벡터가 아니라 두 모서리 토큰으로 표현한다.

```text
e_tl = PE(x_top_left)     + e_top_left
e_br = PE(x_bottom_right) + e_bottom_right
```

따라서 디코더는 박스의 두 좌표와 모서리 역할을 직접 볼 수 있다.

### 6.3 마스크 프롬프트

이전 예측의 저해상도 mask logits는 입력 이미지보다 4배 낮은 해상도로 들어온다. 두 개의 `2 x 2, stride 2` convolution이 채널을 `4`, `16`으로 늘리면서 다시 4배 축소하고, 마지막 `1 x 1` convolution이 256채널로 바꾼다.

```text
[B, 1, 256, 256]
  -> Conv 2x2/s2, C=4
  -> Conv 2x2/s2, C=16
  -> Conv 1x1, C=256
  -> [B, 256, 64, 64]
```

이 결과를 이미지 임베딩에 element-wise로 더한다. 마스크 프롬프트가 없으면 모든 위치에 학습된 `no-mask` 임베딩을 더한다. 이전 마스크의 이진 결과가 아니라 **threshold 전 logits**를 넘겨 정보 손실을 줄인다는 점이 중요하다.

### 6.4 텍스트 프롬프트의 위치

본문은 CLIP 텍스트 인코더를 이용한 proof-of-concept도 제시하지만, 공개된 기본 SAM의 중심 인터페이스는 점, 박스, 마스크다. 텍스트 실험은 수동 수집 마스크의 crop에 대한 CLIP image embedding으로 SAM을 학습한 뒤, 추론 때 같은 공간에 정렬된 CLIP text embedding을 넣는다. 별도 텍스트 주석 없이 가능한 흥미로운 실험이지만 논문 스스로 완전히 robust하지 않다고 밝힌다.

## 7. 경량 마스크 디코더

### 7.1 토큰과 이미지 사이의 양방향 정보 교환

디코더의 hidden dimension은 256이고 두 개의 Transformer layer를 사용한다. 각 layer는 다음 네 단계를 갖는다.

1. prompt/output token 사이 self-attention
2. token을 query, image embedding을 key/value로 하는 cross-attention
3. 각 token에 point-wise MLP
4. image embedding을 query, token을 key/value로 하는 반대 방향 cross-attention

이를 간단히 쓰면 다음과 같다.

```text
T' = SelfAttn(T) + T
T'' = CrossAttn(query=T', key=E, value=E) + T'
T''' = MLP(T'') + T''
E' = CrossAttn(query=E, key=T''', value=T''') + E
```

`T`는 sparse prompt와 output token의 집합이고, `E`는 펼친 `4096 x 256` 이미지 임베딩이다. 모든 attention/MLP에는 residual connection과 LayerNorm이 있고 학습 시 dropout 0.1을 쓴다. 이미지 위치 인코딩과 원래 prompt token을 각 attention에 다시 더해, 여러 번 갱신된 뒤에도 좌표와 prompt type 정보가 희석되지 않게 한다.

cross-attention의 Q/K/V 차원은 256이 아니라 128로 줄이고 8 heads를 사용한다. token MLP의 내부 차원은 2048이지만 prompt token 수가 보통 20개보다 적으므로 공간 전체에 2048차원 MLP를 적용하는 것보다 훨씬 싸다.

### 7.2 dynamic mask head

두 layer 이후 image embedding을 두 번의 `2 x 2, stride 2` transposed convolution으로 4배 키운다.

```text
[B, 256, 64, 64]
  -> ConvTranspose, C=64, 128x128
  -> ConvTranspose, C=32, 256x256
  -> U in R^[B, 32, 256, 256]
```

mask output token `t_k`는 3-layer MLP를 지나 32차원 동적 분류기 벡터 `w_k`가 된다. 각 위치의 마스크 logit은 내적으로 계산한다.

```text
w_k = MLP(t_k), w_k in R^32
L_k(x, y) = w_k^T U(:, x, y)
```

즉, 고정된 K-class semantic classifier가 아니라 프롬프트에 따라 달라지는 선형 mask head다. 이 때문에 클래스 vocabulary가 없어도 임의 객체의 마스크를 생성할 수 있다.

### 7.3 모호성 처리와 IoU head

단일 프롬프트에는 세 마스크를 반환해 흔한 중첩 구조인 whole, part, subpart를 표현한다. 각 후보의 손실을 계산하되 최솟값을 갖는 후보만 역전파한다.

```text
L_mask = min_k [20 * L_focal(M_k, Y) + L_dice(M_k, Y)]
```

focal:dice 비율은 `20:1`이다. 별도 IoU output token의 MLP는 각 마스크와 정답 사이 실제 IoU를 회귀하며 MSE loss를 사용한다. 추론 시 이 예측 IoU가 후보 순위를 정한다.

여러 프롬프트가 쌓이면 모호성이 줄고 세 후보가 거의 같아지는 문제가 있다. 학습에서는 이를 피하기 위해 여러 프롬프트일 때 사용하는 네 번째 single-mask output token을 둔다. 공개 구현 관점에서는 세 개의 multimask 후보와 하나의 단일 후보가 내부에 존재한다고 이해하면 된다.

중요한 한계는 세 후보가 모든 가능한 해석을 완전하게 열거한다는 보장이 없다는 점이다. 또한 `argmax predicted IoU`가 데이터셋의 주석 관례와 다른 유효 마스크를 고르면 자동 mIoU가 낮아질 수 있다.

## 8. 배치 1 tensor shape와 activation memory 계산

아래는 논문 기본 입력 `1024 x 1024`, FP16, batch 1을 가정한 **리뷰어 계산**이다. 논문이 직접 보고한 peak memory가 아니다. `MiB = bytes / 2^20`을 사용하며 allocator, residual, LayerNorm, workspace는 제외한다.

### 8.1 주요 tensor

| 지점 | shape | FP16 크기 |
|---|---:|---:|
| 입력 RGB | `1 x 3 x 1024 x 1024` | 6.00 MiB |
| ViT-H token activation | `1 x 4096 x 1280` | 10.00 MiB |
| 해당 블록의 Q/K/V 합 | `3 x 4096 x 1280` | 30.00 MiB |
| 캐시 image embedding | `1 x 256 x 64 x 64` | 2.00 MiB |
| upsample 중간 feature | `1 x 64 x 128 x 128` | 2.00 MiB |
| mask feature | `1 x 32 x 256 x 256` | 4.00 MiB |
| 내부 4개 저해상도 mask logits | `1 x 4 x 256 x 256` | 0.50 MiB |
| 반환 3개 full-resolution logits | `1 x 3 x 1024 x 1024` | 6.00 MiB |

캐시할 최종 image embedding 자체는 FP16 2 MiB로 작다. 하지만 그것을 만드는 ViT-H의 중간 attention과 약 636M 파라미터가 병목이다. 636M 파라미터를 단순 FP16 저장하면 다음과 같다.

```text
636,000,000 x 2 bytes / 2^30 ~= 1.18 GiB
```

이는 optimizer state가 없는 추론 weight만의 하한이다. FP32면 약 2.37 GiB다.

### 8.2 window와 global attention score

`14 x 14` window를 쓰고 64를 70으로 padding한다고 단순화하면 창은 `5 x 5 = 25`개다. 16 heads의 score tensor는 다음 크기다.

```text
25 x 16 x 196 x 196 x 2 bytes ~= 29.31 MiB
```

반면 global attention은 약 512 MiB다. 따라서 peak activation은 2 MiB 캐시 크기로 추정하면 안 된다. 어떤 attention kernel이 실제 배포 backend에서 선택되는지가 결정적이다.

### 8.3 프롬프트 디코더의 계산량 감각

한 foreground point일 때 sparse token은 대략 point 1개, IoU token 1개, mask output token 4개로 `T=6`이다. 8 heads의 dense token-to-image score는 다음 정도다.

```text
8 x T x 4096 = 196,608 elements
FP16 ~= 0.375 MiB
```

반대 방향 cross-attention도 동일한 원소 수다. dynamic dot product는 내부 mask 네 개를 모두 계산한다고 하면 다음과 같다.

```text
4 masks x 256 x 256 pixels x 32 channels
= 8,388,608 MACs
```

이 계산은 이미지 인코더에 비해 매우 작다. 논문도 decoder 한 interaction이 image encoder compute의 1% 미만이라고 보고한다. 다만 full-resolution 마스크로 업샘플하고 후처리하는 비용은 앱 구현에 따라 추가된다.

### 8.4 메모리 수명 관점

실제 앱에서는 다음 생명주기를 분리해야 한다.

```text
camera/image load
  -> encoder temporaries: large, prompt 전에는 해제 가능
  -> cached embedding: 2 MiB FP16, session 동안 유지
  -> decoder temporaries: small, click마다 생성/해제
  -> displayed mask: 해상도와 저장 형식에 따라 유지
```

인코더와 디코더를 한 그래프에 고정해 compiler가 모든 buffer를 동시에 예약하면 이 이점을 놓칠 수 있다. 두 실행 모듈로 분리하고 캐시 tensor를 명시적으로 관리하는 것이 온디바이스에서 유리하다.

## 9. 학습 절차

### 9.1 interactive prompt simulation

첫 프롬프트는 같은 확률로 foreground point 또는 noisy box를 고른다.

- point: 정답 마스크 내부에서 균일 샘플링
- box: 정답 bounding box 각 좌표에 변 길이의 10% 표준편차, 최대 20 pixel의 noise 추가

이후 현재 예측과 정답의 오류 영역에서 다음 점을 고른다.

```text
false negative region -> positive point
false positive region -> negative point
```

이전 단계의 unthresholded mask logits도 dense prompt로 함께 준다. multi-mask 출력이면 predicted IoU가 가장 큰 후보를 다음 반복에 사용한다.

총 11회 interaction은 다음으로 구성된다.

```text
1 initial point or box
+ 8 error-driven points
+ 2 mask-only refinement iterations
= 11 iterations
```

8점 이후 수익이 감소했고, 두 번의 mask-only 반복은 외부 정보가 추가되지 않아도 자신의 마스크를 개선하도록 가르친다.

### 9.2 최적화 설정

논문 보고 기본 설정은 다음과 같다.

- 입력: `1024 x 1024`
- 초기화: MAE pre-trained ViT-H
- optimizer: AdamW, beta1=0.9, beta2=0.999
- learning rate: warmup 250 iterations 뒤 `8e-4`
- 총 90k iterations, 60k와 86,666에서 10배 decay
- batch size: 256 images, 256 GPUs
- weight decay: 0.1
- drop path: 0.4
- layer-wise learning-rate decay: 0.8
- 기본 설정에서는 별도 data augmentation 없음
- GPU당 임의로 최대 64 masks를 샘플링
- 이미지의 90%보다 넓은 SA-1B mask는 가볍게 필터링

보조 deep supervision은 도움이 되지 않았다고 보고한다. 이는 두 layer decoder의 중간 출력마다 loss를 거는 일반적인 DETR 계열 관행이 여기서는 필수가 아님을 보여준다.

## 10. Data engine과 SA-1B

### 10.1 1단계: assisted-manual

전문 주석자가 브라우저 도구에서 positive/negative point와 brush/eraser로 마스크를 만든다. 모델이 개선되며 평균 마스크 작성 시간이 34초에서 14초로 줄고, 이미지당 마스크가 20개에서 44개로 늘었다. 이 단계에서 120k 이미지의 4.3M 마스크를 모았고 모델을 총 여섯 번 재학습했다.

### 10.2 2단계: semi-automatic

첫 단계 마스크로 category-agnostic box detector를 학습해 확신이 높은 객체를 먼저 자동 채운다. 주석자는 빠진 덜 두드러진 객체에 집중한다. 추가로 180k 이미지에서 5.9M 마스크를 수집해 누적 10.2M이 되었고, 이미지당 평균 마스크는 72개로 증가했다.

### 10.3 3단계: fully automatic

최종 모델을 모든 이미지에 적용한다.

```text
32x32 regular point grid on full image
+ overlapping 2x2 crops with 16x16 grids
+ overlapping 4x4 crops with 8x8 grids
-> three masks per point
-> predicted-IoU filtering
-> stability filtering
-> NMS within/across crops
-> small component/hole postprocessing
```

세부 threshold는 다음과 같다.

- crop 내부와 crop 간 box NMS threshold: 0.7
- predicted IoU threshold: 88.0
- logits를 -1과 +1에서 threshold한 두 마스크의 IoU stability: 95.0 이상
- 이미지 95% 이상을 덮는 자동 마스크 제거
- 100 pixels 미만의 작은 연결 성분 제거, 100 pixels 미만 hole 채우기

최종 SA-1B는 11M 이미지와 1,129M 마스크를 가지며, 99.1%가 완전 자동 생성이다. 공개 이미지는 평균 원본 해상도 `3300 x 4950`에서 짧은 변 1500으로 축소되고 얼굴과 차량 번호판이 흐림 처리된다.

품질 점검에서 500 이미지의 약 50k 자동 마스크를 전문 주석자가 수정했고, 자동/수정 마스크 쌍의 94%가 IoU 90% 이상, 97%가 IoU 75% 이상이었다. 이는 평균이 아니라 해당 threshold를 넘은 쌍의 비율이라는 점에 유의해야 한다.

## 11. 주요 실험 결과

아래 숫자는 특별히 표시하지 않는 한 논문이 보고한 값이다.

### 11.1 23개 데이터셋 single-point 평가

SAM은 23개 서로 다른 데이터셋에서 한 개의 center point로 평가된다. 가장 높은 confidence의 마스크만 고르면 RITM보다 16개 데이터셋에서 우수했다. 정답과 가장 IoU가 높은 세 후보 중 하나를 사후 선택하는 oracle 설정에서는 모든 데이터셋에서 RITM을 앞섰다.

이 결과는 두 가지를 동시에 말한다.

- 세 후보 안에 유효한 해석이 들어 있을 가능성은 높다.
- predicted IoU ranking이 데이터셋의 단일 주석과 항상 일치하지는 않는다.

사람 평가에서는 SAM 마스크가 대체로 7점에서 9점 사이를 받았고, single-output SAM과 RITM보다 높았다. 클릭 수가 1에서 9로 늘면 기존 interactive method와의 격차가 줄었다. SAM이 많은 클릭에서 고 IoU 수렴만을 위해 특화된 모델은 아니기 때문이다.

### 11.2 zero-shot edge detection, BSDS500

| 방법 | ODS | OIS | AP | R50 |
|---|---:|---:|---:|---:|
| HED, supervised | 0.788 | 0.808 | 0.840 | 0.923 |
| EDETR, supervised | 0.840 | 0.858 | 0.896 | 0.930 |
| Canny, zero-shot | 0.600 | 0.640 | 0.580 | - |
| SAM, zero-shot | 0.768 | 0.786 | 0.794 | 0.928 |

SAM은 16x16 point grid에서 768개 후보를 생성하고 NMS 후 확률맵에 Sobel filter를 적용한다. 높은 R50과 상대적으로 낮은 precision은 BSDS 주석에 없는 합리적인 내부 경계까지 많이 생성하는 경향을 반영한다.

### 11.3 zero-shot object proposals, LVIS v1

| 방법 | AR@1000 all | small | medium | large | rare |
|---|---:|---:|---:|---:|---:|
| ViTDet-H | 63.0 | 51.7 | 80.8 | 87.0 | 58.3 |
| SAM single output | 54.9 | 42.8 | 76.7 | 74.4 | 62.0 |
| SAM multi output | 59.3 | 45.5 | 81.6 | 86.9 | 65.8 |

SAM은 전체와 small에서는 LVIS로 학습된 ViTDet-H보다 낮지만 medium, rare에서는 높다. multi-output이 모든 AR 지표에서 single-output보다 좋다는 점은 모호성 처리가 자동 proposal 생성에도 실제 이득을 준다는 근거다.

### 11.4 box-prompt instance segmentation

| 데이터셋 | 방법 | mask AP | AP small | AP medium | AP large |
|---|---|---:|---:|---:|---:|
| COCO | ViTDet-H | 51.0 | 32.0 | 54.3 | 68.9 |
| COCO | SAM | 46.5 | 30.8 | 51.0 | 61.7 |
| LVIS v1 | ViTDet-H | 46.6 | 35.0 | 58.0 | 66.3 |
| LVIS v1 | SAM | 44.7 | 32.5 | 57.6 | 65.5 |

SAM은 detector의 box를 그대로 prompt로 받아 마스크만 zero-shot 생성한다. AP는 fully supervised ViTDet보다 낮지만 사람 품질 평가는 LVIS box에서 SAM 8.1, ViTDet-H 7.9였다. 자동 AP는 주석 polygon의 hole 처리나 modal/amodal 관례까지 학습한 모델을 선호할 수 있어, 경계의 시각적 품질과 완전히 같지 않다는 해석이 제시된다.

### 11.5 데이터와 모델 크기 ablation

- assisted-manual, semi-automatic, automatic 단계를 누적할수록 23-dataset mIoU가 증가했다.
- manual/semi 데이터를 10배 oversampling해 모두 사용할 때가 가장 좋지만, automatic-only는 약 0.5 mIoU만 낮아 기본 학습을 단순화했다.
- 0.1M 이미지에서는 성능이 크게 낮아졌지만 1M 이미지, 약 100M masks는 전체 11M 이미지와 비슷했다.
- 모델 파라미터는 ViT-B 91M, ViT-L 308M, ViT-H 636M으로 증가한다.
- ViT-H는 ViT-B보다 의미 있게 좋지만 ViT-L 대비 이득은 작아 포화 경향을 보였다.

따라서 모바일 student를 설계할 때 ViT-H를 그대로 유지해야 한다는 근거는 약하다. 오히려 ViT-L에서 ViT-H로 갈 때의 작은 이득과 큰 메모리 증가가 distillation의 동기를 제공한다.

## 12. 추론 의사코드

### 12.1 대화형 사용

```python
def set_image(image):
    x = resize_long_side_and_pad(image, size=1024)
    cached_embedding = image_encoder(x)       # expensive, once
    return cached_embedding

def predict(cached_embedding, points=None, box=None, prev_logits=None):
    sparse, dense = prompt_encoder(
        points=points,
        box=box,
        mask_logits=prev_logits,
    )

    low_res_masks, predicted_iou = mask_decoder(
        image_embedding=cached_embedding + dense,
        sparse_tokens=sparse,
        multimask_output=is_ambiguous(points, box),
    )

    k = argmax(predicted_iou)
    full_mask = resize_to_original(low_res_masks[k])
    return full_mask, low_res_masks[k], predicted_iou[k]
```

### 12.2 자동 마스크 생성

```python
def automatic_masks(image):
    candidates = []
    for crop, point_grid in multiscale_crops_and_grids(image):
        embedding = image_encoder(preprocess(crop))
        for point_batch in batch(point_grid):
            masks, iou_scores = decode_multimask(embedding, point_batch)
            candidates += filter_by_iou_and_stability(masks, iou_scores)

    candidates = nms_within_and_across_crops(candidates, threshold=0.7)
    candidates = remove_small_regions_and_holes(candidates, area=100)
    return candidates
```

자동 모드에서는 한 이미지 임베딩만 계산하는 대화형 모드와 달리 여러 crop마다 encoder를 다시 실행한다. 따라서 `segment everything` 비용은 한 번의 prompt latency와 전혀 다르다.

## 13. 온디바이스 관점

### 13.1 무엇이 실제 병목인가

- **image encoder**: 1024 입력, 636M 파라미터, 네 global attention 블록이 절대적 병목이다.
- **mask decoder**: 캐시 이후에는 작고 반복 실행에 적합하다.
- **pre/postprocessing**: resize/pad, 좌표 변환, full-resolution bilinear upsample, threshold, connected component가 CPU/NPU 경계를 오갈 수 있다.
- **automatic mode**: point grid, 세 후보, crop pyramid, NMS가 호출 수와 메모리 traffic을 크게 늘린다.

논문은 prompt path 약 50 ms를 웹 브라우저 CPU에서 보고하지만, target mobile SoC의 encoder latency, peak memory, energy, thermal throttling은 보고하지 않는다. 이 값들은 **미보고**이며 직접 측정해야 한다.

### 13.2 배포 구조 권장안

```text
event or selected frame
  -> image encoder once
  -> cache 64x64x256 embedding
  -> repeated clicks/boxes use mask decoder only
  -> release cache when scene changes
```

카메라 프레임마다 SAM encoder를 실행하는 구조보다 detector를 상시 실행하고, 선택된 ROI나 사용자 요청에만 SAM 계열 student를 실행하는 구조가 현실적이다. scene이 바뀌지 않는 동안 embedding을 재사용하고, 좌표만 정확히 원본/리사이즈/패딩 공간 사이에서 변환한다.

### 13.3 최적화 우선순위

1. ViT-H 대신 ViT-B, MobileSAM/EdgeSAM류 student 비교
2. encoder INT8 또는 mixed precision, decoder FP16 유지
3. global attention kernel 지원 여부 확인
4. encoder/decoder를 별도 graph로 컴파일해 cache 수명 명시
5. output을 바로 full resolution으로 유지하지 말고 ROI/overlay에 필요한 시점에만 upsample
6. 점을 하나씩 호출하지 말고 automatic mode에서는 point batch 크기를 backend에 맞게 조정
7. static 1024가 과한 장치에서는 512/768 해상도와 작은 객체 성능을 함께 측정

### 13.4 반드시 측정할 항목

- cold encoder p50/p95 latency
- warm prompt-to-mask p50/p95 latency
- encoder와 decoder 각각의 peak RSS/accelerator memory
- cache 포함 steady-state memory
- 첫 mask까지의 시간과 추가 click 응답 시간
- 1, 10, 100 prompt에서 amortized latency
- NPU fallback operator와 host-device copy 횟수
- INT8/FP16에 따른 boundary IoU와 predicted-IoU ranking 변화
- 5분 이상 반복 실행 시 전력, 온도, throttling

## 14. 강점

1. **명확한 인터페이스**: 점, 박스, 마스크를 공통 256차원 표현으로 받아 다른 시스템과 조합하기 쉽다.
2. **비용 분리**: 이미지 인코딩과 prompt decoding을 분리해 대화형 반복 비용을 작게 만든다.
3. **모호성 모델링**: 단일 정답 관례를 그대로 따르지 않고 세 개의 유효 후보와 quality score를 출력한다.
4. **모델과 데이터의 공동 설계**: 빠른 decoder가 data engine을 가능하게 하고, 더 큰 데이터가 다시 모델을 개선한다.
5. **폭넓은 zero-shot 검증**: 23개 분할 데이터셋, edge, proposal, instance, text proof-of-concept로 composability를 보여준다.
6. **재현 가능한 세부사항**: appendix에 decoder 구조, prompt simulation, loss ratio, 자동 생성 threshold를 상세히 공개한다.

## 15. 한계와 비판적 해석

### 15.1 전체 SAM은 실시간 모델이 아니다

50 ms라는 숫자는 cached embedding 이후다. 논문도 무거운 image encoder를 포함한 전체 성능은 real-time이 아니라고 명시한다. 새로운 카메라 프레임마다 ViT-H를 다시 실행해야 하는 애플리케이션에 prompt latency만 인용하면 잘못된 시스템 결론을 낸다.

### 15.2 세 마스크는 모호성의 근사다

whole/part/subpart가 흔하지만, 장면에는 더 많은 유효 해석이 있을 수 있다. predicted IoU는 mask quality이지 사용자의 semantic intent를 직접 추정하는 점수도 아니다. VLM이나 text query를 결합할 때는 후보 마스크와 언어 의미를 별도로 정렬해야 한다.

### 15.3 작은 구조와 날카로운 경계

논문은 가는 구조를 놓치고 작은 disconnected component를 hallucinate하며, zoom-in 기반 고비용 모델보다 경계가 덜 선명할 수 있다고 인정한다. 입력을 1024로 고정해도 작은 객체가 16x patch와 4x output stride에서 충분히 표현된다는 보장은 없다.

### 15.4 automatic labels의 자기확증 위험

SA-1B의 대부분은 SAM 자신이 만든 마스크다. predicted IoU, stability, NMS로 품질을 높였지만 모델이 보기 어려운 객체가 데이터에서 반복적으로 빠질 수 있다. 1M 이미지와 전체 데이터가 비슷하다는 결과도 이미지 수보다 teacher가 만들어 내는 마스크 분포의 포화일 가능성을 함께 고려해야 한다.

### 15.5 semantic/panoptic은 직접 해결하지 않는다

SAM은 class label을 출력하지 않는다. semantic segmentation에는 마스크 병합과 클래스 분류가 필요하고, panoptic에는 중복 해소와 stuff/things 규칙이 필요하다. `segment anything`이라는 이름을 모든 분할 과제의 완성된 해답으로 해석해서는 안 된다.

### 15.6 metric과 사람 선호의 불일치

SAM은 더 자연스러운 경계를 만들고도 특정 데이터셋의 polygon convention과 다르면 AP가 낮다. 반대로 사람 평가가 높다고 downstream task AP가 자동으로 높아지는 것도 아니다. 배포 목적에 맞춰 boundary F-score, task metric, human correction time을 함께 봐야 한다.

### 15.7 책임 있는 사용

지역별 데이터 대표성에서 Africa, Latin America/Caribbean, low-income countries는 여전히 부족하다. 사람 분할에 대한 제한적 fairness 분석에서 여러 그룹의 confidence interval이 대체로 겹쳤지만, SAM을 recognition이나 의사결정 시스템의 구성요소로 넣으면 새로운 편향이 생길 수 있다. 영상 감시, 생체정보, 의료 영역은 별도의 안전성 검증이 필요하다.

## 16. 재현 체크리스트

### 모델

- [ ] 입력 resize와 padding 규칙을 동일하게 적용했는가?
- [ ] 점과 박스 좌표를 padded 1024 공간으로 정확히 변환했는가?
- [ ] image embedding shape가 `B x 256 x 64 x 64`인가?
- [ ] point의 foreground/background type embedding을 구분했는가?
- [ ] box를 두 corner token으로 표현했는가?
- [ ] dense mask prompt로 threshold 전 logits를 사용했는가?
- [ ] decoder의 token-to-image와 image-to-token attention을 모두 구현했는가?
- [ ] multimask 세 후보와 single-mask token의 선택 규칙이 맞는가?
- [ ] predicted IoU로 후보를 정렬하는가?

### 학습

- [ ] focal:dice 비율 `20:1`을 적용했는가?
- [ ] 여러 후보 중 minimum mask loss만 역전파하는가?
- [ ] IoU head에 실제 predicted-mask IoU를 MSE로 학습하는가?
- [ ] 첫 point/box와 이후 error-region click sampling을 구분했는가?
- [ ] 총 11 interactions와 두 mask-only refinement를 재현했는가?
- [ ] GPU당 mask sampling cap을 기록했는가?

### 자동 마스크 생성

- [ ] full image와 2x2, 4x4 crop point grids를 구분했는가?
- [ ] predicted-IoU와 stability threshold를 모두 적용했는가?
- [ ] crop 내부/간 NMS 우선순위가 맞는가?
- [ ] small region과 hole 후처리를 동일하게 했는가?
- [ ] encoder 호출 횟수까지 포함해 latency를 측정했는가?

### 온디바이스 평가

- [ ] encoder와 prompt decoder latency를 따로 보고했는가?
- [ ] batch=1과 실제 target resolution을 고정했는가?
- [ ] image embedding cache를 포함한 peak memory를 측정했는가?
- [ ] automatic mode의 point/crop 수를 함께 기록했는가?
- [ ] operator fallback, 전력, 온도, p95 latency를 기록했는가?

## 17. 로드맵에서의 위치와 다음 연결

SAM은 상시 실행 semantic segmenter라기보다 요청형 prompt segmenter의 기준점이다. 온디바이스 로드맵에서는 다음 순서가 자연스럽다.

```text
SAM
  -> heavy encoder / light decoder 분리 이해
  -> EdgeSAM에서 encoder distillation과 prompt-in-the-loop 학습 확인
  -> SAM 2에서 image cache를 temporal memory bank로 확장
  -> detector box 또는 VLM point로 요청형 실행
```

실제 통합 프로젝트에서는 경량 detector가 매 프레임 객체 후보를 만들고, 사용자가 선택하거나 이벤트가 발생했을 때만 prompt segmenter를 호출하는 것이 좋다. 그 결과 마스크 또는 ROI로 VLM에 전달할 visual token을 줄일 수 있다.

## 18. 핵심 정리

- SAM의 본질은 거대한 분할 backbone 하나가 아니라 **캐시 가능한 이미지 임베딩과 반복 가능한 prompt decoder의 분리**다.
- 점 하나의 모호성을 세 마스크와 predicted IoU로 처리하고, 학습에서는 minimum loss로 후보 전문화를 유도한다.
- `1024 -> 64x64x256 -> 256x256 mask logits`가 핵심 shape 흐름이다.
- SA-1B는 단순 수집물이 아니라 모델 개선과 주석 자동화를 반복한 data engine의 결과다.
- 약 50 ms는 encoder 이후 prompt 경로의 보고값이며, ViT-H encoder의 end-to-end 모바일 latency가 아니다.
- 온디바이스 연구의 다음 질문은 무거운 encoder를 어떻게 distill/quantize하고, cache와 요청형 실행으로 실제 peak memory와 p95 latency를 낮출 것인가이다.

# 70. Grounding DINO

## 논문 정보

- 원본 파일: `70_Grounding_DINO.pdf`
- 제목: **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**
- 저자: Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang
- 공개: arXiv 2023, v5 2024-07-19
- 공식 링크: [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
- 태스크: open-set object detection, open-vocabulary detection, phrase grounding, referring expression comprehension
- 핵심 키워드: DINO, grounded pre-training, tight modality fusion, language-guided query selection, sub-sentence prompt, deformable attention

## 한눈에 보는 요약

Grounding DINO는 강한 closed-set DETR 계열 detector인 DINO에 언어를 결합해, 사용자가 텍스트로 지정한 대상을 box로 찾는 모델이다. 핵심은 단순히 마지막 classifier를 text embedding으로 교체하는 데 그치지 않고, detector의 세 지점 모두에서 vision과 language를 결합하는 것이다.

```text
image -> Swin backbone --+-> feature enhancer --+-> language-guided query selection
                         |                       |                    |
text  -> BERT -----------+-----------------------+                    v
                                                              900 queries
                                                                   |
                           image features --------------------------+
                           text features ---------------------------+
                                                                   v
                                                       cross-modality decoder
                                                                   |
                                                         boxes + text phrases
```

세 지점은 다음과 같다.

1. **Feature enhancer**: image와 text token을 양방향 cross-attention으로 일찍 융합한다.
2. **Language-guided query selection**: 모든 image token이 입력 text 중 어느 token과 가장 잘 맞는지 계산하고 상위 900개 위치를 decoder query의 anchor 초기값으로 고른다.
3. **Cross-modality decoder**: 각 query가 image feature뿐 아니라 text feature에도 cross-attention한다.

Grounding DINO L은 COCO 이미지를 학습에 쓰지 않은 설정에서 COCO zero-shot `52.5 AP`, ODinW zero-shot 평균 `26.1 AP`를 보고한다. 다만 논문에서 말하는 zero-shot은 **평가 데이터셋의 train split을 쓰지 않았다**는 뜻이다. Objects365가 COCO 범주를 거의 포함하므로, 이를 곧바로 완전히 보지 못한 개념에 대한 일반화로 해석하면 안 된다.

온디바이스 관점에서는 결과보다 비용 구조가 더 중요하다. Tiny라는 이름의 Swin-T 모델도 `172M` parameters, `464 GFLOPs`, 논문 장비에서 `8.37 FPS`이다. BERT, 6-layer multimodal enhancer, 900-query 6-layer decoder가 모두 있어 상시 실행 detector로는 무겁다. 대신 서버 teacher, annotation 도구, 또는 이벤트 발생 시에만 실행하는 open-vocabulary reference로 가치가 크다.

## 문제 설정: closed-set에서 open-set으로

일반 detector는 미리 정한 `C`개 class에 대해서만 학습한다. 마지막 classification head는 대략 다음과 같다.

```math
Z = HW_{cls}^{\top},\qquad
H\in\mathbb{R}^{N_q\times d},\quad
W_{cls}\in\mathbb{R}^{C\times d}
```

여기서 `W_cls`는 학습 중 고정된 vocabulary에 묶인다. 새로운 class를 추가하려면 head를 바꾸고 box annotation으로 다시 학습해야 한다.

Open-set detector는 class weight 대신 language encoder가 만든 text feature와 region/query feature를 정렬한다.

```math
Z_{text}=H X_T^{\top},\qquad
X_T\in\mathbb{R}^{N_T\times d}
```

이제 prompt가 vocabulary 역할을 한다.

```text
person . bicycle . traffic light .
red cup on the left .
the dog behind the chair .
```

첫 줄은 category detection이고, 뒤의 두 줄은 attribute와 관계까지 포함한 referring expression이다. Grounding DINO는 둘을 `image + text -> box와 phrase`라는 같은 인터페이스로 다룬다.

논문은 open-set, open-world, open-vocabulary object detection을 같은 의미로 사용한다. 실제 연구 비교에서는 각 논문이 사용하는 학습 vocabulary, 사전학습 데이터, 평가 prompt 규칙이 다르므로 이름만 보고 동일한 설정이라고 판단해서는 안 된다.

## DINO를 기반으로 삼은 이유

Faster R-CNN류 모델은 backbone, neck, proposal, ROI head가 서로 다른 형태여서 언어 정보를 모든 단계에 넣기가 번거롭다. 반면 DINO는 encoder와 decoder가 attention block으로 반복되며 다음 장점을 이미 갖는다.

- multi-scale deformable attention
- dynamic anchor box query
- denoising training
- Hungarian bipartite matching
- decoder layer별 auxiliary loss

Transformer의 query-key-value 인터페이스는 image token과 text token을 같은 차원으로 투영해 교차 attention하기 쉽다. Grounding DINO는 DINO의 end-to-end detection 구조는 유지하면서 언어 융합 모듈을 세 군데 추가한다.

이 선택에는 대가도 있다. DINO의 기본 `900` queries를 유지하므로, 원래 DETR의 100 queries보다 decoder self-attention과 후속 head 비용이 크다. 여기에 BERT와 text cross-attention이 추가된다.

## 전체 아키텍처와 텐서 흐름

배치 크기를 `B`, hidden dimension을 `d=256`, image token 수를 `N_I`, text token 수를 `N_T`, decoder query 수를 `N_q=900`이라 하자.

```text
input image I
  -> Swin-T or Swin-L
  -> multi-scale features
  -> flatten/project/concatenate
  -> vanilla image tokens: X_I^0 [B, N_I, 256]

input text T
  -> BERT-base tokenizer and encoder
  -> vanilla text tokens: X_T^0 [B, N_T, 256]

(X_I^0, X_T^0)
  -> 6 feature enhancer layers
  -> fused X_I [B, N_I, 256], X_T [B, N_T, 256]

(X_I, X_T)
  -> language-guided query selection
  -> top indices [B, 900]
  -> dynamic anchor positions + learned content queries

queries + X_I + X_T
  -> 6 cross-modality decoder layers
  -> query states [B, 900, 256]
  -> box coordinates + token alignment logits
```

논문은 보통 `N_I > 10,000`, `N_T < 256`이라고 명시한다. text maximum length는 256이고 BPE tokenizer를 쓴다. image branch가 text branch보다 token 수에서 훨씬 크지만, cross-modal similarity와 900-query decoder 때문에 양쪽 크기가 함께 latency를 결정한다.

## Feature Enhancer

Feature enhancer는 detector의 neck에 해당한다. 각 layer는 대략 다음 연산을 포함한다.

```text
image tokens -> deformable self-attention -> FFN ------+
                                                      +-> updated image tokens
text tokens  -----------------> text-to-image cross-attention

text tokens  -> self-attention -> FFN ----------------+
                                                      +-> updated text tokens
image tokens -----------------> image-to-text cross-attention
```

방향 이름은 query가 어느 modality인지로 이해하면 쉽다.

- text-to-image: image query가 text key/value를 읽는다.
- image-to-text: text query가 image key/value를 읽는다.

Image self-attention에는 multi-scale deformable attention을 사용한다. 모든 image token 쌍에 dense attention을 만들면 `O(N_I^2)`가 되어 10,000 token에서 실용적이지 않다. Deformable attention은 각 query가 각 scale의 소수 sampling point만 읽도록 제한해 비용을 대략 token 수에 선형으로 낮춘다.

반면 cross-modal attention은 다음 모양을 갖는다.

```math
A_{I\leftarrow T}=\mathrm{softmax}
\left(\frac{Q_IK_T^{\top}}{\sqrt{d_h}}\right)
\in\mathbb{R}^{B\times h\times N_I\times N_T}
```

```math
A_{T\leftarrow I}=\mathrm{softmax}
\left(\frac{Q_TK_I^{\top}}{\sqrt{d_h}}\right)
\in\mathbb{R}^{B\times h\times N_T\times N_I}
```

수학적 원소 수는 둘 다 `B h N_I N_T`이다. 실제 구현은 양 방향 score를 재사용하거나 memory-efficient kernel을 쓸 수 있으므로, 이 식만으로 실측 peak memory를 단정할 수는 없다.

논문의 ablation에서 encoder fusion을 제거하면 O365로만 pre-train한 Swin-T의 성능이 다음처럼 하락한다.

| 설정 | COCO zero-shot AP | COCO fine-tune AP | LVIS zero-shot AP |
|---|---:|---:|---:|
| 전체 모델 | 46.7 | 56.9 | 16.1 |
| encoder fusion 제거 | 45.8 | 56.1 | 13.1 |

특히 LVIS에서 `-3.0 AP`로 가장 큰 ablation 차이가 난다. 언어와 시각 특징을 query 초기화나 마지막 head에서만 만나는 것보다, encoder 단계에서 반복적으로 정렬하는 것이 long-tail transfer에 중요하다는 증거다.

## Language-Guided Query Selection

### 핵심 수식

Feature enhancer 출력이 다음과 같다고 하자.

```math
X_I\in\mathbb{R}^{N_I\times d},\qquad
X_T\in\mathbb{R}^{N_T\times d}
```

모든 image-text token의 내적을 계산한다.

```math
S=X_I X_T^{\top}\in\mathbb{R}^{N_I\times N_T}
```

각 image token에 대해 가장 잘 맞는 text token의 score만 남긴다.

```math
s_i=\max_{1\leq t\leq N_T}S_{i,t}
```

그중 상위 `N_q`개 image 위치를 선택한다.

```math
\mathcal{I}_{N_q}=\mathrm{TopN_q}
\left(\mathrm{Max}_{-1}(X_I X_T^{\top})\right)
```

논문의 PyTorch형 의사 코드는 거의 다음과 같다.

```python
# image_feat: [B, N_I, d]
# text_feat:  [B, N_T, d]
# num_query:  900
logits = torch.einsum("bic,btc->bit", image_feat, text_feat)
scores = logits.max(dim=-1).values       # [B, N_I]
topk_idx = scores.topk(900, dim=1).indices
```

논문 Algorithm 1의 마지막 변수에는 `logits_per_img_feat`와 `logits_per_img_feature`가 혼용되는 작은 표기 오류가 있다. 개념은 동일하며 `scores`에 top-k를 적용하면 된다.

### Query의 content와 position

DINO의 mixed query selection을 따른다.

- positional part: encoder에서 선택한 위치로 초기화한 dynamic anchor box
- content part: 학습 가능한 query embedding

즉 text가 직접 900개의 content vector를 만드는 것은 아니다. Text는 어느 image 위치를 anchor 후보로 고를지 결정하고, 학습 가능한 content query가 그 위치에서 image와 text를 다시 읽으며 box를 정제한다.

Static query selection으로 바꾸면 COCO zero-shot `46.7 -> 46.3`, LVIS zero-shot `16.1 -> 13.6`으로 내려간다. Prompt에 따라 후보 위치가 달라지는 성질이 특히 큰 vocabulary에서 유효하다.

### Top-k의 의미와 한계

`max`는 한 image token이 text 중 단 하나와만 강하게 맞아도 후보가 되게 한다. 계산은 단순하지만 다음 문제가 있다.

- 여러 token으로 이뤄진 phrase의 조합 점수를 직접 표현하지 않는다.
- 흔한 단어나 punctuation에 높은 score가 생기면 후보가 오염될 수 있다.
- top-k는 미분 가능한 soft selection이 아니며 선택 경계 주변 정보가 사라진다.
- 900개가 대부분 background인 이미지에서는 decoder가 많은 negative query를 처리한다.

그럼에도 dense region proposal head를 별도로 만들지 않고 text-conditioned proposal을 DINO 구조 안에 넣는다는 장점이 있다.

## Cross-Modality Decoder

각 decoder layer의 순서는 다음과 같다.

```text
queries [B, 900, 256]
  -> query self-attention
  -> deformable image cross-attention
  -> text cross-attention
  -> FFN
  -> refined queries
```

Text cross-attention을 식으로 쓰면 다음과 같다.

```math
Q' = Q + \mathrm{MHA}(Q, X_T, X_T)
```

Image cross-attention은 dense `QX_I^T` 대신 deformable sampling을 사용한다. 각 layer는 box reference point를 갱신하고, auxiliary box 및 text alignment loss를 받는다.

Text cross-attention을 제거한 ablation은 COCO zero-shot `46.7 -> 46.1`, fine-tune `56.9 -> 56.3`, LVIS zero-shot `16.1 -> 14.3`이다. Encoder fusion보다 이득은 작지만, query가 마지막까지 prompt 의미를 유지하도록 돕는다.

900 query self-attention은 query 수에 대해 제곱 비용이다.

```math
\text{self-attention score elements per head}=N_q^2=900^2=810,000
```

Query 수를 1200 또는 1500으로 늘린 appendix 실험은 LVIS rare AP를 개선하지 못했다.

| Query 수 | COCO AP | LVIS AP | LVIS APr | LVIS APf |
|---:|---:|---:|---:|---:|
| 900 | 46.7 | 15.8 | 9.4 | 28.4 |
| 1200 | 46.7 | 15.7 | 9.0 | 29.1 |
| 1500 | 46.9 | 15.8 | 9.2 | 29.0 |

900개로 이미 대부분 object를 덮고, query를 늘리면 sampled category와 background 사이의 data imbalance가 커진다는 것이 논문의 해석이다.

## Sub-Sentence Level Text Representation

Detection dataset의 class 이름을 한 번에 prompt로 만들면 다음과 같다.

```text
cat . baseball glove . person . mouse .
```

논문은 text 표현 방식을 세 가지로 구분한다.

### Sentence level

전체 문장을 하나의 embedding으로 압축한다. 서로 다른 category의 상호작용은 줄지만, 어느 단어가 어느 box에 대응하는지에 필요한 fine-grained token 정보가 사라진다.

### Word level

모든 token feature를 유지한다. 하지만 임의 순서로 붙인 `cat`과 `baseball glove`가 BERT self-attention에서 서로 영향을 준다. 이 관계는 자연 문장의 문맥이 아니라 data loader가 만든 우연이다.

### Sub-sentence level

Punctuation으로 구분된 phrase 내부 attention은 허용하고 phrase 사이 attention은 mask한다.

```text
[cat] . [baseball glove] . [person] . [mouse] .
  |             |             |          |
within-phrase attention: allowed
cross-phrase attention: blocked
```

개념적 mask는 다음과 같다.

```math
M_{ij}=\begin{cases}
0, & \text{token }i,j\text{가 같은 sub-sentence에 속할 때}\\
-\infty, & \text{그 외}
\end{cases}
```

BERT attention logit에 더한다.

```math
A=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_h}}+M\right)
```

Word-level prompt로 되돌리면 LVIS zero-shot `16.1 -> 15.6 AP`로 낮아진다. 차이는 작지만 추가 모델 parameter나 image-side 연산 없이 얻는 이득이다. 실제 재현에서는 punctuation, special token, padding token의 segment 할당을 원 구현과 맞춰야 한다.

## Phrase Classification과 Box Loss

각 decoder query와 text token의 내적으로 classification logit을 만든다.

```math
Z=QX_T^{\top}\in\mathbb{R}^{N_q\times N_T}
```

GLIP처럼 token별 focal loss를 적용한다. Box에는 L1과 generalized IoU loss를 쓴다.

Hungarian matching cost를 단순화해 쓰면 다음과 같다.

```math
\mathcal{C}(i,j)=
\lambda_{cls}\mathcal{C}_{focal}(i,j)
+\lambda_{L1}\|b_i-\hat b_j\|_1
+\lambda_{giou}\mathcal{C}_{giou}(b_i,\hat b_j)
```

Main text가 명시한 matching weight는 `2.0, 5.0, 2.0`, 최종 loss weight는 `1.0, 5.0, 2.0`이다. 그러나 appendix Table 8은 `set cost class=1.0`, `ce loss coef=2.0`으로 반대로 적혀 있다. 논문 내부 표기가 일치하지 않으므로 재현할 때는 사용한 checkpoint의 config를 기준으로 해야 한다.

Decoder 각 layer와 encoder output에도 auxiliary loss를 둔다. DINO의 denoising training과 bipartite matching을 계승하므로 core detector는 별도의 class-wise NMS를 학습 목표로 요구하지 않는다. 다만 실제 서비스는 900개 출력 중 score threshold와 top-k filtering을 수행하며, downstream 정책에 따라 중복 제거가 여전히 필요할 수 있다.

## 학습 데이터의 통합

Grounding DINO는 세 종류의 데이터를 phrase grounding 형태로 통합한다.

| 데이터 종류 | 예 | 변환 방식 |
|---|---|---|
| detection | COCO, Objects365, OpenImages | category name들을 text prompt로 연결하고 box와 token을 대응 |
| grounding | GoldG, RefCOCO/+/g | 주어진 phrase와 box pair를 직접 사용 |
| caption | Cap4M | GLIP teacher가 만든 pseudo box-label을 사용 |

GoldG는 Flickr30K Entities와 Visual Genome을 MDETR 방식으로 전처리한 데이터다. `RefC`는 RefCOCO, RefCOCO+, RefCOCOg를 뜻한다.

Objects365도 두 버전이 있다.

- O365v1: 약 600K images, Grounding DINO T의 공정 비교에 사용
- O365v2: 약 1.7M images, Grounding DINO L에 사용

따라서 `Grounding DINO T O365`와 `Grounding DINO L O365`는 backbone만 다른 완전한 통제 실험이 아니다. 데이터 크기, 추가 OpenImages/COCO/RefC 사용 여부까지 함께 기록해야 한다.

## 학습 설정

논문의 기본 hyperparameter는 다음과 같다.

| 항목 | 값 |
|---|---:|
| optimizer | AdamW |
| main learning rate | `1e-4` |
| image backbone LR | `1e-5` |
| text backbone LR | `1e-5` |
| weight decay | `1e-4` |
| gradient clip max norm | `0.1` |
| encoder layers | 6 |
| decoder layers | 6 |
| hidden dimension | 256 |
| FFN dimension | 2048 |
| attention heads | 8 |
| dropout | 0.0 |
| decoder queries | 900 |
| max text tokens | 256 |

Grounding DINO T는 16 V100, total batch 32로 학습한다. Swin-T에서 stride 8, 16, 32 feature를 뽑고 stride 32를 한 번 더 downsample해 stride 64 feature를 만든다. DINO 관례로 이를 4-scale이라 부른다.

Grounding DINO L은 64 A100, total batch 64를 사용한다. Swin-L의 stride 4, 8, 16, 32 네 scale을 직접 쓴다. 이 자원 요구량은 개인 연구자가 전체 pre-training을 그대로 재현하기 어렵다는 점을 보여준다.

## COCO 결과를 올바르게 읽기

### Zero-shot와 fine-tuning

| 모델 | Pre-training | COCO zero-shot AP | COCO fine-tune val/test-dev AP |
|---|---|---:|---:|
| Grounding DINO T | O365 | 46.7 | 56.9 / - |
| Grounding DINO T | O365, GoldG | 48.1 | 57.1 / - |
| Grounding DINO T | O365, GoldG, Cap4M | 48.4 | 57.2 / - |
| Grounding DINO L | O365, OI, GoldG | 52.5 | 62.6 / 62.7 |
| Grounding DINO L, 1.5x input | 같은 조건 후 COCO fine-tune | - | 63.0 / 63.0 |

Grounding data를 더하면 Tiny zero-shot이 `46.7 -> 48.1`, caption data까지 더하면 `48.4`가 된다. 하지만 O365가 COCO category를 거의 포함한다. 논문 각주도 이 사실과 근사 category mapping을 명시한다.

그러므로 여기서 검증된 것은 대체로 다음이다.

- COCO train image를 쓰지 않은 **domain/dataset transfer**
- 언어 interface를 통한 category mapping
- large-scale detection/grounding pre-training의 효과

다음까지 입증한 것은 아니다.

- 사전학습 어디에도 없던 의미 개념의 완전한 zero-shot 이해
- 임의 자연어 표현에 대한 안정적인 compositional reasoning
- 사용자 환경에서의 calibration과 false-positive 안전성

COCO closed-set 1x appendix에서 ResNet-50 Grounding DINO는 `48.1 AP`, 원 DINO-4scale은 `49.0 AP`다. 언어 모듈을 넣는다고 같은 데이터의 closed-set 성능이 자동으로 좋아지는 것은 아니며 최적화가 더 어려울 수 있다.

## LVIS: Long-tail에서 드러나는 데이터 의존성

| 모델 | Pre-training | LVIS zero-shot AP | APr / APc / APf |
|---|---|---:|---:|
| GLIP-T | O365, GoldG | 24.9 | 17.7 / 19.5 / 31.0 |
| Grounding DINO T | O365, GoldG | 25.6 | 14.4 / 19.6 / 32.2 |
| Grounding DINO T | + Cap4M | 27.4 | 18.1 / 23.3 / 32.7 |
| Grounding DINO L | 더 큰 혼합 데이터 | 33.9 | 22.2 / 30.7 / 38.8 |
| DetCLIPv2-T | O365, GoldG, CC15M | 40.4 | 36.0 / 41.7 / 40.0 |

Grounding DINO T는 전체 AP에서 GLIP-T보다 높지만 rare AP는 `14.4`로 GLIP의 `17.7`보다 낮다. DETR-like 구조의 rare-category 약점과 학습 데이터 분포가 함께 작용한다.

Cap4M 추가 시 `25.6 -> 27.4 AP`, rare AP도 `14.4 -> 18.1`로 좋아진다. 반면 훨씬 큰 CC15M을 쓴 DetCLIPv2는 `40.4 AP`다. Open-vocabulary 결과는 architecture ranking이라기보다 **architecture, concept coverage, pseudo-label quality, data distribution의 결합 결과**로 읽어야 한다.

LVIS fine-tuning에서는 O365+GoldG로 pre-train한 Grounding DINO T가 `52.1 AP`를 기록하며, 같은 표의 DetCLIPv2-T `50.7 AP`보다 높다. Target annotation을 주면 representation이 강하다는 증거지만 zero-shot gap의 원인이 사라졌다는 뜻은 아니다.

## ODinW: 평균과 중앙값을 같이 봐야 하는 이유

ODinW는 서로 성격이 다른 35개 detection dataset을 묶는다.

| 설정 | 모델 | Parameters | AP average | AP median |
|---|---|---:|---:|---:|
| zero-shot | Grounding DINO T, O365+GoldG | 172M | 20.0 | 9.5 |
| zero-shot | Grounding DINO T, +Cap4M | 172M | 22.3 | 11.9 |
| zero-shot | Grounding DINO L | 341M | 26.1 | 18.4 |
| few-shot | Grounding DINO T | 172M | 46.4 | 51.1 |
| full-shot | Grounding DINO T | 172M | 70.7 | 76.2 |

평균과 중앙값 차이가 크다. 일부 dataset에서 높은 AP가 평균을 끌어올리지만 다수 dataset은 낮을 수 있다. 실제 배포 도메인을 대표하는 별도 benchmark가 없으면 `26.1 mean AP`만으로 성능을 예측하기 어렵다.

논문 세부 표에서도 uncommon class나 특수 촬영 조건에서 거의 실패하는 dataset이 있다. 따라서 사용자 prompt를 바꿔 보는 qualitative demo보다 도메인별 AP, class별 recall, false positive rate, threshold sensitivity를 측정해야 한다.

## Referring Expression Comprehension

REC는 text 하나에 해당하는 가장 높은 score의 box 하나를 출력한다. 단순 category prompt보다 `왼쪽 끝의 남자`, `안경 쓴 여자`처럼 attribute와 relation이 중요하다.

RefC 없이 O365+GoldG로 pre-train한 Grounding DINO T의 RefCOCO val은 `50.41`이다. RefC를 pre-training에 넣으면 `73.98`, 이어 fine-tuning하면 `89.19`로 크게 오른다. 이는 범용 grounding data만으로 fine-grained REC를 해결하지 못함을 보여준다.

데이터를 넣는 효과에는 trade-off가 있다.

| Pre-training | COCO zero-shot | COCO fine-tune | LVIS zero-shot | ODinW zero-shot |
|---|---:|---:|---:|---:|
| O365, GoldG | 48.1 | 57.1 | 25.6 | 20.0 |
| + RefC | 48.5 | 57.3 | 21.9 | 17.7 |
| + COCO | 56.1 | 57.5 | 22.3 | 17.4 |

RefC는 COCO 관련 지표를 조금 높이지만 LVIS와 ODinW를 낮춘다. COCO를 넣은 `56.1`은 당연히 진짜 COCO zero-shot이 아니며 논문도 이를 명시한다. 또 COCO와 RefC 사이에는 image overlap 가능성이 있어 Grounding DINO L의 REC 수치에는 data leak 주의 표시가 있다.

BERT-base와 BERT-large 비교에서는 base가 대부분 같거나 조금 낫다. 예를 들어 RefCOCO val은 `87.4` 대 `87.0`이다. 이 설정에서는 언어 모델 크기보다 detection branch와 target data가 병목이라는 논문의 해석이 타당하다.

## Ablation을 통한 기여도 분해

O365만 사용한 Swin-T 결과를 한 표로 정리하면 다음과 같다.

| ID | 변경 | COCO zero-shot | COCO fine-tune | LVIS zero-shot |
|---:|---|---:|---:|---:|
| 0 | 전체 모델 | 46.7 | 56.9 | 16.1 |
| 1 | encoder fusion 제거 | 45.8 | 56.1 | 13.1 |
| 2 | static query selection | 46.3 | 56.6 | 13.6 |
| 3 | text cross-attention 제거 | 46.1 | 56.3 | 14.3 |
| 4 | word-level prompt | 46.4 | 56.6 | 15.6 |

이 표에서 중요한 패턴은 COCO fine-tune보다 LVIS zero-shot 차이가 훨씬 크다는 것이다. Target data가 충분할 때는 head가 class 경계를 다시 맞출 수 있지만, target annotation이 없고 vocabulary가 클수록 각 융합 장치의 inductive bias가 중요해진다.

효과 크기만 보면 encoder fusion이 가장 크다. 그러나 비용도 가장 크다. 온디바이스 최적화에서는 단순히 가장 작은 AP 손실 모듈부터 지울 것이 아니라, 각 모듈의 latency와 memory를 측정해 Pareto curve를 그려야 한다.

## Activation Memory와 연산량 손계산

아래 값은 논문이 보고한 실측이 아니라 `B=1`, `N_I=10,000`, `N_T=256`, `N_q=900`, `d=256`, `h=8`, FP16을 가정한 리뷰어 계산이다. Framework가 저장하는 intermediate와 fused kernel에 따라 실제 peak memory는 크게 달라진다.

### 기본 feature 저장량

```math
X_I: 10,000\times256\times2\text{ bytes}
=5,120,000\text{ bytes}\approx4.88\text{ MiB}
```

```math
X_T: 256\times256\times2
=131,072\text{ bytes}=0.125\text{ MiB}
```

```math
Q: 900\times256\times2
=460,800\text{ bytes}\approx0.44\text{ MiB}
```

이 값은 한 tensor만 저장한 크기다. Training에서는 Q/K/V, residual, FFN activation, gradient, optimizer state가 더해진다.

### Query selection similarity

```math
S: 10,000\times256\times2
=5,120,000\text{ bytes}\approx4.88\text{ MiB}
```

내적의 대략적인 scalar multiply-accumulate 수는 다음과 같다.

```math
N_I N_T d=10,000\times256\times256
=655,360,000
```

약 655M개의 길이 256 dot-product 성분을 처리한다. 엄밀한 FLOPs 표기는 multiply와 add를 1개 또는 2개로 세는 관례에 따라 달라지므로 여기서는 MAC-like count로만 본다. Text가 짧아지면 이 부분은 선형으로 줄어든다.

### Decoder attention map

Query self-attention score는 layer당 다음 크기다.

```math
8\times900\times900\times2
=12,960,000\text{ bytes}\approx12.36\text{ MiB}
```

Text cross-attention은 다음과 같다.

```math
8\times900\times256\times2
=3,686,400\text{ bytes}\approx3.52\text{ MiB}
```

두 score가 순차적으로 생성되고 즉시 해제되면 합이 peak가 되지 않을 수 있다. Flash-style 또는 fused attention은 score matrix 전체를 materialize하지 않을 수도 있다. 반대로 일반 eager implementation은 softmax, dropout, backward용 buffer를 별도로 저장한다.

Dense image cross-attention을 썼다면 head를 제외해도 `900 x 10,000` score가 생긴다. Grounding DINO가 deformable image cross-attention을 쓰는 이유가 여기에 있다. 다만 deformable sampling은 모바일 NPU에서 native operator가 없으면 gather, reshape, interpolation로 분해되어 이론 FLOPs보다 느릴 수 있다.

## 논문이 보고한 효율과 빠진 지표

Appendix의 비교는 다음과 같다.

| 모델 | Parameters | GFLOPs | FPS |
|---|---:|---:|---:|
| GLIP-T | 232M | 488G | 6.11 |
| Grounding DINO T | 172M | 464G | 8.37 |

Grounding DINO T가 GLIP-T보다 작고 빠르지만, 이 수치는 모바일 실시간을 뜻하지 않는다. 논문 표는 다음 정보를 충분히 제공하지 않는다.

- 정확한 GPU와 precision 및 warm-up 조건
- 입력 해상도와 prompt 길이를 포함한 FPS 측정 protocol
- p50/p95 latency
- peak activation memory
- text tokenizer와 host-device transfer 포함 여부
- 전력, 온도, throttling

따라서 `8.37 FPS`를 타깃 장치 latency로 변환하거나 GFLOPs 비율로 NPU 속도를 예측해서는 안 된다.

## 실제 구현 의사 코드

```python
class GroundingDINO(nn.Module):
    def forward(self, images, texts, text_mask):
        image_pyramid = self.image_backbone(images)
        image_tokens, spatial_shapes = self.project_and_flatten(image_pyramid)

        text_tokens = self.text_backbone(texts, text_mask)
        text_tokens = self.text_proj(text_tokens)

        for layer in self.feature_enhancer:
            image_tokens, text_tokens = layer(
                image_tokens,
                text_tokens,
                spatial_shapes=spatial_shapes,
                text_mask=text_mask,
            )

        similarity = torch.einsum(
            "bic,btc->bit", image_tokens, text_tokens
        )
        image_scores = similarity.masked_fill(
            ~text_mask[:, None, :], float("-inf")
        ).amax(dim=-1)
        topk_idx = image_scores.topk(self.num_queries, dim=1).indices

        anchors = gather_encoder_anchors(topk_idx)
        queries = self.learned_content_queries.expand(images.size(0), -1, -1)

        outputs = []
        for layer in self.decoder:
            queries, anchors = layer(
                queries=queries,
                anchors=anchors,
                image_tokens=image_tokens,
                text_tokens=text_tokens,
                spatial_shapes=spatial_shapes,
                text_mask=text_mask,
            )
            boxes = self.box_head(queries, anchors)
            token_logits = queries @ text_tokens.transpose(-1, -2)
            outputs.append({"boxes": boxes, "logits": token_logits})

        return outputs
```

실제 구현에는 multi-scale level index, valid ratio, positional encoding, denoising queries, attention mask, auxiliary heads가 더 필요하다. 위 코드는 tensor 흐름을 보여주기 위한 축약형이다.

## 재현 시 자주 틀리는 부분

### Prompt formatting

Category 사이에 punctuation을 넣고 sub-sentence mask를 정확히 구성해야 한다. `cat. dog.`와 `cat dog`는 같은 입력이 아니다. 대소문자, synonym, plural form도 score를 바꿀 수 있으므로 evaluation prompt template을 고정해야 한다.

### Padding mask

Padded text token은 query selection의 `max`에서 반드시 제외해야 한다. Padding logit이 0이고 실제 logit이 음수라면 padding이 최대값이 되는 오류가 생길 수 있다. Attention mask와 loss mask도 같은 token indexing을 써야 한다.

### Threshold 두 종류

공개 구현은 보통 box confidence threshold와 text token threshold를 구분한다. 하나는 query를 남길지, 다른 하나는 어떤 phrase와 연결할지를 정한다. Benchmark와 demo의 threshold가 다르면 AP와 qualitative output을 비교할 수 없다.

### Dataset overlap

COCO, RefC, Visual Genome, OpenImages, caption pseudo-label 사이 image 및 concept overlap을 기록해야 한다. `train split을 사용하지 않음`만으로 unseen concept을 보장하지 않는다.

### Config가 논문 표보다 우선

Matching class cost와 final classification loss weight가 main text와 appendix에서 일치하지 않는다. Release checkpoint의 config, commit hash, tokenizer 버전을 함께 보존해야 한다.

## 강점

1. **세 단계의 일관된 fusion 설계**: neck, query initialization, decoder를 한 architecture 안에서 언어 조건화한다.
2. **강한 localization backbone 활용**: CLIP의 global embedding에만 의존하지 않고 DINO의 box refinement를 보존한다.
3. **다양한 데이터 형식 통합**: detection, grounding, caption pseudo-label을 token-region alignment로 묶는다.
4. **Open-set와 REC를 함께 평가**: category list뿐 아니라 자연어 지시의 한계도 드러낸다.
5. **SAM 등 다른 foundation model과 연결하기 좋은 box interface**: text-to-box teacher/reference로 활용하기 쉽다.
6. **Ablation이 구조적 주장과 대응**: encoder fusion, query selection, decoder text attention, prompt mask를 각각 제거한다.

## 한계와 비판적 해석

### Zero-shot 용어의 범위

논문의 정의는 target dataset train split 미사용이다. 사전학습 dataset이 target category를 포함할 수 있다. 그래서 `open-set` 능력을 architecture의 순수 compositional generalization으로 과대평가하기 쉽다.

### 데이터 규모와 분포가 결과를 지배

LVIS에서 DetCLIPv2가 더 큰 데이터를 사용해 훨씬 높고, RefC 추가는 REC를 높이지만 다른 benchmark를 낮춘다. 범용성은 하나의 checkpoint와 하나의 평균 AP로 요약되지 않는다.

### False positive와 hallucination

논문 스스로 일부 false positive를 limitation으로 든다. Open-vocabulary detector는 prompt에 어떤 물체가 있다고 전제하는 듯한 score를 낼 수 있다. 안전 관련 시스템에서는 unknown rejection, calibration, negative prompt set, temporal consistency가 필요하다.

### Segmentation 부재

모델 출력은 box와 phrase다. Pixel mask가 필요하면 SAM/EdgeSAM 같은 별도 segmenter를 붙여야 한다. 이때 detector box error가 mask 단계로 전파된다.

### 실시간 및 온디바이스 비용

172M parameter의 Tiny 모델, 464G 연산, BERT와 12개 multimodal Transformer layer는 상시 카메라 detector에 크다. Operator 지원이 약한 deformable attention과 dynamic top-k도 배포 난도를 높인다.

### REC의 formulation mismatch

일반 grounding pre-training은 한 phrase에 여러 box를 내는 경향이 있지만 RefCOCO는 보통 한 expression에 하나의 target을 요구한다. Target-specific data가 없을 때 성능이 낮은 이유다.

## 온디바이스 배포 전략

Grounding DINO를 그대로 매 frame 실행하기보다는 역할을 분리하는 편이 현실적이다.

```text
camera frames
  -> small closed/open-vocabulary detector every frame
  -> event or user query?
       no: reuse tracking result
       yes: Grounding DINO teacher/reference on selected frame
              -> selected box
              -> EdgeSAM/SAM mask
              -> ROI-cropped VLM response
```

### 가능한 최적화

- **Prompt cache**: BERT의 vanilla text embedding은 같은 prompt에 재사용할 수 있다.
- **주의할 점**: Feature enhancer가 image와 text를 융합하므로 fused image feature는 임의의 새 prompt에 그대로 재사용할 수 없다.
- **짧은 vocabulary**: `N_T`를 줄이면 query selection과 cross-attention이 선형으로 감소한다.
- **Query pruning**: 900 queries를 작은 student에서 100-300개로 줄이고 recall 손실을 측정한다.
- **Backbone 교체**: Swin 대신 MobileNetV4/EfficientViT 계열을 쓰되 multi-scale projection을 다시 학습한다.
- **Distillation**: Grounding DINO의 box, token alignment, encoder feature를 YOLO-World/YOLOE 계열 student에 증류한다.
- **Mixed precision**: image/text projection과 FFN INT8, 민감한 attention/box head FP16 같은 task별 조합을 탐색한다.
- **Operator audit**: multi-scale deformable attention, `topk`, `gather`, dynamic text length가 NPU delegate에서 지원되는지 먼저 확인한다.

텍스트 prompt가 고정 vocabulary라면 매번 BERT를 실행할 이유가 없다. 그러나 사용자가 자유롭게 prompt를 바꾸는 서비스라면 tokenizer와 BERT latency를 TTFT에 포함해야 한다.

## 이 로드맵에서의 위치

Grounding DINO는 다음 연결고리다.

```text
GLIP
  -> detection을 grounding으로 재정의, early fusion

Grounding DINO
  -> DINO에 early fusion + text-guided query + decoder fusion
  -> 강한 text-to-box teacher/reference

YOLO-World / YOLOE
  -> 실시간 open-vocabulary student 방향

SAM / EdgeSAM
  -> box-to-mask

VLM
  -> 선택된 ROI를 설명하거나 질의 응답
```

통합 프로젝트에서 Grounding DINO는 모든 frame을 처리하는 경량 detector보다 다음 용도에 적합하다.

- 새로운 사용자 vocabulary로 pseudo-label 생성
- student detector 학습용 teacher
- 실패 사례에서 고비용 second-stage verifier
- text query로 ROI를 고른 뒤 segmenter와 VLM 호출

## 권장 재현 실험

### 실험 1: Prompt sensitivity

같은 category에 대해 다음을 비교한다.

```text
dog
a dog
dogs
puppy
dog . cat . person .
```

Class별 AP, recall, false positives/image와 token score 변화를 기록한다.

### 실험 2: Query count ablation

`N_q={100, 300, 600, 900}`으로 바꾸고 다음을 측정한다.

- COCO/LVIS AP와 small-object recall
- decoder latency
- peak activation memory
- p50/p95 latency
- output filtering 시간

논문의 1200/1500 실험은 900 이상만 보므로, 온디바이스에 중요한 저-query 구간은 별도 실험이 필요하다.

### 실험 3: Fusion 단계별 비용

논문 ablation 네 변형에 대해 AP뿐 아니라 실제 장치 latency를 잰다.

```text
full
w/o encoder fusion
static query selection
w/o decoder text cross-attention
word-level prompt
```

각 모듈의 `delta AP / delta ms / delta MiB`를 계산하면 배포용 Pareto frontier를 만들 수 있다.

### 실험 4: Box-to-mask pipeline

Grounding DINO와 EdgeSAM을 결합해 다음 오류를 분리한다.

- detector가 target을 놓침
- detector box는 맞지만 mask가 틀림
- multiple candidate 중 threshold가 틀림
- VLM이 잘못된 ROI를 설명함

End-to-end 성공률만 보면 어느 단계가 병목인지 알 수 없다.

### 실험 5: Student distillation

Grounding DINO를 offline teacher로 두고 경량 detector에 다음 신호를 증류한다.

- teacher box와 confidence
- phrase-token alignment logit
- hard negative region
- multi-scale image feature

Seen, synonym, rare, truly novel concept split을 별도로 평가해야 teacher data exposure를 구분할 수 있다.

## 실험 기록 템플릿

```text
Model/checkpoint:
Code commit:
Image resolution / resize policy:
Prompt and punctuation:
Text token length:
Box threshold / text threshold:
Number of queries:
Precision by module:
Execution provider / delegate:

Dataset and overlap audit:
AP / AP50 / AP75:
APr / APc / APf:
small-object recall:
false positives per image:

p50 latency:
p95 latency:
tokenization + BERT latency:
image backbone latency:
feature enhancer latency:
query selection latency:
decoder latency:
post-processing latency:
peak memory:
power / temperature / throttling:
```

## 최종 평가

Grounding DINO의 가장 중요한 공헌은 `text embedding을 classifier로 사용한다`는 한 줄이 아니다. Detector의 feature enhancement, proposal/query initialization, iterative decoding을 모두 text-conditioned하게 만든 **tight fusion 설계**다. 이 덕분에 category list와 referring expression을 하나의 box prediction interface로 처리하고, 강력한 zero-shot dataset transfer 성능을 얻는다.

동시에 이 논문은 open-vocabulary 연구를 읽을 때의 주의점도 잘 보여준다. COCO `52.5 AP`는 강한 결과지만 Objects365의 category coverage, 더 큰 pre-training data, prompt 규칙, target split 미사용이라는 zero-shot 정의를 함께 봐야 한다. LVIS rare class와 REC 결과는 데이터 분포가 architecture 못지않게 중요함을 보여준다.

온디바이스 연구에서는 Grounding DINO를 최종 상시 detector라기보다, 정확한 text-to-box teacher이자 통합 pipeline의 고비용 reference로 보는 것이 적절하다. 핵심 연구 질문은 이 모델의 능력을 얼마나 보존하면서 `BERT + 6-layer enhancer + 900-query 6-layer decoder`를 더 작은 backbone, 적은 query, sparse fusion, caching, distillation으로 줄일 수 있는가이다.

## 체크리스트

- [ ] 논문의 zero-shot 정의와 pre-training concept overlap을 구분했는가?
- [ ] Prompt punctuation과 sub-sentence attention mask를 고정했는가?
- [ ] Text padding token을 query-selection max에서 제외했는가?
- [ ] Main text와 appendix의 matching/loss weight 불일치를 config로 확인했는가?
- [ ] 900-query self-attention과 text cross-attention memory를 측정했는가?
- [ ] Deformable attention, top-k, gather의 target delegate 지원을 확인했는가?
- [ ] Box threshold와 text threshold를 따로 기록했는가?
- [ ] AP뿐 아니라 false positives/image와 calibration을 평가했는가?
- [ ] p50/p95 latency, peak memory, 전력, 온도, throttling을 기록했는가?
- [ ] Grounding DINO를 teacher로 쓸지 on-device runtime으로 쓸지 역할을 명확히 했는가?

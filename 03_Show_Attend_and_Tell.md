# 03. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\03_Show_Attend_and_Tell.pdf`
- 제목: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
- 저자: Kelvin Xu et al.
- 발표: ICML 2015
- 주제: 이미지 captioning에서 spatial visual attention의 이론화
- 핵심 키워드: visual attention, image captioning, soft attention, hard attention, doubly stochastic regularization

## 한눈에 보는 요약

이 논문은 attention을 machine translation 밖으로 확장해 image captioning에 적용한다.

CNN feature map의 각 spatial location을 annotation으로 보고, LSTM decoder가 단어를 생성할 때마다 이미지의 어느 위치를 볼지 결정한다.

Soft attention은 differentiable weighted sum이고, hard attention은 위치를 sampling하는 stochastic attention이다.

## 연구 배경과 문제의식

이미지를 하나의 global vector로 압축해 caption decoder에 넣으면, 단어별로 어떤 이미지 영역을 참조했는지 표현하기 어렵다.

이 논문은 CNN feature map의 각 spatial location을 annotation으로 보고, LSTM decoder가 단어를 생성할 때마다 다른 image region을 보게 만든다.

중요한 이론적 구분은 soft attention과 hard attention이다. 하나는 weighted average라 미분 가능하고, 다른 하나는 stochastic location selection이라 sampling 기반 학습이 필요하다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `이미지 captioning에서 spatial visual attention의 이론화`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

기본 CNN-caption 모델과 비교하면 차이는 이미지 표현을 하나의 vector로 압축하느냐, spatial memory로 남겨두느냐에 있다.

```text
global image vector:
image -> CNN -> v_image -> LSTM decoder

visual attention:
image -> CNN -> a_1 ... a_L
decoder state h_t queries image regions
z_t = sum_i alpha_ti a_i
```

이 차이 때문에 attention 모델은 단어별로 다른 이미지 영역을 볼 수 있다.

## 핵심 개념 상세 해설

### Soft attention과 hard attention

Soft attention은 모든 image location을 확률 가중합한다. 그래서 context `z_t`가 미분 가능하고 일반적인 backpropagation으로 학습된다.

Hard attention은 하나의 위치를 sampling한다. 실제 시선 이동처럼 해석하기 쉽지만, sampling은 미분 불가능하므로 variational lower bound나 REINFORCE류 estimator가 필요하다.

```text
soft: z_t = sum_i alpha_ti a_i
hard: s_t ~ Categorical(alpha_t), z_t = a_{s_t}
```

### Doubly stochastic regularization

Caption 전체를 생성하는 동안 특정 image region만 계속 보는 것을 막기 위해, 각 region이 어느 정도 attention을 받도록 regularization을 둔다.

직관적으로는 `모든 중요한 영역을 한 번쯤 훑어보라`는 제약이다. 이 항은 visual attention이 단어별 glimpse로만 흩어지지 않고 이미지 전체 coverage를 갖게 한다.

## Tensor shape와 자료구조

CNN feature map을 `L`개의 spatial annotation으로 펼친다고 하자. 예를 들어 `14 x 14` feature map이면 `L = 196`개의 image region memory가 생긴다.

```text
image feature map: [H, W, C]
flattened annotations: a_1, ..., a_L
a_i shape: [C]
decoder hidden h_t: [d]
attention scores e_t: [L]
visual context z_t: [C]
```

Caption decoder 입장에서는 source sentence가 아니라 image region sequence를 읽는 셈이다. 그래서 수식은 NMT attention과 거의 같고, memory의 의미만 달라진다.

## 수식과 계산 전개

### 위치별 score

```text
e_ti = f_att(a_i, h_{t-1})
alpha_ti = softmax_i(e_ti)
```

Decoder hidden state가 현재 단어를 만들기 위해 image region과 얼마나 관련되는지 계산한다.

### Soft context

```text
z_t = sum_i alpha_ti a_i
```

모든 region feature의 weighted sum이다. NMT attention의 source annotation을 image annotation으로 바꾼 것과 같다.

### Hard attention

```text
s_t ~ Categorical(alpha_t)
z_t = a_{s_t}
```

위치를 하나 선택한다. 학습에는 stochastic gradient estimator가 필요하다.

## 전체 알고리즘 흐름

1. 이미지를 CNN에 넣어 spatial feature map을 만든다.
2. Feature map을 위치별 annotation sequence로 펼친다.
3. LSTM decoder가 이전 단어와 hidden state를 사용해 attention score를 만든다.
4. Image location 방향 softmax로 visual attention distribution을 만든다.
5. Weighted sum으로 visual context를 만들고 다음 단어 분포를 예측한다.

## 작은 예시로 보는 직관

이미지 caption에서 `a dog chasing a ball`을 생성한다고 하자. `dog`를 만들 때는 동물 영역, `ball`을 만들 때는 공 영역, `chasing`을 만들 때는 두 객체 사이의 관계 영역이 중요할 수 있다.

```text
t = dog:  alpha high on dog region
t = ball: alpha high on ball region
t = chasing: alpha spread over dog and ball regions
```

Soft attention은 이런 영역 선택을 확률분포로 표현한다.

## 계산 복잡도와 병목

- Hard attention은 variance가 큰 gradient estimator를 사용해야 해서 학습 안정성이 낮다.
- 이 논문에서 말하는 효율 또는 성능 개선이 어느 병목을 줄이는지 분리해서 봐야 한다.
- 수식상 복잡도, 실제 메모리 사용량, GPU 실행 효율, 학습 안정성, 추론 latency는 서로 다른 축이다.

예를 들어 같은 `더 효율적이다`라는 주장도 Linformer에서는 low-rank 근사, FlashAttention에서는 IO 절감, PagedAttention에서는 KV cache memory management, ViT에서는 대규모 pretraining을 통한 inductive bias 보완을 뜻한다.

## 구현할 때 확인할 부분

- 수식에서 normalization이 어느 축에 적용되는지 확인한다.
- Mask, position index, cache index, spatial flatten order처럼 off-by-one 오류가 나기 쉬운 부분을 별도로 검증한다.
- 논문이 exact 계산을 유지하는지, 근사 또는 sparse 제한을 쓰는지 구분한다.
- 학습 시 이득과 추론 시 이득이 같은지 분리해서 본다.

## 자주 헷갈리는 지점

- Attention weight가 항상 사람이 해석하는 원인 설명과 일치한다고 보면 안 된다. 모델 내부의 정보 routing 가중치로 보는 편이 안전하다.
- Score에 추가되는 bias나 position term은 value에 직접 더해지는 정보와 역할이 다르다. Score는 무엇을 볼지, value는 무엇을 가져올지를 정한다.
- Soft alignment는 hard word alignment와 다르다. 여러 source 위치를 동시에 볼 수 있고, 언어학적 정렬과 완전히 같지는 않다.

## 관련 방법과 비교

| 관점 | 설명 |
| --- | --- |
| Query | 정보를 찾는 현재 representation |
| Key | 검색 대상의 index representation |
| Value | 실제로 가져올 content representation |

## 실험 결과를 읽는 관점

실험은 제안한 이론적 병목이 실제 성능 또는 효율 차이로 이어지는지를 확인하는 근거로 읽으면 된다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- Hard attention은 gradient variance가 크다.
- Spatial resolution은 CNN feature map 해상도에 제한된다.
- Attention map이 사람의 시선 또는 진짜 설명이라고 보기는 어렵다.

## 후속 논문과의 연결점

이후 visual attention은 non-local networks, attention augmented convolution, Vision Transformer, DETR로 이어진다.

## 개인 학습/연구 메모

이 논문은 `단어 위치` attention을 `이미지 위치` attention으로 바꾸면 같은 수식이 다른 modality에서도 작동한다는 점을 보여주는 좋은 예다.


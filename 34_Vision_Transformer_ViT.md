# 34. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\34_Vision_Transformer_ViT.pdf`
- 제목: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- 저자: Alexey Dosovitskiy et al.
- 발표: ICLR 2021
- 주제: Vision Transformer의 patch sequence modeling
- 핵심 키워드: Vision Transformer, image patches, class token, Transformer encoder

## 한눈에 보는 요약

ViT는 이미지를 fixed-size patch sequence로 바꾸고, 표준 Transformer encoder를 적용해 image classification을 수행한다.

CNN의 강한 local inductive bias 없이도 대규모 pretraining data가 있으면 Transformer가 vision에서 강력하게 작동함을 보였다.

이미지 patch를 NLP의 token처럼 다루는 관점이 핵심이다.

## 연구 배경과 문제의식

기존 vision model은 convolution을 기본 연산으로 사용했다. Convolution은 local receptive field와 weight sharing을 통해 이미지 구조에 강한 bias를 준다.

이미지를 patch token sequence로 바꾸고 Transformer encoder로 image classification을 수행한다.

Transformer는 이러한 bias가 약하지만, 모든 patch 사이 global interaction을 직접 모델링할 수 있다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `Vision Transformer의 patch sequence modeling`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

ViT가 CNN과 근본적으로 다른 점은 이미지를 처음부터 patch sequence로 본다는 것이다. CNN은 작은 kernel을 sliding하면서 local feature를 쌓고, ViT는 patch token들 사이의 전역 상호작용을 Transformer encoder에 맡긴다.

```text
CNN:
image -> local convolution -> hierarchical feature map -> classifier

ViT:
image -> fixed-size patches -> linear patch embeddings
     -> Transformer encoder -> class token classifier
```

따라서 ViT는 convolution의 translation equivariance와 locality bias를 상당 부분 내려놓는다. 대신 충분한 pretraining data가 있으면 patch들 사이의 관계를 더 유연하게 학습할 수 있다.

## 핵심 개념 상세 해설

### Vision attention의 핵심 수식

ViT는 이미지를 patch sequence로 바꾼다. Patch 하나가 NLP의 token에 해당한다.

```text
image -> patches x_p^1, ..., x_p^N
z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos
z_l = TransformerEncoder(z_{l-1})
classification = MLP(z_L^class)
```

Class token은 전체 image 정보를 모으는 learnable summary token처럼 동작한다.

## Tensor shape와 자료구조

Vision attention은 입력을 어떤 sequence로 보느냐에 따라 shape가 달라진다. CNN feature map은 `[B, C, H, W]`이고, ViT/DETR에서는 이를 token sequence로 펼친다.

```text
CNN feature: [B, C, H, W]
spatial tokens: [B, H*W, C]
image patches: [B, N, P*P*C]
patch embeddings: [B, N, D]
object queries: [B, num_queries, D]
```

SE/CBAM은 token-token attention이 아니라 channel/spatial gate를 만든다. ViT/DETR은 patch 또는 object query를 Transformer token처럼 다룬다. 따라서 같은 `attention`이라는 말이지만 계산 대상이 상당히 다르다.

## 수식과 계산 전개

### Patchification

```text
이미지 `x`를 `N = HW / P^2`개의 patch `x_p^1, ..., x_p^N`으로 나눈다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Patch embedding

```text
입력 token은 `z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos`다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Class token

```text
각 layer는 `z'_l = MSA(LN(z_{l-1})) + z_{l-1}`를 계산한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Transformer encoder block

```text
이후 `z_l = MLP(LN(z'_l)) + z'_l`로 FFN을 적용한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Classification head

```text
분류는 `y = MLP(LN(z_L^0))`로 class token 위치의 output을 사용한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

## 전체 알고리즘 흐름

1. 이미지 또는 CNN feature를 patch, spatial token, channel descriptor, object query 입력으로 바꾼다.
2. Attention 또는 gate score를 계산한다.
3. Feature를 재가중하거나 token representation을 갱신한다.
4. 분류, captioning, detection 등 task head로 output을 만든다.

## 작은 예시로 보는 직관

224x224 이미지를 16x16 patch로 나누면 한 축에 14개 patch가 생기고 전체 patch 수는 196개다. 여기에 class token 1개를 더해 Transformer에는 197개 token이 들어간다.

```text
image size: 224 x 224
patch size: 16 x 16
num patches: (224 / 16) * (224 / 16) = 196
sequence length with class token: 197
```

Patch 크기를 32로 키우면 sequence는 짧아지지만 세밀한 공간 정보가 줄어든다. Patch 크기를 8로 줄이면 세밀해지지만 attention 비용이 크게 증가한다.

## 계산 복잡도와 병목

- 데이터가 작을 때는 CNN보다 inductive bias가 약해 성능이 떨어질 수 있다.
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
- Vision attention이 항상 CNN보다 낫다는 뜻은 아니다. 데이터 크기와 local inductive bias가 큰 영향을 준다.
- Channel attention, spatial attention, token self-attention은 모두 attention이라고 불리지만 계산 대상이 다르다.

## 관련 방법과 비교

| 모델 | 기본 단위 | Inductive bias | 장점 | 약점 |
| --- | --- | --- | --- | --- |
| CNN | pixel/local patch convolution | locality, translation equivariance | 적은 데이터에서도 강함 | 장거리 관계는 깊은 layer 필요 |
| ViT | fixed-size image patch token | 상대적으로 약함 | 전역 patch 관계를 유연하게 학습 | 대규모 pretraining 필요 |
| Hybrid ViT | CNN feature + Transformer | 중간 | local feature와 global modeling 결합 | 구조가 복잡 |

## 실험 결과를 읽는 관점

Vision 논문에서는 attention이 CNN 대비 어떤 inductive bias를 잃고 어떤 global modeling 능력을 얻는지 봐야 한다.

ImageNet classification, detection, segmentation, video understanding 결과를 볼 때는 input resolution, pretraining data scale, backbone 차이를 함께 확인해야 공정한 비교가 된다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- 데이터가 작을 때는 CNN보다 inductive bias가 약해 성능이 떨어질 수 있다.
- Patch 크기가 크면 세밀한 local detail을 잃고, 작으면 sequence length가 증가해 attention 비용이 커진다.
- 고해상도 dense prediction에는 hierarchical 구조나 local attention 변형이 필요하다.

## 후속 논문과의 연결점

Swin Transformer, DeiT, MAE, DINO, DETR 계열 등 vision Transformer 연구의 출발점이다.

## 개인 학습/연구 메모

ViT는 `이미지를 patch token 문장으로 바꾼 뒤 Transformer encoder를 그대로 쓴다`는 관점이 핵심이다.


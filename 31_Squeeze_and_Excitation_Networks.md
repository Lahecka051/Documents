# 31. Squeeze-and-Excitation Networks

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\31_Squeeze_and_Excitation_Networks.pdf`
- 제목: Squeeze-and-Excitation Networks
- 저자: Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
- 발표: CVPR 2018
- 주제: Squeeze-and-Excitation의 channel recalibration
- 핵심 키워드: SE block, channel attention, squeeze, excitation

## 한눈에 보는 요약

Squeeze-and-Excitation(SE) block은 CNN feature의 channel 중요도를 동적으로 재가중한다.

Spatial dimension을 global average pooling으로 squeeze해 channel descriptor를 만들고, 작은 MLP로 channel gate를 계산한다.

Attention을 token pair가 아니라 channel selection 관점에서 적용한 대표적인 구조다.

## 연구 배경과 문제의식

CNN convolution은 local spatial pattern을 잘 잡지만, 각 channel이 현재 입력에서 얼마나 중요한지는 명시적으로 조절하지 않는다.

Global context로 channel별 gate를 계산해 CNN feature channel 중요도를 동적으로 조절한다.

Feature channel은 서로 다른 semantic detector처럼 볼 수 있다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `Squeeze-and-Excitation의 channel recalibration`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

CNN과 vision attention의 차이는 feature aggregation 범위다.

```text
convolution:
output at position i uses local neighborhood around i

visual attention:
output at position/query i may use distant positions,
channels, patches, or object-level memory slots
```

다만 vision에서는 local structure가 강하므로 attention이 항상 convolution을 대체해야 하는 것은 아니다. 많은 논문은 둘을 결합하거나, 충분한 data scale에서만 순수 Transformer를 사용한다.

## 핵심 개념 상세 해설

### Vision attention의 핵심 수식

SE block은 spatial 정보를 global average pooling으로 압축한 뒤 channel gate를 만든다.

```text
z_c = 1/(H W) sum_i sum_j U_c(i,j)
s = sigmoid(W_2 ReLU(W_1 z))
U'_c = s_c U_c
```

이는 token-token attention이 아니라 channel-wise reweighting이다.

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

### Squeeze operation

```text
입력 feature `U`의 shape가 `[C, H, W]`라고 하자.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Excitation MLP

```text
Squeeze는 `z_c = (1 / HW) sum_i sum_j U_c(i,j)`로 channel별 global average pooling을 한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Channel gate

```text
Excitation은 `s = sigmoid(W_2 ReLU(W_1 z))`로 channel gate를 만든다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Feature scaling

```text
Scale은 `tilde_U_c = s_c U_c`로 각 channel을 재가중한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Reduction ratio

```text
Reduction ratio `r`은 MLP hidden dimension을 `C/r`로 줄여 parameter를 절약한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

## 전체 알고리즘 흐름

1. 이미지 또는 CNN feature를 patch, spatial token, channel descriptor, object query 입력으로 바꾼다.
2. Attention 또는 gate score를 계산한다.
3. Feature를 재가중하거나 token representation을 갱신한다.
4. 분류, captioning, detection 등 task head로 output을 만든다.

## 작은 예시로 보는 직관

이미지에서 강아지와 공이 멀리 떨어져 있어도 `chasing`이라는 관계를 이해하려면 두 영역이 상호작용해야 한다. Convolution만 쓰면 여러 layer를 거쳐야 하지만 attention은 두 위치를 직접 연결할 수 있다.

```text
dog patch query -> attends to ball patch key
object query -> attends to dog region and ball region
channel gate -> emphasizes object-related channels
```

Vision attention은 이처럼 공간, channel, object slot 중 어떤 축에서 상호작용을 만들지에 따라 성격이 달라진다.

## 계산 복잡도와 병목

- Spatial attention은 직접 다루지 않는다. channel별로만 같은 gain을 전체 위치에 곱한다.
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

| 방법 | Attention 대상 | 역할 |
| --- | --- | --- |
| SE | channel | feature channel 재가중 |
| CBAM | channel + spatial | 무엇과 어디를 순차 강조 |
| Non-local | spatial/space-time position | global relation modeling |
| ViT | image patch token | Transformer encoder classification |
| DETR | object query와 image memory | set prediction detection |

## 실험 결과를 읽는 관점

Vision 논문에서는 attention이 CNN 대비 어떤 inductive bias를 잃고 어떤 global modeling 능력을 얻는지 봐야 한다.

ImageNet classification, detection, segmentation, video understanding 결과를 볼 때는 input resolution, pretraining data scale, backbone 차이를 함께 확인해야 공정한 비교가 된다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- Spatial attention은 직접 다루지 않는다. channel별로만 같은 gain을 전체 위치에 곱한다.
- Global average pooling은 spatial distribution 정보를 많이 잃는다.
- 매우 작은 객체나 위치별 중요도가 큰 task에서는 공간 attention이 추가로 필요할 수 있다.

## 후속 논문과의 연결점

CBAM은 SE의 channel attention에 spatial attention을 추가해 channel과 공간을 순차적으로 재보정한다.

## 개인 학습/연구 메모

SE block은 `global context로 channel별 볼륨 노브를 조절한다`고 생각하면 된다.


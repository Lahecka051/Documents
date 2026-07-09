# 02. Effective Approaches to Attention-based Neural Machine Translation

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\02_Luong_Effective_Approaches_to_Attention_based_NMT.pdf`
- 제목: Effective Approaches to Attention-based Neural Machine Translation
- 저자: Minh-Thang Luong, Hieu Pham, Christopher D. Manning
- 발표: EMNLP 2015
- 주제: Bahdanau attention 이후 NMT attention 설계 공간의 체계화
- 핵심 키워드: global attention, local attention, dot/general/concat score, input feeding

## 한눈에 보는 요약

이 논문은 Bahdanau attention 이후의 attention-based NMT를 더 실용적이고 체계적인 형태로 정리한다.

Global attention은 매 target step마다 source 전체를 보고, local attention은 예측된 중심 위치 주변 window만 본다.

Dot, general, concat scoring과 input-feeding을 제안해 decoder가 이전 attention 결정을 다음 step에 반영하도록 한다.

## 연구 배경과 문제의식

Bahdanau attention은 효과적이었지만, attention score를 어떻게 계산할지, source 전체를 볼지 일부만 볼지, attention 결과를 decoder에 어떻게 다시 넣을지에 대한 설계 선택지는 더 정리될 필요가 있었다.

Luong et al.은 NMT attention을 global attention과 local attention으로 나누고, scoring function을 dot, general, concat으로 체계화한다.

이 논문의 이론적 의미는 attention을 하나의 단일 구조가 아니라 계산 범위, score 함수, decoder 결합 방식의 조합으로 분석했다는 점이다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `Bahdanau attention 이후 NMT attention 설계 공간의 체계화`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

기본 encoder-decoder와 비교하면 차이는 context vector의 위치에 있다.

```text
without attention:
encoder(x_1 ... x_S) -> single context c
p(y_i | y_<i, x) = decoder(y_<i, c)

with attention:
encoder(x_1 ... x_S) -> h_1 ... h_S
c_i = Attention(decoder state, h_1 ... h_S)
p(y_i | y_<i, x) = decoder(y_<i, c_i)
```

즉, attention은 decoder의 입력 정보원을 고정 vector 하나에서 step별 memory lookup으로 바꾼다.

## 핵심 개념 상세 해설

### Global attention과 local attention

Global attention은 모든 source 위치를 매번 본다. 이는 정보 손실이 적지만 target step마다 `S`개 source state와 score를 계산해야 한다. Local attention은 target step에 대응하는 source 중심 위치를 잡고 주변 window만 본다.

```text
global:
c_t = sum_s alpha_t(s) h_s, s = 1..S

local:
p_t = predicted aligned position
c_t = sum_{s in [p_t-D, p_t+D]} alpha_t(s) h_s
```

Local attention은 기계번역 alignment가 대체로 단조적이라는 bias를 쓴다. 이 bias가 맞으면 효율적이고, 틀리면 필요한 source 위치를 놓칠 수 있다.

### Input-feeding

Input-feeding은 이전 step의 attentional hidden state를 다음 step의 decoder input에 넣는다. 이렇게 하면 decoder는 이전에 어디를 봤는지 간접적으로 기억한다.

이는 coverage에 가까운 효과를 준다. Source의 같은 영역을 반복해서 보거나 중요한 영역을 빼먹는 문제를 줄이는 데 도움이 된다.

## Tensor shape와 자료구조

Source 길이를 `S`, target 길이를 `T`, encoder hidden dimension을 `d_h`라고 하자. Encoder는 source 위치마다 annotation `h_j`를 만든다.

```text
source tokens: x_1, ..., x_S
encoder annotations: h_1, ..., h_S
h_j shape: [d_h]
decoder state s_i shape: [d_s]
alignment scores e_i: [S]
attention weights alpha_i: [S]
context c_i: [d_h]
```

중요한 점은 `c_i`가 target 위치 `i`마다 새로 계산된다는 것이다. 이전 fixed-vector encoder-decoder에서는 source 전체가 하나의 `c`였지만, attention 기반 모델에서는 `c_1, c_2, ..., c_T`가 모두 다를 수 있다.

## 수식과 계산 전개

### Score 함수 세 가지

```text
dot:     score(h_t, h_s) = h_t^T h_s
general: score(h_t, h_s) = h_t^T W_a h_s
concat:  score(h_t, h_s) = v_a^T tanh(W_a [h_t; h_s])
```

Dot은 빠르고 단순하다. General은 bilinear matrix를 학습한다. Concat은 additive attention과 유사한 비선형 scoring이다.

### Attentional hidden state

```text
a_t(s) = softmax(score(h_t, h_s))
c_t = sum_s a_t(s) h_s
tilde_h_t = tanh(W_c [c_t; h_t])
```

`tilde_h_t`는 source context와 decoder state를 합친 표현이며, output softmax와 다음 step input-feeding에 사용된다.

### Local-p 중심 예측

```text
p_t = S * sigmoid(v_p^T tanh(W_p h_t))
```

`p_t`는 source 길이 `S` 안의 실수 위치다. 그 주변 window에 Gaussian bias를 곱해 local attention을 만든다.

## 전체 알고리즘 흐름

1. Source token을 embedding한 뒤 encoder가 위치별 hidden annotation을 만든다.
2. Decoder가 이전 target token과 이전 state로 현재 decoder state를 계산한다.
3. 현재 decoder state를 query로 삼아 source annotation들과 score를 계산한다.
4. Source 위치 방향 softmax로 attention weight를 만든다.
5. Source annotation weighted sum으로 context vector를 만든다.
6. Context와 decoder state를 결합해 다음 target token 확률을 만든다.

## 작은 예시로 보는 직관

예를 들어 source가 `I love machine learning`이고 target에서 `apprentissage`를 생성한다고 하자. Decoder는 이 단어를 만들 때 `machine`과 `learning` 위치에 높은 weight를 줄 수 있다.

```text
alpha_t = [0.02, 0.05, 0.41, 0.52]
c_t = 0.02 h_I + 0.05 h_love + 0.41 h_machine + 0.52 h_learning
```

이것이 hard alignment와 다른 점은 하나의 target token이 여러 source token을 동시에 참고할 수 있다는 것이다.

## 계산 복잡도와 병목

- Local attention은 alignment 중심 예측이 틀리면 필요한 source 정보를 놓칠 수 있다.
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

| 모델 | Memory 형태 | Attention 범위 | 핵심 차이 |
| --- | --- | --- | --- |
| 기본 encoder-decoder | 단일 context vector | 없음 | source 전체를 하나로 압축 |
| Bahdanau | source annotation sequence | source 전체 | additive soft alignment |
| Luong | encoder hidden sequence | global 또는 local | dot/general/concat score와 input-feeding |
| Transformer cross-attention | encoder output sequence | source 전체 | Q/K/V multi-head attention |

## 실험 결과를 읽는 관점

실험은 attention이 번역 품질을 높이는지뿐 아니라 긴 문장에서 fixed-vector bottleneck을 완화하는지를 확인하는 방향으로 읽어야 한다.

BLEU 점수보다 더 중요한 관찰은 문장 길이가 길어질수록 attention 없는 encoder-decoder가 급격히 약해지고, attention 기반 모델은 source를 다시 조회할 수 있어 성능 저하가 완화된다는 점이다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- Local attention은 중심 위치 예측이 틀리면 중요한 source 정보를 놓칠 수 있다.
- RNN 기반이라 학습 병렬화 한계는 여전하다.
- Input-feeding은 alignment history를 간접적으로 전달하지만 명시적 coverage constraint는 아니다.

## 후속 논문과의 연결점

Transformer의 encoder-decoder attention은 Luong식 dot-product attention을 scale하고 multi-head화한 일반화로 볼 수 있다.

## 개인 학습/연구 메모

Bahdanau와 비교할 때 `attention을 언제 계산하는가`, `score 함수가 무엇인가`, `source 전체를 보는가 일부만 보는가`를 기준으로 정리하면 좋다.


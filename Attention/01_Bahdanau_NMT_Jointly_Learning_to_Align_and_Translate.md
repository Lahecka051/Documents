# 01. Neural Machine Translation by Jointly Learning to Align and Translate

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\01_Bahdanau_NMT_Jointly_Learning_to_Align_and_Translate.pdf`
- 제목: Neural Machine Translation by Jointly Learning to Align and Translate
- 저자: Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio
- 발표: ICLR 2015
- 주제: 고정 길이 encoder-decoder 병목을 soft alignment로 푸는 NMT 이론
- 핵심 키워드: RNN encoder-decoder, additive attention, soft alignment, context vector

## 한눈에 보는 요약

이 논문은 고정 길이 context vector 하나에 source sentence 전체를 압축하던 초기 encoder-decoder NMT의 병목을 attention으로 완화한다.

Decoder는 target token을 생성할 때마다 source annotation 전체를 다시 조회하고, alignment score를 softmax로 정규화해 위치별 context vector를 만든다.

핵심은 alignment를 외부 전처리나 별도 잠재 변수 추정으로 두지 않고, translation objective 안에서 end-to-end로 학습되는 differentiable module로 만든 점이다.

## 연구 배경과 문제의식

이 논문의 출발점은 초기 RNN encoder-decoder가 source sentence 전체를 하나의 고정 길이 벡터로 압축한다는 데 있다. 입력 길이는 가변적이지만 표현 용량은 고정되어 있으므로, 긴 문장에서는 세부 의미와 어순 정보가 압축 과정에서 손실되기 쉽다.

기계번역은 target 단어마다 필요한 source 단어가 다르다. 예를 들어 target의 관사를 만들 때는 source의 명사 성질을 보고, 동사를 만들 때는 source의 주어와 동사 주변을 본다. 따라서 decoder가 모든 step에서 같은 context vector만 사용하는 것은 구조적으로 부자연스럽다.

Bahdanau attention은 source를 하나의 vector로 접지 않고 위치별 annotation sequence로 남겨둔다. Decoder는 target step마다 이 annotation sequence에서 필요한 정보를 골라 읽는다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `고정 길이 encoder-decoder 병목을 soft alignment로 푸는 NMT 이론`이다.

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

### Fixed-length bottleneck

Bahdanau 이전의 encoder-decoder는 입력 문장 전체를 하나의 vector에 압축했다. 이 vector는 decoder의 모든 time step에서 같은 정보원으로 쓰인다. 문제는 target token마다 필요한 source 정보가 다르다는 점이다.

Attention은 source를 `h_1, ..., h_S`라는 memory sequence로 남겨둔다. Decoder는 `y_i`를 만들 때 `s_{i-1}`로 source memory를 검색하고, `c_i`를 새로 만든다. 이때 `c_i`가 고정 vector bottleneck을 푸는 핵심이다.

### Additive alignment score

Alignment model `a(s_{i-1}, h_j)`는 작은 neural network로 볼 수 있다. Dot product처럼 단순 내적만 쓰는 것이 아니라, decoder state와 source annotation을 함께 비선형 변환해 compatibility를 계산한다.

```text
e_ij = v_a^T tanh(W_s s_{i-1} + W_h h_j)
alpha_ij = softmax_j(e_ij)
c_i = sum_j alpha_ij h_j
```

이 구조 때문에 source와 target hidden dimension이 달라도 score를 만들 수 있고, alignment function 자체를 학습할 수 있다.

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

### Alignment score

```text
e_ij = a(s_{i-1}, h_j)
alpha_ij = exp(e_ij) / sum_k exp(e_ik)
```

`a`는 작은 feed-forward network로 볼 수 있다. Decoder의 이전 상태와 source annotation을 함께 넣어 compatibility를 계산한다.

### Context 계산

```text
c_i = sum_j alpha_ij h_j
p(y_i | y_<i, x) = g(y_{i-1}, s_i, c_i)
```

Softmax로 얻은 `alpha_ij`가 source 위치별 가중치가 되고, weighted sum 결과가 decoder의 동적 context가 된다.

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

- Encoder와 decoder가 여전히 RNN이라 sequence 방향 병렬화가 제한된다.
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

- RNN encoder-decoder 구조이므로 sequence 병렬화 문제는 그대로 남는다.
- Target step마다 모든 source 위치를 보므로 source 길이와 target 길이의 곱에 비례하는 score 계산이 필요하다.
- Attention weight는 해석에 도움을 주지만 항상 언어학적 alignment와 일치한다고 단정할 수 없다.

## 후속 논문과의 연결점

Luong attention은 scoring 함수를 dot/general/concat으로 정리하고 global/local attention으로 계산 범위를 조절한다. Transformer는 RNN recurrence를 제거하고 attention 자체를 주된 sequence modeling 연산으로 바꾼다.

## 개인 학습/연구 메모

이 논문을 읽을 때는 `fixed vector bottleneck -> source annotation sequence -> alignment score -> softmax -> weighted sum` 흐름을 먼저 잡으면 된다.


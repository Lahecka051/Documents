# 06. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\06_Transformer_XL.pdf`
- 제목: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
- 저자: Zihang Dai et al.
- 발표: ACL 2019
- 주제: Transformer language model의 fixed context 문제를 segment memory로 해결
- 핵심 키워드: segment-level recurrence, relative positional encoding, long context language modeling

## 한눈에 보는 요약

Transformer-XL은 Transformer language model이 fixed-length segment 안에서만 context를 보는 문제를 segment-level recurrence로 해결한다.

이전 segment의 hidden states를 memory로 cache하고, 현재 segment가 이 memory를 attention key/value로 참조한다.

Memory 재사용과 충돌하지 않도록 absolute position 대신 relative positional encoding을 사용한다.

## 연구 배경과 문제의식

Vanilla Transformer language model은 긴 문서를 fixed-length segment로 잘라 학습한다. Segment 경계 밖의 token은 보지 못하므로 context fragmentation이 생긴다.

더 긴 context를 보려면 전체 sequence를 다시 넣어야 하는데, attention 비용이 커지고 학습 길이와 평가 길이의 불일치도 생긴다.

Transformer-XL은 이전 segment의 hidden states를 memory로 재사용하고, relative positional encoding으로 segment 간 위치를 일관되게 표현한다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `Transformer language model의 fixed context 문제를 segment memory로 해결`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

기본 Transformer는 위치 정보를 입력 embedding에 더한 뒤 attention을 한다. 위치 표현 논문들은 이 위치 정보가 들어가는 위치를 바꾼다.

```text
absolute position:
X = token_embedding + position_embedding
S = QK^T

relative/bias/rotary variants:
Q, K = position-aware projection or rotation
S_ij = content_score(i,j) + position_term(i,j)
```

따라서 이 계열을 비교할 때는 `position_term(i,j)`가 table lookup인지, 회전인지, 선형 bias인지, scaling된 position index인지 보면 된다.

## 핵심 개념 상세 해설

### Segment-level recurrence

Transformer-XL은 이전 segment의 hidden states를 memory로 저장하고 다음 segment에서 attention 대상으로 사용한다. 이는 RNN처럼 정보를 다음 시간으로 넘기지만, hidden vector 하나가 아니라 token-level hidden states 전체를 넘긴다는 점이 다르다.

```text
previous memory: M = stop_gradient(previous hidden states)
current hidden: H
attention key/value source: [M; H]
query source: H
```

Stop-gradient를 쓰므로 memory를 통해 무한히 backpropagation하지 않는다. 계산은 제한하면서도 evaluation context를 길게 가져갈 수 있다.

### 왜 relative position이 필요한가

Memory를 붙이면 현재 segment의 token과 이전 segment의 token이 같은 absolute position table을 공유하기 어렵다. Relative position은 `현재 token 기준으로 얼마나 과거인가`를 표현하므로 segment가 바뀌어도 일관된다.

## Tensor shape와 자료구조

위치 표현 논문들은 대개 Transformer의 기본 shape는 유지한다. 입력 hidden state `X`는 `[B, n, d_model]`이고, attention head마다 Q/K/V를 만든다.

```text
X: [B, n, d_model]
Q = X W_Q: [B, h, n, d_k]
K = X W_K: [B, h, n, d_k]
V = X W_V: [B, h, n, d_v]
score S: [B, h, n, n]
output O: [B, h, n, d_v]
```

차이는 `S = QK^T / sqrt(d_k)`를 그대로 쓰지 않는다는 점이다. 상대 위치 embedding, relative bias, rotary transformation, linear bias, interpolation scaling 등이 `Q`, `K`, 또는 `S`에 들어간다.

## 수식과 계산 전개

### Memory 결합

```text
tilde_h_tau^{n-1} = [SG(h_{tau-1}^{n-1}); h_tau^{n-1}]
```

현재 segment의 layer 입력 앞에 이전 segment memory를 붙인다. `SG`는 stop-gradient다.

### Attention source

```text
q_tau^n = h_tau^{n-1} W_Q
k_tau^n = tilde_h_tau^{n-1} W_K
v_tau^n = tilde_h_tau^{n-1} W_V
```

Query는 현재 segment 위치에서 나오고, key/value는 memory와 현재 segment를 합친 곳에서 나온다.

### Relative score 분해

```text
score = content-content + content-position + global content bias + global position bias
```

Transformer-XL의 relative attention은 위치 항과 content 항을 분해해 계산한다.

## 전체 알고리즘 흐름

1. Token embedding을 만들고 필요한 경우 absolute/relative position 정보를 준비한다.
2. 각 layer에서 Q/K/V를 projection한다.
3. 논문별 방식에 따라 Q/K를 회전, 위치 bias 추가, relative embedding 결합, position scaling 등을 적용한다.
4. 수정된 score로 softmax attention을 수행한다.
5. Value를 합산하고 residual/FFN을 거쳐 다음 layer로 넘긴다.

## 작은 예시로 보는 직관

문장 `not very good`에서 `good`은 바로 앞의 `very`, 두 칸 앞의 `not`과 관계가 있다. 절대 위치 17, 18, 19보다 중요한 것은 `good` 기준으로 -1, -2 위치라는 상대 관계다.

```text
absolute: position(good) = 19
relative: distance(good, very) = -1
relative: distance(good, not) = -2
```

Relative position, RoPE, ALiBi 계열은 이런 상대 거리 정보를 attention score가 직접 느끼게 만든다.

## 계산 복잡도와 병목

- Memory에는 gradient가 흐르지 않으므로 아주 먼 과거 정보에 대한 end-to-end credit assignment는 제한된다.
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
- 긴 position index를 계산할 수 있다는 것과 모델이 그 길이에서 잘 일반화한다는 것은 다르다.
- RoPE scaling은 attention 비용을 줄이지 않는다. 위치 일반화를 개선할 뿐이다.

## 관련 방법과 비교

| 방법 | 위치 정보가 들어가는 곳 | 긴 길이 일반화 관점 |
| --- | --- | --- |
| Absolute embedding | input embedding | 학습 위치 밖 취약 |
| Relative position | attention score 또는 value | 상대 거리 표현에 강함 |
| RoPE | Q/K rotation | 상대 phase를 dot product에 내장 |
| ALiBi | score bias | position table 없이 거리 penalty |
| PI/YaRN/LongRoPE | RoPE position scaling | 기존 RoPE LLM context 확장 |

## 실험 결과를 읽는 관점

위치 표현 논문의 실험은 보통 두 가지를 본다. 첫째, 같은 길이 또는 학습 길이 안에서 성능이 좋아지는가. 둘째, 학습보다 긴 context에서 extrapolation이 되는가.

특히 RoPE scaling 계열에서는 짧은 context 성능 보존과 긴 context perplexity 개선을 동시에 봐야 한다. 긴 길이만 좋아지고 짧은 길이가 무너지면 실용적인 확장이라고 보기 어렵다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- Memory에는 gradient가 흐르지 않아 아주 먼 과거까지 end-to-end 학습되지는 않는다.
- Memory 길이를 늘리면 attention 비용과 cache 메모리가 증가한다.
- Relative attention 구현이 기본 Transformer보다 복잡하다.

## 후속 논문과의 연결점

Transformer-XL의 긴 context 문제의식은 Longformer, BigBird, RoPE 계열 context extension, KV cache serving 연구로 이어진다.

## 개인 학습/연구 메모

Transformer-XL은 `긴 문서를 한 번에 넣지 않고도 이전 segment 표현을 다시 attention 대상으로 쓰는 방법`으로 이해하면 된다.


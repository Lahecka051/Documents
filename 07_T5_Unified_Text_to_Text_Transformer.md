# 07. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\07_T5_Unified_Text_to_Text_Transformer.pdf`
- 제목: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- 저자: Colin Raffel et al.
- 발표: JMLR 2020
- 주제: NLP task를 text-to-text transfer learning 문제로 통일하는 이론과 실험 체계
- 핵심 키워드: T5, text-to-text, span corruption, relative position bias, transfer learning

## 한눈에 보는 요약

T5는 모든 NLP task를 text-to-text 형식으로 통일하고, Transformer encoder-decoder로 처리한다.

논문의 핵심은 새로운 attention 수식 하나가 아니라, pretraining objective, data, model scale, task format을 체계적으로 비교한 대규모 연구다.

모델 구조 측면에서는 relative position bias를 attention logit에 더하는 단순하고 강력한 방식을 사용한다.

## 연구 배경과 문제의식

BERT, GPT, 번역 모델은 모두 Transformer를 쓰지만 task 형식과 objective가 달랐다. Classification은 label head, QA는 span head, translation은 decoder를 쓰는 식이다.

T5는 모든 NLP task를 `text input -> text output`으로 통일한다. 이는 모델 구조와 loss를 통일하고, transfer learning 비교를 체계화한다.

논문의 핵심은 단일 새로운 수식보다, model architecture, objective, data, scale을 통제해 transfer learning의 설계 원칙을 분석한 데 있다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `NLP task를 text-to-text transfer learning 문제로 통일하는 이론과 실험 체계`이다.

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

### Text-to-text 통일

T5는 classification, translation, summarization, QA를 모두 `input text -> output text`로 만든다. 모델 구조는 task마다 바뀌지 않고, target string만 달라진다.

```text
translate English to German: That is good. -> Das ist gut.
summarize: <document> -> <summary>
sst2 sentence: ... -> positive
```

이 통일성 때문에 pretraining과 fine-tuning의 objective가 같은 autoregressive text generation으로 정리된다.

### Span corruption

단일 token을 맞히는 MLM보다 span corruption은 여러 token짜리 의미 단위를 복원하게 한다. Encoder는 sentinel token이 들어간 손상 문장을 읽고, decoder는 빠진 span들을 순서대로 생성한다.

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

### Text-to-text 조건부 생성

```text
p(y | x) = product_t p(y_t | y_<t, x)
```

모든 task가 같은 conditional generation loss로 표현된다.

### Span corruption 예시

```text
input:  Thank you <extra_id_0> me to your party <extra_id_1> week.
target: <extra_id_0> for inviting <extra_id_1> last <extra_id_2>
```

Sentinel token이 빠진 span의 위치와 순서를 표시한다.

### Relative bias

```text
score_ij = q_i k_j^T / sqrt(d) + b_{bucket(i-j)}
```

거리 bucket별 bias가 attention logit에 더해진다.

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

- 논문은 방대한 ablation이 장점이지만, 개별 task에 대한 최적 구조를 찾는 연구는 아니다.
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

- 대규모 실험 연구라 개별 task 최적 구조를 제안하는 논문은 아니다.
- Encoder-decoder 구조는 decoder-only serving보다 무거울 수 있다.
- 결론이 data scale과 compute budget에 크게 의존한다.

## 후속 논문과의 연결점

T5는 FLAN-T5, UL2, instruction tuning, text-to-text multi-task learning의 기반이 되었다. Relative bias는 여러 encoder-decoder 모델에서 표준 선택지가 되었다.

## 개인 학습/연구 메모

T5를 볼 때는 `모든 task를 같은 입출력 인터페이스와 같은 loss로 푼다`는 통일성이 핵심이다.


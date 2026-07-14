# 62. Visual Instruction Tuning

## 논문 정보

- 저자: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
- 소속: University of Wisconsin-Madison, Microsoft Research, Columbia University
- 발표: NeurIPS 2023
- 공개: arXiv:2304.08485v2, 2023년 12월 11일
- 원본 파일: `62_LLaVA_Visual_Instruction_Tuning.pdf`
- 논문 링크: https://arxiv.org/abs/2304.08485
- 프로젝트: https://llava-vl.github.io

## 한눈에 보는 요약

LLaVA는 CLIP ViT-L/14의 patch feature를 단순한 linear projection으로 Vicuna의 word embedding 공간에 옮기고, GPT-4가 생성한 158K visual instruction data로 LLM을 instruction-tune한 모델이다. BLIP-2처럼 별도의 Q-Former를 두지 않고 모든 visual patch token을 LLM prefix로 직접 넣는 단순한 구조가 특징이다.

핵심 기여는 세 가지다.

1. text-only GPT-4에 COCO caption과 bounding box를 제공해 conversation, detailed description, complex reasoning 형식의 visual instruction data를 자동 생성한다.
2. 1단계에서 frozen vision encoder와 frozen LLM 사이의 linear projector만 학습하고, 2단계에서 vision encoder는 계속 고정한 채 projector와 Vicuna를 instruction-tune한다.
3. GPT-4를 evaluator로 사용하는 LLaVA-Bench를 만들고, ScienceQA에서 pure LLaVA 90.92%, LLaVA와 text-only GPT-4 judge 조합 92.53%를 보고한다.

온디바이스 관점에서 linear projector는 작지만 visual token이 약 256개이고 Vicuna 7B/13B 전체를 실행해야 한다. 논문이 공개한 compressed checkpoint도 25GB다. 따라서 architecture의 단순함과 실제 inference 비용을 구분해야 한다.

## 논문이 해결하려는 문제

CLIP은 text로 class나 retrieval query를 정의할 수 있지만 자유 형식의 multi-turn answer를 생성하지 않는다. BLIP-2와 Flamingo는 image-conditioned generation이 가능하지만, 당시 공개 모델들은 사용자의 구체적인 지시를 따르기보다 일반 caption을 출력하는 경우가 많았다.

LLaVA의 가설은 NLP의 instruction tuning을 image-language 공간으로 확장하면, 기존 LLM의 대화 능력과 visual encoder의 open-world representation을 결합할 수 있다는 것이다. architecture를 복잡하게 만들기보다 high-quality instruction data를 만드는 데 중심을 둔다.

## GPT-assisted visual instruction data generation

### GPT-4는 실제 image를 보지 않는다

데이터 생성 당시 사용하는 GPT-4는 text-only다. COCO image 자체 대신 다음 두 symbolic context를 제공한다.

- captions: 같은 image를 설명하는 여러 문장
- bounding boxes: object label과 normalized coordinate

예를 들어 다섯 caption과 `person: [x1,y1,x2,y2]`, `suitcase: [...]` 같은 box 목록이 image를 대신한다. 몇 개의 manually designed seed examples를 in-context demonstration으로 넣어 GPT-4가 instruction-response를 생성한다.

이 방식은 저렴하지만 중요한 한계가 있다. teacher response는 원 image가 아니라 caption과 detector annotation이 담은 정보만 볼 수 있다. caption에 없는 작은 글자나 box가 놓친 객체는 teacher가 알 수 없고, complex reasoning 답변은 실제 pixel보다 언어 prior에서 만들어질 수 있다.

### 세 가지 response 유형

1. Conversation: object, count, action, location, relative position 등을 묻는 multi-turn dialogue다. image에서 확실히 답할 수 있는 질문을 만들도록 prompt한다.
2. Detailed description: image 전체를 풍부하게 설명하는 single-turn response다.
3. Complex reasoning: scene에서 한 단계 더 나아간 배경 지식, 원인, 결과, 계획을 step-by-step으로 설명한다.

최종 LLaVA-Instruct-158K 구성은 다음과 같다.

| 유형 | Samples |
|---|---:|
| Conversation | 58K |
| Detailed description | 23K |
| Complex reasoning | 77K |
| 합계 | 158K |

약 80K unique images를 사용한다. early experiment에서 ChatGPT보다 GPT-4가 spatial reasoning 등에서 더 높은 데이터 품질을 보였다고 보고한다.

### Data generation의 숨은 supervision

human annotation은 seed example 몇 개뿐이지만 pipeline 전체가 human-free인 것은 아니다. COCO caption과 box 자체가 human 또는 기존 annotation process에서 왔고, seed prompt 설계와 질문 유형도 사람이 정의한다. “GPT-4 generated”는 최종 instruction-response를 자동 확장했다는 의미다.

또한 GPT-4가 generator이면서 뒤의 benchmark judge 역할도 한다. generator의 문체와 선호에 맞춘 모델이 같은 family evaluator에서 유리할 가능성을 고려해야 한다.

## Architecture

LLaVA는 세 module로 구성된다.

```text
image Xv
  -> frozen CLIP ViT-L/14 g
  -> grid visual features Zv [B, Nv, Dv]
  -> trainable linear projector W
  -> visual language tokens Hv [B, Nv, Dllm]

system + visual tokens + user instruction + history
  -> Vicuna 7B 또는 13B decoder
  -> assistant response tokens
```

수식은 매우 단순하다.

$$
Z_v=g(X_v),
\qquad
H_v=Z_vW,
$$

$$
Z_v\in\mathbb{R}^{B\times N_v\times D_v},
\quad
W\in\mathbb{R}^{D_v\times D_{LLM}},
\quad
H_v\in\mathbb{R}^{B\times N_v\times D_{LLM}}.
$$

논문은 CLIP ViT-L/14의 last layer 전후 grid feature를 비교한다. Q-Former, resampler, cross-attention adapter는 없고 patch sequence의 각 token에 같은 linear layer를 적용한다.

### Shape example

표준 224×224 CLIP ViT-L/14를 가정하면 14×14 patch가 $16\times16=256$개이고 visual width는 1,024다. CLS를 제외한 grid feature를 쓰는 일반적인 LLaVA 구성은 다음 shape를 갖는다.

```text
Xv: [1, 3, 224, 224]
Zv: [1, 256, 1024]
W for Vicuna-7B:  [1024, 4096]
W for Vicuna-13B: [1024, 5120]
Hv-7B:  [1, 256, 4096]
Hv-13B: [1, 256, 5120]
```

224 resolution, hidden widths, layer 수는 해당 CLIP/LLaMA backbone configuration에서 가져온 reviewer reconstruction이다. PDF 본문은 $N_v$, $D_v$, $D_{LLM}$의 숫자 table을 별도로 제공하지 않으므로 실제 checkpoint config를 최종 기준으로 삼아야 한다.

### Projector parameter와 payload 계산

linear projector만 보면 매우 작다.

- 7B: $1024\times4096=4.19$M weights, FP16 약 8 MiB
- 13B: $1024\times5120=5.24$M weights, FP16 약 10 MiB

그러나 projected visual token payload는 batch=1 FP16에서 다음과 같다.

- 7B: $256\times4096\times2$ bytes = 2.0 MiB
- 13B: $256\times5120\times2$ bytes = 2.5 MiB

이는 한 지점의 hidden state만 센 값이다. LLM 각 layer의 QKV, residual, attention workspace와 KV cache가 추가된다.

## BLIP-2와 architecture 차이

| 항목 | BLIP-2 | LLaVA |
|---|---|---|
| Vision-to-language bridge | 188M Q-Former + FC | single linear projection |
| LLM에 전달하는 visual tokens | 고정 32 queries | patch tokens 약 256 |
| Visual information selection | learned cross-attention bottleneck | 모든 patch를 동일 projection |
| 2단계 LLM | frozen | trainable |
| 주된 alignment data | image-text pretraining | visual instruction data |

LLaVA는 bridge compute가 작고 구현이 단순하지만 LLM prefill에 더 많은 visual tokens를 보낸다. BLIP-2는 Q-Former compute를 추가하는 대신 visual prefix를 32개로 제한하고 LLM knowledge를 고정한다.

## Multi-turn training sequence

각 image에는 $(X_q^1,X_a^1,\ldots,X_q^T,X_a^T)$ conversation이 있다. 첫 turn은 image와 question 순서를 무작위로 바꾼다.

$$
X^t_{instruct}=
\begin{cases}
[X_q^1,X_v]\ \text{or}\ [X_v,X_q^1], & t=1,\\
X_q^t, & t>1.
\end{cases}
$$

Vicuna-v0 system message를 앞에 두고 `<STOP>`은 `###`로 설정한다.

```text
system message ###
Human: image + question_1 ### Assistant: answer_1 ###
Human: question_2 ### Assistant: answer_2 ###
...
```

loss는 assistant answer와 stop token에만 적용한다. system message와 human instruction은 context로 입력하지만 target loss에서는 mask한다.

$$
\mathcal{L}
=-\sum_{i\in\mathcal{A}}
\log p_\theta(x_i\mid X_v,X_{instruct,<i},X_{a,<i}),
$$

여기서 $\mathcal{A}$는 assistant response position 집합이다.

### Training pseudocode

```python
visual_grid = clip_vision(image)              # frozen, [B, Nv, Dv]
visual_tokens = projector(visual_grid)        # [B, Nv, Dllm]

input_embeds, labels = build_conversation(
    visual_tokens=visual_tokens,
    system_message=system,
    user_turns=questions,
    assistant_turns=answers,
    stop_token="###",
)

 # labels for system/user/visual positions are ignore_index
logits = vicuna(inputs_embeds=input_embeds)
loss = causal_cross_entropy(logits, labels)
```

stage 1에서도 LLM weight는 frozen이지만 projector까지 gradient를 전달하려면 LLM 연산 graph를 통해 input gradient를 계산해야 한다. frozen weight에 optimizer state는 없지만 LLM forward와 일부 activation/recompute 비용이 사라지는 것은 아니다.

## Two-stage visual instruction tuning

### Stage 1: feature alignment

CC3M을 595K image-text pairs로 filtering한다. caption의 noun phrase를 추출하고 frequency 3 미만을 제외하며, 높은 frequency phrase는 최대 100 captions로 제한해 concept coverage와 규모를 균형 있게 만든다.

각 pair는 single-turn instruction으로 바꾼다. 여러 표현의 `Describe the image concisely.` 계열 질문 중 하나를 random sample하고 원래 caption을 answer로 쓴다.

- frozen: CLIP vision encoder, Vicuna
- trainable: linear projector $W$만
- 목적: visual feature를 LLM word embedding과 호환되는 visual tokenizer로 정렬

### Stage 2: end-to-end instruction tuning

vision encoder는 계속 frozen이다. projector와 Vicuna parameter $\phi$를 함께 update한다.

$$
\theta=\{W,\phi\}.
$$

논문은 이를 end-to-end fine-tuning이라 부르지만 image encoder까지 end-to-end로 학습하는 것은 아니다. vision feature에서 response까지의 language side를 공동 최적화한다는 의미다.

두 use case를 별도로 학습한다.

- Multimodal chatbot: LLaVA-Instruct-158K를 사용하며 세 유형을 uniformly sample한다.
- ScienceQA: question/context를 instruction, reasoning+answer를 assistant response로 구성한다.

원 논문의 stage 2는 LoRA가 아니라 full LLM fine-tuning이다. projector/LoRA만 학습하는 소형 VLM recipe는 후속 deployment 실험이지 이 논문의 설정이 아니다.

## Training recipe와 compute

모든 모델은 8×A100에서 학습했다.

### Feature alignment

- CC-595K, 1 epoch
- batch size 128
- learning rate $2\times10^{-3}$
- 4 hours 이내

### Instruction fine-tuning

- LLaVA-Instruct-158K, 3 epochs
- batch size 32
- learning rate $2\times10^{-5}$
- 10 hours 이내

공통으로 Adam, weight decay 0, cosine LR, warm-up ratio 3%를 사용한다. fine-tuning에는 FSDP와 gradient checkpointing을 쓰고 CPU offloading은 사용하지 않는다. BF16과 TF32를 활성화한다. ScienceQA fine-tuning은 4 hours 이내라고 보고한다.

compressed model checkpoint는 25GB로 당시 GitHub LFS 5GB 제한을 초과했다고 Appendix에 명시되어 있다. 이는 projector가 작아도 full LLM checkpoint가 deployment artifact의 대부분임을 직접 보여준다.

## LLaVA-Bench 평가 방식

candidate model은 image와 question을 보고 answer를 생성한다. reference는 text-only GPT-4가 ground-truth caption/description과 box를 보고 만든다. judge 역할의 text-only GPT-4에 question, symbolic visual information, 두 answer를 주고 helpfulness, relevance, accuracy, detail을 1에서 10으로 평가하게 한다. 결과는 reference GPT-4 대비 relative score다.

### LLaVA-Bench COCO

COCO-Val-2014에서 30 images를 뽑고 image당 conversation, detailed description, complex reasoning 질문을 하나씩 만들어 총 90 questions를 사용한다.

| Training data | Conversation | Detail | Complex | Overall |
|---|---:|---:|---:|---:|
| Full 158K | 83.1 | 75.3 | 96.5 | 85.1 |
| Detail + Complex | 81.5 | 73.3 | 90.8 | 81.9 |
| Conv + 5% Detail + 10% Complex | 81.0 | 68.4 | 91.5 | 80.5 |
| Conversation only | 76.5 | 59.8 | 84.9 | 73.8 |
| No instruction tuning | 22.0 | 24.0 | 18.5 | 21.5 |

instruction tuning 유무가 overall 63.6 point 차이를 만들었다. Conversation only에 소량의 detail/reasoning을 더하면 6.7 point, full data를 쓰면 추가 4.6 point가 오른다. reasoning data가 conversation score도 높여 task 유형 간 positive transfer를 보인다.

### LLaVA-Bench In-the-Wild

24 diverse images와 manually curated description, 총 60 questions로 구성한다.

| 모델 | Conversation | Detail | Complex | Overall |
|---|---:|---:|---:|---:|
| OpenFlamingo | 19.3±0.5 | 19.0±0.5 | 19.1±0.7 | 19.1±0.4 |
| BLIP-2 | 54.6±1.4 | 29.1±1.2 | 32.9±0.7 | 38.1±1.0 |
| LLaVA | 57.3±1.9 | 52.5±6.3 | 81.7±1.8 | 67.3±2.0 |

LLaVA가 user instruction에 맞춘 답을 생성하는 능력이 BLIP-2/OpenFlamingo보다 높다는 증거다. 그러나 24 images와 GPT-4 judge라는 작은 protocol이므로 일반적인 VQA accuracy나 human preference와 동일하지 않다.

### GPT-4 judge의 한계

- 같은 GPT-4 family가 training data generator와 evaluator다.
- reference GPT-4는 pixel이 아니라 ground-truth textual description을 본다.
- judge가 visual fact를 직접 확인할 수 없고 제공된 description에 의존한다.
- 90/60 questions는 domain coverage가 작다.
- response verbosity가 detail score에 유리할 수 있다.

논문도 multimodal evaluation의 어려움과 visual hallucination, fine-grained understanding을 추가 평가해야 한다고 인정한다.

## ScienceQA 결과

ScienceQA는 21K multiple-choice questions, train/val/test 12,726/4,241/4,241 samples로 구성된다. LLaVA는 CLIP last layer 전 feature를 사용하고 reasoning을 먼저 생성한 뒤 answer를 선택하도록 12 epochs 학습한다.

| 방법 | Test accuracy |
|---|---:|
| GPT-3.5 + CoT | 75.17 |
| LLaMA-Adapter | 85.19 |
| MM-CoT Large | 91.68 |
| text-only GPT-4, 2-shot | 82.69 |
| LLaVA | 90.92 |
| LLaVA + GPT-4 complement | 90.97 |
| LLaVA + GPT-4 judge | 92.53 |

논문의 headline 92.53%는 LLaVA 단독이 아니다. LLaVA와 text-only GPT-4가 다른 답을 낼 때 GPT-4에게 두 reasoning을 비교해 final answer를 고르게 하는 external model ensemble이다. 순수 모델 결과는 90.92%다.

text-only GPT-4가 image context question까지 일부 고친 이유는 몇몇 문제가 image 없이 common knowledge로 풀리기 때문이다. 이 방식은 정확도를 높이지만 API latency, 비용, network availability가 추가되어 on-device pipeline으로 볼 수 없다.

## ScienceQA design ablation

| 변경 | Accuracy | Best 대비 |
|---|---:|---:|
| Best, feature before last layer | 90.92 | 0 |
| Last layer visual feature | 89.96 | -0.96 |
| Answer first | 89.77 | -1.15 |
| Alignment pretraining 생략 | 85.81 | -5.11 |
| Vicuna 7B | 89.84 | -1.08 |

last layer보다 그 전 layer가 localized detail을 더 보존한다는 가설을 제시한다. reasoning-first는 final accuracy 향상은 제한적이지만 6 epochs에 89.77에 빨리 도달해 convergence를 돕는다. 가장 큰 ablation은 alignment pretraining으로, projector를 바로 ScienceQA에서 처음부터 학습하면 5.11 point 하락한다.

## Visual token과 batch=1 memory

LLaVA의 단순 projector는 256 patch tokens를 줄이지 않는다. standard multi-head attention과 common Vicuna config를 가정하면 visual prefix의 KV cache reviewer estimate는 다음과 같다.

$$
M_{KV,visual}\approx2\times N_v\times D_{LLM}\times L_{layers}\times s.
$$

- Vicuna-7B 가정: $N_v=256$, $D=4096$, 32 layers, FP16 -> 약 128 MiB
- Vicuna-13B 가정: $N_v=256$, $D=5120$, 40 layers, FP16 -> 약 200 MiB

이는 visual prefix만의 KV payload다. text history와 generated tokens가 길어질수록 선형으로 증가하고, weights와 temporary workspace가 별도다. grouped-query attention이 아닌 당시 standard LLaMA MHA 가정이며 논문 측정값은 아니다.

BLIP-2의 32 visual queries와 비교하면 LLaVA는 visual KV token 수가 8배다. 대신 Q-Former cross-attention이 없다. device latency에서는 vision adapter compute와 LLM prefill/KV 비용 중 어느 쪽이 더 큰지 실제 profile해야 한다.

### Prefill attention 예시

system+instruction+history가 128 text tokens라고 가정하면 first layer의 naive score 크기는 다음처럼 달라진다.

- text only: $128^2=16,384$
- LLaVA visual prefix 포함: $(256+128)^2=147,456$
- score elements 약 9배

causal fused kernel에서는 전체 matrix를 저장하지 않을 수 있지만 token 수가 TTFT를 늘리는 본질은 남는다. multi-turn history가 길어지면 visual token 비중은 줄어도 total context와 KV cache가 계속 커진다.

## On-device latency 모델

LLaVA의 TTFT는 다음 경로로 분해해야 한다.

```text
TTFT = image decode/resize
     + CLIP ViT-L/14
     + 256-token linear projection
     + Vicuna prefill over visual + system + history + query
     + first-token sampling
```

이후 tokens/s는 Vicuna decode와 memory bandwidth가 지배한다. projector 4M에서 5M parameter만 측정해 LLaVA가 가볍다고 결론 내리면 안 된다.

같은 image의 multi-turn conversation에서는 projected $H_v$ 또는 visual prefix KV를 cache할 수 있다. 논문 형식도 image를 첫 turn에만 넣고 이후 question은 history 뒤에 붙인다. 다만 cache 정책은 논문에서 mobile optimization으로 평가하지 않았다.

## Weight memory와 quantization

논문은 25GB compressed checkpoint를 보고하지만 mobile quantization을 실험하지 않는다. 단순 weight 하한은 다음처럼 생각할 수 있다.

- 7B LLM W4: 약 3.5GB decimal, 여기에 CLIP, projector, quantization metadata 추가
- 13B LLM W4: 약 6.5GB decimal, 여기에 동일 overhead 추가
- FP16은 각각 약 14GB와 26GB decimal의 LLM weight payload

실제 file과 resident memory는 tokenizer, embedding, scale, packing alignment, graph runtime, KV cache 때문에 더 크다. accuracy와 hallucination이 W4/INT8에서 유지되는지도 이 논문으로 알 수 없다.

## 논문이 보고하지 않은 시스템 metric

- batch=1 vision encoder와 LLM latency
- TTFT와 tokens/s
- peak resident/activation/KV memory
- p50/p95와 cold start
- FP16, INT8, W4A16 accuracy
- mobile CPU/GPU/NPU operator coverage
- 전력, 온도, sustained throttling

8×A100 training time은 제공하지만 inference deployment 결과는 없다. 따라서 LLaVA는 architecture 및 data baseline이지 on-device-ready model 증거가 아니다.

## 실패 사례와 한계

### Bag of patches

냉장고 image에 yogurt와 strawberries가 따로 있는데 strawberry-flavored yogurt가 있느냐는 질문에 yes라고 답했다. local patch concept은 감지했지만 entity와 attribute binding을 잘못 결합한 사례다. patch token을 모두 전달해도 relation을 올바르게 묶는다는 보장은 없다.

### High-resolution detail

yogurt brand, 작은 text, restaurant identity 같은 질문은 표준 CLIP resolution에서 어렵다. resolution을 높이면 patch token 수와 LLM prefill이 크게 늘어난다. 논문은 tiling, token pruning, ROI crop을 다루지 않는다.

### Hallucination과 inherited bias

CLIP, LLaMA, Vicuna의 bias와 knowledge error를 상속한다. image와 무관한 fluent answer를 생성할 수 있으며 의료 같은 critical application에서 위험하다. Appendix는 text input filter와 NSFW image filter를 mitigation으로 제안하지만 factual grounding을 해결하지는 않는다.

### Synthetic instruction의 오류

GPT-4는 caption/box만 보고 answer를 만들기 때문에 missing visual fact를 언어 prior로 보충할 수 있다. student는 이런 hallucination을 supervision으로 학습할 수 있다. 생성 데이터의 human audit rate와 error taxonomy가 보고되지 않는다.

### Evaluation circularity와 작은 benchmark

GPT-4가 data generator, reference answer 생성자, judge로 여러 역할을 한다. LLaVA-Bench 규모도 작다. human blind evaluation, task-specific accuracy, hallucination benchmark가 보완되어야 한다.

### 92.53%의 external dependency

ScienceQA 최고 수치는 text-only GPT-4 ensemble을 요구한다. network가 끊긴 독립 on-device 실행에서는 사용할 수 없으며 latency, cost, privacy 조건도 달라진다.

## 재현 체크리스트

### Data generation

- [ ] GPT prompt, seed examples, model version을 보존했다.
- [ ] captions와 boxes가 어느 annotation에서 왔는지 기록했다.
- [ ] 58K/23K/77K 유형 비율을 확인했다.
- [ ] image가 text-only GPT에 직접 들어가지 않았음을 재현했다.
- [ ] generated answer의 visual grounding을 표본 human audit했다.
- [ ] data leakage와 personal identity example을 검사했다.

### Stage 1

- [ ] CC3M filtering과 595K sample list를 고정했다.
- [ ] CLIP과 Vicuna를 모두 frozen으로 설정했다.
- [ ] projector만 optimizer에 들어갔다.
- [ ] answer/stop token에만 loss를 적용했다.
- [ ] last layer 전후 feature 위치를 명시했다.

### Stage 2

- [ ] CLIP은 frozen 상태를 유지했다.
- [ ] projector와 full Vicuna가 trainable인지 확인했다.
- [ ] first turn image-question 순서를 randomize했다.
- [ ] conversation/detail/reasoning을 uniformly sample했다.
- [ ] system message와 `###` stop convention을 고정했다.
- [ ] multi-turn history truncation policy를 기록했다.

### 평가와 모바일

- [ ] pure LLaVA 90.92와 GPT-4 ensemble 92.53을 분리했다.
- [ ] GPT-4 judge model/version과 반복 variance를 기록했다.
- [ ] visual token 수, prompt length, output length를 함께 기록했다.
- [ ] CLIP/projector/prefill/decode latency를 분리했다.
- [ ] TTFT, tokens/s, p50/p95, peak memory를 측정했다.
- [ ] FP16 vision, INT8 projector, W4A16 LLM을 비교했다.
- [ ] 동일 image multi-turn cache hit와 KV 증가를 측정했다.
- [ ] OCR, small object, attribute binding, hallucination을 평가했다.

## 온디바이스용 확장 방향

원 논문을 그대로 배포하기보다 다음 ablation이 유용하다.

1. CLIP ViT-L/14를 MobileCLIP 또는 작은 efficient encoder로 교체한다.
2. 256 patch tokens를 ROI selection, pooling, token merging으로 128/64/32개로 줄인다.
3. full Vicuna fine-tuning 대신 projector + LoRA만 학습한다.
4. 1B에서 3B LLM에 W4A16을 적용한다.
5. detector의 ROI/mask로 필요한 patch만 LLM에 전달한다.
6. 같은 image의 multi-turn에서는 visual feature와 prefix KV를 cache한다.

visual token 256, 128, 64, 32 조건에서 ScienceQA/VQA, OCR, small-object accuracy, TTFT, peak KV memory를 같이 그리면 LLaVA의 가장 중요한 on-device ablation이 된다.

## 로드맵에서의 위치

LLaVA는 BLIP-2 다음에 읽을 때 차이가 선명하다. BLIP-2가 frozen LLM과 learned query bottleneck으로 modality를 연결한다면, LLaVA는 simple projector와 instruction-tuned LLM으로 data quality의 힘을 보여준다.

후속 TinyLLaVA, MobileVLM V2, FastVLM을 읽을 때 다음 기준을 유지하면 된다.

- vision encoder latency는 얼마나 줄었는가?
- visual token 수는 256에서 얼마나 줄었는가?
- projector가 linear/MLP/resampler 중 무엇인가?
- LLM은 full tune, LoRA, frozen 중 무엇인가?
- TTFT 감소가 vision encoder와 token compression 중 어디에서 왔는가?

로드맵의 구현 목표처럼 vision encoder와 LLM을 frozen하고 projector 또는 LoRA만 학습하는 것은 LLaVA의 data format과 loss를 재사용하면서 training memory를 줄이는 현실적인 변형이다. 다만 original LLaVA 결과와 동일한 recipe라고 표기해서는 안 된다.

## 최종 평가

LLaVA의 가장 큰 기여는 복잡한 connector가 아니라 visual instruction tuning이라는 data-centric recipe를 공개한 것이다. caption과 box를 text-only GPT-4가 이해할 수 있는 symbolic scene으로 바꾸고, 158K instruction data만으로 Vicuna가 image에 대해 사용자 의도에 맞춰 대화하도록 만들었다. projector-only alignment 뒤 full LLM instruction tuning을 하는 2단계 구조도 이후 많은 공개 VLM의 기본형이 되었다.

반면 256 visual tokens와 7B/13B Vicuna는 모바일에 무겁고, 25GB checkpoint, GPT-4 judge 의존, synthetic grounding 오류, high-resolution detail 실패가 남는다. 온디바이스 연구에서는 이 논문을 완성된 배포 모델이 아니라 기준 architecture로 사용하고, `shared/efficient vision encoder + ROI visual token compression + LoRA/W4A16 + prefix cache`를 결합해 TTFT, tokens/s, peak memory, factual accuracy를 함께 최적화해야 한다.

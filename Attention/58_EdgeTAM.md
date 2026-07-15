# 58. EdgeTAM: On-Device Track Anything Model

## 논문 정보

- 제목: **EdgeTAM: On-Device Track Anything Model**
- 저자: Chong Zhou, Chenchen Zhu, Yunyang Xiong, Saksham Suri, Fanyi Xiao, Lemeng Wu, Raghuraman Krishnamoorthi, Bo Dai, Chen Change Loy, Vikas Chandra, Bilge Soran
- 소속: Meta Reality Labs, Nanyang Technological University, Shanghai AI Laboratory
- 공개: arXiv:2501.07256v1, 2025-01-13
- 논문: [arXiv](https://arxiv.org/abs/2501.07256) / [PDF](https://arxiv.org/pdf/2501.07256)
- 원본 파일: `58_EdgeTAM.pdf`
- 기반 구현: [SAM 2](https://github.com/facebookresearch/sam2)
- 핵심 키워드: promptable video segmentation, on-device tracking, memory attention, 2D Spatial Perceiver, memory compression, knowledge distillation

## 한눈에 보는 요약

EdgeTAM은 SAM 2를 모바일에 올릴 때 image encoder만 줄여서는 충분하지 않다는 사실에서 출발한다. SAM 계열에서는 image encoder가 주된 비용이었지만, SAM 2에는 과거 frame memory와 현재 image feature를 결합하는 **memory cross-attention**이 새로 생겼고 이것이 iPhone latency의 또 다른 병목이 된다.

```text
SAM 2
 -> image encoder를 RepViT-M1로 교체
 -> memory-attention block 4개를 2개로 축소
 -> raw 64×64 memory를 512 latent token으로 압축
 -> global + local 2D Spatial Perceiver
 -> image feature와 memory-attention feature distillation
 = EdgeTAM
```

EdgeTAM의 핵심은 frame memory `64×64=4096` token을 global latent 256개와 spatial latent 256개, 합계 512개로 줄이면서 local 2D structure를 보존하는 것이다. Memory cross-attention token 수가 8분의 1이 되며, RepViT-M1 + memory-attention 2-block model이 iPhone 15 Pro Max에서 `2.5 -> 15.7 FPS`로 빨라진다.

최종 model은 DAVIS 2017 `J&F 87.7`, MOSE 70.0, SA-V val 72.3, SA-V test 71.7을 기록하고 iPhone에서 약 16 FPS로 동작한다. SAM 2-B+의 iPhone 0.7 FPS보다는 약 22배 빠르지만 SA-V/DAVIS accuracy는 낮다.

## 왜 image encoder 압축만으로 부족한가

SAM은 image encoder, prompt encoder, mask decoder로 구성되고 mask decoder는 가볍다. 따라서 EdgeSAM, EfficientSAM 같은 연구는 image encoder를 주로 압축했다. SAM 2는 video를 처리하기 위해 다음 module을 추가한다.

- Memory encoder: 과거 frame feature와 predicted mask를 memory로 encoding
- Memory bank: frame-level memories와 object pointer 저장
- Memory attention: 현재 frame feature가 과거 memory를 cross-attention으로 조회

논문 Figure 2에서 image encoder를 HieraB+에서 RepViT-M1까지 줄여도 decoder latency가 남는다. Memory-attention block 수를 줄이면 latency가 거의 선형으로 줄고, block 안에서 cross-attention을 제거할 때 가장 큰 speed-up이 나타난다.

즉 parameter가 적은 module도 token matrix가 크면 latency bottleneck이 될 수 있다. Mobile optimization은 parameter count뿐 아니라 **attention의 query-key length와 backend의 parallelism**을 봐야 한다.

## SAM 2에서 상속한 전체 구조

EdgeTAM은 SAM 2의 meta architecture를 유지한다.

```text
current frame I
 -> image encoder E_img
 -> multi-scale {F4, F8, F16}
 -> F16 + memory bank -> memory attention A -> F_M
 -> prompt embedding P + {F_M,F4,F8} -> mask decoder D -> mask O
 -> {F16,O} -> memory encoder E_mem -> new memory M_{T+1}
 -> FIFO memory bank update
```

Image encoder는 stride 4, 8, 16 feature를 출력한다.

```math
\lbrace F_4,F_8,F_{16}\rbrace=E_{img}(I)
```

현재 low-resolution feature와 과거 `T`개 memory를 결합한다.

```math
F_M=A(F_{16},M_1,M_2,\ldots,M_T)
```

Prompt와 high-resolution skip을 사용해 mask를 decoding한다.

```math
O=D(F_M,F_4,F_8,P)
```

현재 prediction을 다음 frame을 위한 memory로 encoding한다.

```math
M_{T+1}=E_{mem}(F_{16},O)
```

EdgeTAM은 default image encoder로 ImageNet-pretrained RepViT-M1을 사용하고 memory-attention block은 SAM 2의 4개에서 2개로 줄인다. Mask decoder와 prompt interface를 유지하므로 point, box, mask prompt를 image/video 어느 frame에서도 받을 수 있다.

## Raw memory-attention의 tensor shape

논문이 직접 명시한 memory shape는 다음과 같다.

```math
M_t\in\mathbb{R}^{C\times H\times W},
\qquad C=64,\ H=W=64
```

Flatten하면 frame당 `4096` token, token dimension 64다.

```math
M_t^{flat}\in\mathbb{R}^{4096\times64}
```

Default frame-level memory bank 크기는 `T=7`이며 object pointer는 16개다. 논문은 object pointer 비용이 작다고 보고 수식에서 생략한다.

현재 `F16`의 4096 query가 7 frame의 `7×4096=28,672` memory key를 조회한다고 보면 dense cross-attention score shape는 head당 다음과 같다.

```math
S_{raw}\in\mathbb{R}^{4096\times28672}
```

```math
4096\times28672=117{,}440{,}512\text{ elements}
```

Reviewer 계산으로 이를 그대로 materialize하면 FP16 head당 약 `224 MiB`다. 실제 kernel은 tiling/fusion으로 전체 score를 동시에 저장하지 않을 수 있고 multi-head projection 세부도 있으므로 이 값은 peak-memory 보고치가 아니라 **attention matrix 규모를 보여주는 상한적 payload 예시**다.

논문이 제시한 raw complexity는 다음과 같다.

```math
O(TCH^2W^2)
```

Spatial `HW`가 query와 key 양쪽에 들어가므로 high-resolution memory attention이 급격히 비싸진다.

## Global Perceiver

Perceiver는 learnable latent query `Z_g`가 dense memory를 cross-attention으로 요약한다.

```math
Z_g\in\mathbb{R}^{C\times N_g},
\qquad N_g\ll HW
```

```math
Z'_g=CA(Q(Z_g),K(M_t+p),V(M_t+p))
```

```math
G_t=SA(Z'_g)
```

각 latent가 frame 전체를 볼 수 있어 content-dependent global summary를 만든다. Complexity는 다음으로 줄어든다.

```math
O(TCHWN_g)
```

그러나 output latent는 정해진 2D 위치를 갖지 않아 positional embedding을 input에 더해도 explicit spatial grid가 사라진다. Dense mask tracking에서는 object가 어디에 있었는지가 중요해 global-only compression은 accuracy가 크게 떨어진다.

## 2D Spatial Perceiver

EdgeTAM은 local latent `Z_l`마다 non-overlapping spatial patch 하나를 담당시킨다.

```math
M'_t=\mathrm{window\_partition}(M_t)
```

```math
Z'_l=CA(Q(Z_l),K(M'_t),V(M'_t))
```

```math
L'_t=SA(Z'_l)
```

```math
L_t=\mathrm{window\_unpartition}(L'_t)+p'
```

Global latent는 frame 전체에서 자유롭게 움직이며 object-level summary를 만들고, 2D latent는 local patch identity를 유지한다. 두 output을 flattened token dimension으로 concatenate한다.

```math
\widetilde M_t=\mathrm{Concat}(G_t,L_t)
\in\mathbb{R}^{C\times(N_g+N_l)}
```

Default는 `N_g=256`, `N_l=256`, 합계 512이며 Global/2D Perceiver block을 각각 두 번 stack한다. Global과 2D path는 같은 network architecture와 parameter를 공유한다고 논문이 설명한다.

Memory-attention complexity는 다음으로 바뀐다.

```math
O(TCHW(N_g+N_l))
```

압축 비율은 다음과 같다.

```math
\frac{HW}{N_g+N_l}
=\frac{4096}{512}=8
```

논문은 이를 memory-attention 약 8배 speed-up 설계로 설명한다. 또 `HW/(Ng+Nl)`을 대략 `T`와 맞춰 memory-attention 내부 self-attention과 cross-attention 복잡도가 비슷해지도록 한다. 여기서는 8과 memory bank 7이 근접한다.

## Batch 1 memory 계산

Paper shape를 사용한 reviewer 계산이다.

### Frame memory payload

Raw frame memory 한 장:

```math
64\times64\times64=262{,}144\text{ elements}
```

FP16/BF16 약 `0.5 MiB`, 7-frame bank는 약 `3.5 MiB`다.

Compressed frame memory 한 장:

```math
512\times64=32{,}768\text{ elements}
```

FP16/BF16 약 `0.0625 MiB`, 7-frame bank는 약 `0.4375 MiB`다. Frame-level payload만 보면 약 3.06 MiB를 줄인다. Object pointer, positional embedding, allocator overhead는 제외했다.

### Compressed cross-attention score

7 frame의 compressed key length는 `7×512=3584`다.

```math
S_{compressed}\in\mathbb{R}^{4096\times3584}
```

```math
4096\times3584=14{,}680{,}064\text{ elements}
```

FP16 head당 약 `28 MiB`로 raw 224 MiB 예시의 정확히 1/8이다. 다시 말해 실제 runtime peak는 attention kernel 구현에 따라 다르지만 token compression이 bandwidth와 multiply workload를 동시에 줄이는 방향은 분명하다.

## Distillation pipeline

Teacher는 공개 SAM2-HieraB+ checkpoint이고 student는 EdgeTAM이다. Inference graph에는 teacher가 없다.

### Stage 1: image segmentation pretraining

Memory module을 사용하지 않고 teacher/student image encoder의 `F16`을 MSE로 맞춘다.

```math
L_{SAM}=L_{task}(O,GT)
+\gamma L_{img}(F^{t}_{16},F^{s}_{16})
```

Task loss는 focal/dice mask loss와 mask confidence를 위한 L1 loss를 포함한다.

### Stage 2: video segmentation training

Image feature뿐 아니라 memory-attention output도 맞춘다.

```math
L_{SAM2}=L_{task}(O,GT)
+\alpha L_{img}(F^{t}_{16},F^{s}_{16})
+\beta L_{mem}(F^{t}_{M},F^{s}_{M})
```

Video task loss에는 occlusion prediction cross-entropy가 추가된다. Distillation은 SA-V val/test에서 각각 +1.3, +3.3 J&F를 주며 inference overhead는 없다.

## Training recipe

### Image stage

- Dataset: SA-1B
- Resolution: 1024×1024
- Epoch/step: 본문 2 epoch, Appendix 약 175K steps
- Batch: 128
- Precision: bfloat16
- Optimizer: AdamW, beta `(0.9,0.999)`
- LR: `4e-4`, reciprocal-square-root schedule
- Warm-up/cooldown: 1K/5K iterations
- Weight decay: 0.1
- L2 gradient clipping: 0.1
- Max objects: 64
- Iterative correction points: 7
- Augmentation: horizontal flip

### Video stage

- Dataset: SA-V, SA-1B 10% subset, DAVIS, MOSE, YTVOS
- Resolution: 1024×1024
- Steps: 약 130K
- Batch: 256
- Sample: 8 frames, 약 3 objects
- Backbone LR: `6e-5`
- Other module LR: `3e-4`
- Schedule: cosine, warm-up 15K
- Augmentation: horizontal flip, affine, color jitter, grayscale
- Max masks: image 32, video 3

본문과 Appendix Table 5 사이에 loss-weight 표기 불일치가 있다. 본문은 dice/focal/IoU/distill을 `20/1/1/1`로 서술하지만 Appendix 표는 focal 20, dice 1로 적는다. Video stage도 같은 방향으로 충돌한다. 따라서 재현 시 논문 문장만 택하지 말고 official config 또는 저자 clarification으로 확인해야 한다.

## Progressive longer-sequence fine-tuning

기본 8-frame training 뒤 16-frame sample로 fine-tuning하고, EdgeTAM의 낮은 VRAM을 이용해 다시 32-frame sample로 fine-tuning한다.

- Image encoder freeze
- Distillation 미사용
- 원 video schedule의 1/3 iteration
- Inference memory-bank size는 7로 유지

Training clip만 길어지고 inference bank가 커지지 않으므로 inference cost는 동일하다. 긴 temporal pattern을 더 보면서 occlusion/reappearance 학습을 개선하려는 방식이다.

Table 3 ablation은 짧은 43K-step schedule로 수행됐고 Table 2 final result는 full/progressive recipe를 반영한다. `65.7` ablation SA-V val과 `72.3` final result를 같은 training 조건으로 비교하면 안 된다.

## 구현 pseudocode

```python
class SpatialPerceiver2D:
    def __init__(self, channels=64, n_global=256, n_local=256):
        self.global_latents = Parameter([n_global, channels])
        self.local_latents = Parameter([n_local, channels])
        self.shared_cross_attn = SingleHeadCrossAttention(channels)
        self.shared_self_attn = SelfAttention(channels)

    def forward(self, memory_map):
        # memory_map: [B, 64, 64, 64]
        tokens = flatten_hw(memory_map)              # [B, 4096, 64]
        global_tokens = self.shared_cross_attn(
            query=self.global_latents,
            key=tokens + sinusoidal_position(tokens),
            value=tokens,
        )
        global_tokens = self.shared_self_attn(global_tokens)

        windows = non_overlapping_windows(memory_map, count=256)
        local_tokens = self.shared_cross_attn(
            query=self.local_latents,
            key=windows,
            value=windows,
            one_latent_per_window=True,
        )
        local_tokens = self.shared_self_attn(local_tokens)
        local_tokens = add_2d_rope_or_position(local_tokens)
        return concatenate([global_tokens, local_tokens], dim="token")
```

Frame loop:

```python
for frame in video:
    f4, f8, f16 = repvit_image_encoder(frame)
    compressed_bank = [perceiver(m) for m in frame_memory_bank]
    conditioned = memory_attention(f16, compressed_bank, object_pointers)
    mask, pointer = mask_decoder(conditioned, f4, f8, user_prompt)
    new_memory = memory_encoder(f16, mask)
    fifo_enqueue(frame_memory_bank, new_memory, max_frames=7)
    fifo_enqueue(object_pointers, pointer, max_items=16)
```

효율적인 실제 구현은 과거 raw memory를 매 frame 다시 compress하지 않고 memory 생성 시 한 번 compress해 bank에 저장해야 한다. Paper architecture diagram상 Spatial Perceiver가 memory encoder와 bank 사이에 있으므로 이 해석과 일치한다.

## Image-only Segment Anything 결과

Memory module을 detach하면 image segmenter로 쓸 수 있다. SA-23의 1-click/5-click mIoU는 다음과 같다.

| Model | SA-23 All | Image subset | Video-frame subset | iPhone FPS |
| --- | ---: | ---: | ---: | ---: |
| SAM | 58.1 / 81.3 | 60.8 / 82.1 | 54.5 / 80.3 | 미보고 |
| SAM 2, SA-1B | 58.9 / 81.7 | 60.8 / 82.1 | 56.4 / 81.2 | 1.3 |
| SAM 2.1, internal mix | 61.9 / 83.5 | 63.3 / 83.8 | 60.1 / 83.2 | 1.3 |
| EdgeTAM | 55.5 / 81.7 | 56.0 / 81.9 | 54.8 / 81.5 | **40.4** |

EdgeTAM은 1-click에서는 낮지만 5-click에서는 SAM과 비슷하다. 40.4 FPS는 memory module을 뗀 image mode이며 video tracking 15.7 FPS와 구분해야 한다.

## VOS 최종 결과

Table 2는 YTVOS에 `G`, 나머지 dataset에 `J&F`를 보고한다.

`J`는 region IoU, `F`는 contour accuracy를 나타내며 일반적인 DAVIS `J&F`는 두 값의 평균이다.

```math
J=\frac{|M_{pred}\cap M_{gt}|}{|M_{pred}\cup M_{gt}|},
\qquad
J\&F=\frac{J+F}{2}
```

YTVOS의 `G`는 seen/unseen category에 대한 region과 contour 지표를 종합한다. 따라서 YTVOS 86.2와 SA-V 72.3은 metric 이름과 dataset 난도가 달라 절대 숫자를 직접 비교하면 안 된다. SA-V val/test는 fast motion, occlusion, disappearance와 다양한 annotation granularity를 강조해 mobile memory compression의 어려움을 더 잘 드러낸다.

| Method | MOSE | DAVIS17 | SA-V val | SA-V test | YTVOS | A100 FPS | iPhone FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM 2-B+ | 75.8 | **90.9** | 73.6 | 74.1 | 88.4 | 64.8 | 0.7 |
| SAM 2.1-B+ | **76.6** | 90.2 | **76.8** | **77.0** | **88.6** | 64.1 | 0.7 |
| EdgeTAM | 70.0 | 87.7 | 72.3 | 71.7 | 86.2 | **150.9** | **15.7** |

Reviewer가 FPS의 reciprocal로 계산하면 EdgeTAM은 iPhone 약 63.7 ms/frame, SAM 2는 약 1.43 s/frame이다. 이는 throughput reciprocal이며 camera capture, prompt UI, mask rendering을 포함한 paper-reported p50 latency가 아니다.

A100에서 EdgeTAM은 150.9 FPS지만 SM utilization이 50% 미만이고 CPU kernel launch/IO가 병목이라고 저자들이 설명한다. 작은 model의 FLOPs 감소가 high-end GPU latency에 선형 반영되지 않는 사례다.

논문은 EdgeTAM 전체 parameter 수, MACs, CoreML package 크기와 peak resident memory를 표로 보고하지 않는다. 따라서 2D Perceiver의 8배 token 감소를 전체 model이 8배 작거나 모든 latency가 8배 감소한다는 주장으로 확대하면 안 된다. 실제 baseline 대비 iPhone throughput gain은 `15.7/2.5`, 약 6.28배이고 SAM 2 대비는 `15.7/0.7`, 약 22.4배다.

## Component ablation

모든 ablation은 full schedule의 1/3인 43K step이다.

| Memory method | Distill | SA-V val | SA-V test | iPhone FPS |
| --- | --- | ---: | ---: | ---: |
| Raw baseline | No | 63.5 | 62.1 | 2.5 |
| 4×4 average pooling | No | 61.8 | 59.8 | 15.7 |
| 2D Perceiver | No | 64.4 | 62.5 | 15.7 |
| 2D Perceiver | Yes | **65.7** | **65.8** | 15.7 |

Average pooling도 같은 15.7 FPS지만 accuracy가 떨어진다. Learned compression의 핵심 가치는 속도보다 spatially meaningful information 보존이다.

### Latent allocation

| Global | 2D | SA-V val | SA-V test | FPS |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 63.5 | 62.1 | 2.5 |
| 256 | 0 | 62.0 | 60.6 | 15.7 |
| 0 | 256 | 63.1 | 62.4 | 15.7 |
| 256 | 256 | **64.4** | **62.5** | 15.7 |

Global-only는 spatial structure 손실로 baseline보다 낮다. 2D-only는 test에서 조금 높고 global+2D가 val에서 가장 좋다.

### Backbone과 memory-attention block 수

| Encoder | Blocks | SA-V val/test | FPS |
| --- | ---: | ---: | ---: |
| ViT-Tiny | 1 | 65.1 / 64.1 | 8.5 |
| ViT-Tiny | 2 | **67.9 / 66.0** | 7.4 |
| RepViT-M1 | 1 | 64.3 / 61.6 | **22.2** |
| RepViT-M1 | 2 | 65.7 / 65.8 | 15.7 |
| RepViT-M1 | 4 | 65.0 / 65.6 | 10.0 |

RepViT 4-block은 2-block보다 느리면서 val/test가 개선되지 않는다. 1-block은 빠르지만 test가 크게 낮아 default는 2-block이다.

Self-attention을 Perceiver에 넣으면 val은 `62.6 -> 64.4`지만 test는 `62.7 -> 62.5`로 0.2 낮다. 원문은 communication 효과를 긍정적으로 해석하지만 모든 split에서 개선된 것은 아니다.

## Prompt type별 zero-shot 결과

17개 video dataset에서 첫 frame에만 prompt를 제공한다.

| Method | 1-click | 3-click | 5-click | Box | GT mask |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM 2 | 64.3 | 73.2 | 75.4 | 72.9 | 77.6 |
| SAM 2.1 | 64.7 | 75.3 | 77.6 | 74.4 | 79.3 |
| EdgeTAM | 54.4 | 72.7 | 75.5 | 71.3 | 77.0 |

Prompt가 정확해질수록 SAM 2와 gap이 줄지만 1-click에서는 9.9 point 낮다. On-device UI에서 사용자 correction 수와 total interaction time까지 함께 평가해야 한다.

## 장점

- Image encoder뿐 아니라 video memory attention을 실제 iPhone에서 bottleneck으로 진단했다.
- Token 수를 줄이는 동시에 global context와 2D local structure를 보존한다.
- Same-speed average pooling보다 learned Perceiver가 정확함을 ablation으로 보였다.
- Image와 memory feature를 모두 distill해 compression loss를 training에서 보완한다.
- Fixed memory bank로 long-sequence fine-tuning의 inference cost 증가를 막는다.
- CoreML/iPhone 15 Pro Max 실측을 제공해 server GPU FLOPs만 제시하는 연구보다 deployment evidence가 강하다.

## 한계와 비판적 관점

### 1. SAM 2 accuracy를 완전히 유지하지 못한다

MOSE, DAVIS, SA-V, YTVOS에서 SAM 2/2.1보다 낮다. 특히 one-click prompt와 difficult video에서 gap이 크다.

### 2. iPhone benchmark 세부가 제한적이다

CoreML, Xcode performance report, iOS 18.1, CPU/NPU를 명시하지만 preprocessing, mask rendering, object 수, warm-up, p50/p95, sustained thermal run은 보고하지 않는다.

### 3. 첫 prompt latency와 steady-state FPS가 분리되지 않는다

첫 frame은 bank가 비어 있고 prompt/memory initialization이 필요하다. VLM의 TTFT와 유사한 first-mask latency를 별도 보고하지 않는다.

### 4. Multi-object memory scaling이 불명확하다

Training은 video당 약 3 object, image 최대 32/64 mask를 사용하지만 deployment FPS가 object 수에 따라 어떻게 변하는지 표가 없다.

### 5. Training data와 compute가 매우 크다

SA-1B, SA-V와 여러 VOS dataset, 175K+130K step, large SAM2-HieraB+ teacher가 필요하다. Inference는 작아도 재현 비용은 높다.

### 6. Loss-weight 문서 불일치

본문과 Appendix 표의 focal/dice 20/1 배치가 반대다. 공개 config 없이 paper-only 재현에 위험하다.

### 7. Failure granularity

Qualitative example에서 bird feet가 이전 memory에 없으면 이후에도 더 coarse한 granularity를 유지한다. Memory compression과 teacher bias가 작은 part 회복을 제한할 수 있다.

## 온디바이스 latency, memory, first-mask 분석

### Image encoder

RepViT-M1은 convolution-friendly mobile backbone이지만 1024 input의 F4/F8 high-resolution feature가 mask decoder와 skip에 필요하다. Image-only 40.4 FPS가 전체 video 15.7 FPS보다 빠르므로 memory path가 여전히 약 2.6배 throughput 차이를 만든다.

### Memory attention

Raw bank payload 자체는 수 MiB지만 attention score workload가 훨씬 크다. 2D Perceiver는 bank 저장, key/value projection, score matrix를 함께 줄이는 것이 핵심이다.

### First-mask latency

다음 구간을 따로 측정해야 한다.

```text
camera preprocess
 -> first-frame image encoder
 -> prompt encoder + mask decoder
 -> first memory encoding/compression
 -> first rendered mask
```

이후 frame은 memory-attention이 추가된다. 논문은 first-mask/TTFT를 보고하지 않으므로 63.7 ms reciprocal을 first-mask latency로 사용하면 안 된다.

### Sustained tracking

16 FPS가 camera 30 FPS보다 낮으므로 frame skipping, asynchronous queue, dynamic execution period가 필요할 수 있다. 5-10분 sustained run에서 temperature, power, p95, dropped frames를 기록해야 한다.

### Cache policy

Memory bank 7 frame과 object pointer 16개를 고정하면 memory가 video 길이에 선형 증가하지 않는다. 대신 long occlusion에서 오래된 informative frame이 FIFO로 사라질 수 있다. Accuracy-latency뿐 아니라 bank replacement policy를 ablation할 여지가 있다.

## 재현 체크리스트

- [ ] arXiv v1과 사용한 SAM 2 checkpoint/version을 고정했는가?
- [ ] 원본 파일명이 `58_EdgeTAM.pdf`와 일치하는가?
- [ ] Input resolution 1024와 F4/F8/F16 stride를 확인했는가?
- [ ] RepViT-M1 ImageNet initialization을 사용했는가?
- [ ] Memory shape가 `64×64×64`, frame bank 7, pointer 16인가?
- [ ] Global/2D latent가 각각 256이고 총 compression ratio가 8인가?
- [ ] Global/2D Perceiver block을 두 번 stack하는가?
- [ ] Global sinusoidal position과 local 2D-RoPE를 구분했는가?
- [ ] Teacher가 SAM2-HieraB+ 공개 checkpoint인가?
- [ ] Image stage에서 F16, video stage에서 F16과 FM을 distill하는가?
- [ ] 본문/Appendix loss-weight 불일치를 config로 해결했는가?
- [ ] 8-frame main training과 16/32-frame progressive fine-tuning을 구분했는가?
- [ ] Ablation 43K와 final 130K/progressive result를 섞지 않았는가?
- [ ] CoreML conversion, iOS 18.1, compute-unit 설정을 기록했는가?
- [ ] Image-only와 video FPS를 구분했는가?
- [ ] Object 수별 first-mask, p50/p95, peak memory를 측정했는가?
- [ ] 7-view가 아니라 실제 streaming frame queue 조건에서 측정했는가?
- [ ] Power, thermal throttling, dropped-frame rate를 기록했는가?

## 로드맵에서의 연결

```text
SAM: promptable image segmentation
 -> EdgeSAM: image encoder를 on-device로 압축
 -> SAM 2: memory bank와 memory attention으로 video 확장
 -> EdgeTAM: image encoder + memory attention을 함께 최적화
```

EdgeTAM은 로드맵의 중요한 시스템 교훈을 제공한다. Image encoder만 빨라져도 새 temporal module이 Amdahl bottleneck으로 남는다. 최종 통합 pipeline에서는 detector를 매 frame 실행하고 EdgeTAM은 선택 object가 있을 때만 활성화하며, frame memory와 ROI feature를 cache하는 구조가 적합하다.

추가 연구 주제로는 다음이 자연스럽다.

- ROI 기반 memory token compression
- Object motion에 따른 adaptive latent 수
- FIFO 대신 importance-aware memory replacement
- Occlusion 시 dynamic resolution/attention block 수
- Detector/segmenter shared encoder
- Frame-rate와 thermal state에 따른 execution period 제어

## 최종 평가

EdgeTAM의 가장 큰 공헌은 "SAM 2의 image encoder를 작게 만들었다"가 아니라 **video foundation model에서는 memory attention도 mobile bottleneck이라는 사실을 device measurement로 밝히고 token compression으로 직접 해결했다**는 점이다. 4096 memory token을 global/local 512 latent로 줄이는 설계는 dense tracking에 필요한 2D structure와 효율을 잘 절충한다.

다만 16 FPS는 camera real-time 30 FPS에 못 미치고, SAM 2 대비 difficult-set accuracy와 one-click 성능이 낮다. First-mask latency, multi-object scaling, p95, power/thermal도 남아 있다. 따라서 EdgeTAM은 완성된 universal mobile tracker라기보다 **image encoder, memory token, cache policy, user interaction을 함께 최적화해야 한다는 강한 on-device baseline**으로 보는 것이 정확하다.

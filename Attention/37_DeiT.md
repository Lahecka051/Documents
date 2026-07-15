# 37. Training Data-Efficient Image Transformers & Distillation through Attention

## 논문 정보

- 제목: **Training Data-Efficient Image Transformers & Distillation through Attention**
- 저자: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou
- 소속: Facebook AI, Sorbonne University
- 발표: ICML 2021
- arXiv: [https://arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877)
- PDF 기준 버전: arXiv v2, 2021-01-15
- 원본 파일: `37_DeiT.pdf`
- 핵심 키워드: Vision Transformer, data-efficient training, knowledge distillation, distillation token, strong augmentation, ImageNet-1K

## 한눈에 보는 요약

DeiT(Data-efficient image Transformer)는 새로운 attention 연산을 제안한 논문이 아니다. ViT의 architecture를 거의 그대로 두고, **ImageNet-1K만으로 convolution 없는 Transformer를 제대로 학습시키는 recipe**와 **attention을 통해 teacher 신호를 전달하는 distillation token**을 제안한다.

```text
224×224 RGB image
 -> 16×16 non-overlapping patches, 196개
 -> patch embeddings
 -> [CLS] + patch tokens (+ [DIST] token)
 -> 12 Transformer encoder blocks
 -> class head는 ground-truth label 학습
 -> distillation head는 teacher의 hard prediction 학습
 -> 추론 시 두 head의 softmax를 late fusion
```

핵심 결과는 다음과 같다.

- 외부 data 없이 ImageNet-1K에서 DeiT-B가 `81.8%` top-1을 달성한다.
- `384×384` fine-tuning 후에는 `83.1%`다.
- RegNetY-16GF teacher와 distillation token을 쓰면 300 epochs에 `83.4%`, `384×384`에서 `84.5%`다.
- 1000 epochs distilled DeiT-B를 `384×384`로 fine-tune하면 `85.2%`다.
- DeiT-B 300-epoch pretraining은 단일 8-GPU node에서 `53시간`, 2개 node에서 `37시간`이라고 보고한다.

이 논문의 가장 중요한 결론은 "ViT는 원래 data-efficient하다"가 아니다. **강한 augmentation, AdamW, stochastic depth, repeated augmentation, Mixup/CutMix, 장기 학습과 teacher의 convolutional inductive bias를 결합하면 ImageNet-1K 규모에서도 ViT를 경쟁력 있게 만들 수 있다**는 것이다.

## 연구 배경과 문제의식

### 원형 ViT의 성공 조건

ViT는 image를 patch token sequence로 바꾸고 표준 Transformer encoder를 적용했다. 하지만 원 논문의 최고 결과는 ImageNet-21k나 비공개 JFT-300M 같은 대규모 data로 사전학습한 뒤 downstream task로 transfer한 결과였다. ImageNet-1K만으로 학습한 ViT-B/16은 당시 recipe에서 top-1 `77.9%`에 그쳤다.

CNN은 다음 image-specific inductive bias를 구조에 내장한다.

- local receptive field
- spatial weight sharing
- translation equivariance에 가까운 동작
- stage별 downsampling hierarchy

Global self-attention은 이런 bias가 약하다. 큰 data에서는 관계를 직접 학습할 유연성이 장점이지만, 제한된 data에서는 sample efficiency가 낮아질 수 있다. DeiT는 architecture를 convolutional하게 바꾸기보다 optimization과 supervision으로 이 격차를 줄인다.

### 논문이 답하려는 두 질문

1. ViT를 수억 장의 외부 image와 대규모 infrastructure 없이 ImageNet-1K만으로 학습할 수 있는가?
2. CNN teacher의 지식을 Transformer student에게 전달할 때, 단일 output logit을 맞추는 고전 distillation보다 token과 attention을 활용할 방법이 있는가?

첫 질문의 답은 강한 training recipe이고, 두 번째 답은 distillation token이다.

## DeiT architecture

### Patch embedding

`224×224×3` image를 `16×16` patch로 나누면 patch 수는

```math
N=\frac{224}{16}\frac{224}{16}=14\times14=196
```

이다. Patch 하나를 flatten하면 `16×16×3=768`차원이다. 각 patch를 model dimension `D`로 선형 투영한다.

```math
X_p\in\mathbb{R}^{B\times196\times768},
\qquad
Z_p=X_pE,
\quad
E\in\mathbb{R}^{768\times D}
```

이는 `kernel_size=16`, `stride=16`, `out_channels=D`인 Conv2D와 동일하게 구현할 수 있지만, 논문의 model은 이를 patch linear projection으로 해석한다.

### Class token과 position embedding

일반 DeiT는 class token 하나를 붙인다.

```math
Z_0=[x_{cls};Z_p]+E_{pos}
\in\mathbb{R}^{B\times197\times D}
```

Distilled DeiT는 distillation token을 하나 더 붙인다.

```math
Z_0^{distill}=[x_{cls};Z_p;x_{dist}]+E_{pos}
\in\mathbb{R}^{B\times198\times D}
```

두 special token은 모든 patch 및 서로와 self-attention한다. Class token은 ground-truth label을, distillation token은 teacher prediction을 담당한다.

### Transformer block

DeiT는 ViT와 같은 Pre-LayerNorm block을 12개 쌓는다.

```math
\begin{aligned}
Z'_l&=Z_{l-1}+\mathrm{MSA}(\mathrm{LN}(Z_{l-1})),\\
Z_l&=Z'_l+\mathrm{FFN}(\mathrm{LN}(Z'_l)).
\end{aligned}
```

FFN은 `D -> 4D -> D`, GELU를 사용한다. Attention 한 head는

```math
\mathrm{Attn}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
```

이며 head dimension은 모든 model에서 `d_h=64`로 고정한다.

### Model variants - 논문 보고값

| Model | `D` | Heads | Layers | Params | 224 throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeiT-Ti | 192 | 3 | 12 | 5M | 2536 image/s |
| DeiT-S | 384 | 6 | 12 | 22M | 940 image/s |
| DeiT-B | 768 | 12 | 12 | 86M | 292 image/s |

Throughput은 GPU model과 batch 조건을 생략한 일반 device 수치가 아니다. Table 5 설명에 따르면 한 대의 16GB V100에서 model별 가능한 가장 큰 batch를 사용하고 30회 평균으로 계산했다. 따라서 `batch=1` 모바일 latency로 바꿔 읽으면 안 된다.

## Distillation objective

### Soft distillation

Teacher와 student logits을 `Z_t`, `Z_s`, temperature를 `τ`, ground-truth loss와 distillation loss의 비중을 `λ`라 하자. 논문의 soft distillation 식은

```math
\mathcal{L}_{global}
=(1-\lambda)\mathcal{L}_{CE}(\psi(Z_s),y)
+\lambda\tau^2
\mathrm{KL}
\left(\psi(Z_s/\tau),\psi(Z_t/\tau)\right)
```

이다. `ψ`는 softmax다. 논문은 soft baseline에 `τ=3.0`, `λ=0.1`을 사용한다.

Temperature가 커지면 class probability가 평평해져 teacher가 정답 class 외 class 사이에 가진 상대적 선호, 이른바 dark knowledge를 전달할 수 있다. `τ²`는 temperature 때문에 logit gradient scale이 줄어드는 것을 보정한다.

### Hard-label distillation

Teacher의 argmax를

```math
y_t=\arg\max_c Z_t(c)
```

로 두고 이를 pseudo-label처럼 사용한다.

```math
\mathcal{L}_{global}^{hard}
=\frac{1}{2}\mathcal{L}_{CE}(\psi(Z_s),y)
+\frac{1}{2}\mathcal{L}_{CE}(\psi(Z_s),y_t)
```

Augmentation된 image마다 teacher prediction을 다시 계산하므로 같은 원본 image라도 crop과 color transform에 따라 `y_t`가 달라질 수 있다. Hard distillation은 temperature와 KL을 없애 단순하지만 teacher의 class 간 확률 구조는 버린다. 이 논문의 실험에서는 오히려 hard target이 Transformer student에 더 잘 맞았다.

### Distillation token

고전 distillation baseline은 하나의 student output을 두 target에 맞춘다. DeiT의 제안은 역할을 분리한다.

```text
[CLS]  -> class head -> ground-truth y
[DIST] -> dist head  -> teacher hard label y_t
```

두 token은 독립된 branch에 갇히지 않는다. 모든 encoder layer의 self-attention에서 patch 및 서로와 상호작용한다. 즉 teacher signal이 마지막 classifier에만 걸리는 것이 아니라 distillation token의 hidden state가 attention을 통해 representation learning에 참여한다.

추론 때는 class head와 distillation head의 softmax 출력을 더해 late fusion한다.

```math
p(x)=\frac{1}{2}
\left[
\mathrm{softmax}(z_{cls})
+\mathrm{softmax}(z_{dist})
\right]
```

Argmax만 필요하면 `1/2`는 생략해도 결과가 같다.

## Distillation token이 단순한 두 번째 CLS와 다른 이유

저자들은 두 special token의 표현을 직접 분석했다.

- 학습된 class token과 distillation token parameter의 평균 cosine similarity: `0.06`
- 마지막 layer의 두 output embedding cosine similarity: `0.93`
- 두 token 모두 같은 ground-truth target을 쓰는 two-class-token 대조군의 token cosine similarity: `0.999`

서로 다른 target을 받은 두 token은 encoder를 지나며 공통 image 정보를 공유해 비슷해지지만 완전히 같아지지 않는다. 같은 target을 받으면 사실상 collapse하므로 token을 하나 더 넣는 것 자체가 성능 향상의 원인은 아니다.

다만 cosine 결과는 distillation token이 어떤 특정 CNN feature를 layer별로 복제한다는 증거가 아니다. 이 방법은 feature-map distillation이 아니라 teacher의 최종 hard label supervision을 special token에 배정한 것이다.

## Batch=1 tensor shape와 계산 예시

### 예시: distilled DeiT-S, `224×224`

설정은 `B=1`, patch `16`, `D=384`, heads `h=6`, head dimension `64`, layers `12`다.

```text
input image               [1, 3, 224, 224]
patch projection          [1, 196, 384]
+ CLS + DIST              [1, 198, 384]
Q, K, V after reshape     [1, 6, 198, 64] each
attention logits          [1, 6, 198, 198]
attention output          [1, 198, 384]
FFN hidden                [1, 198, 1536]
class/distill logits      [1, 1000] each
```

### Attention activation - 리뷰어 계산값

Attention matrix 원소 수는

```math
1\times6\times198\times198=235{,}224
```

이다. 이를 FP16으로 materialize하면

```math
235{,}224\times2
=470{,}448\ \text{bytes}
\approx459.4\ \text{KiB}
```

이고 FP32 softmax라면 약 `918.8 KiB`다. Q/K/V를 모두 FP16으로 보관하면

```math
3\times198\times384\times2
=456{,}192\ \text{bytes}
\approx445.5\ \text{KiB}
```

이다. FFN의 expanded activation은

```math
198\times1536\times2
=608{,}256\ \text{bytes}
\approx594.0\ \text{KiB}
```

다. 이것들은 tensor별 분석값이지 runtime peak memory가 아니다. Fused attention, buffer reuse, LayerNorm/softmax workspace, residual lifetime에 따라 실제 peak가 달라진다. 논문은 DeiT의 peak activation memory를 보고하지 않는다.

Distillation token 하나의 `197 -> 198` 증가는 attention 원소를 head당

```math
198^2-197^2=395
```

개 늘린다. 6 heads에서 2,370개로 매우 작다. Distillation token의 주된 비용은 compute가 아니라 두 번째 classifier head다.

### 한 block의 MAC - 리뷰어 계산값

Bias, LayerNorm, GELU, softmax를 제외한 근사 MAC은

```math
\mathrm{MAC}_{block}
\approx
4ND^2
+2N^2D
+8ND^2
=12ND^2+2N^2D
```

이다.

- `4ND²`: Q/K/V와 output projection
- `2N²D`: `QK^T`와 attention-value product
- `8ND²`: `D -> 4D -> D` FFN

`N=198`, `D=384`를 대입하면

```math
\mathrm{MAC}_{block}
=350{,}355{,}456+30{,}108{,}672
=380{,}464{,}128
```

으로 약 `380.5M MAC`이다. 12개 block은 약 `4.57G MAC`, patch projection까지 더하면 약 `4.62G MAC`이다. 이 값은 리뷰어의 구조 기반 근사이며 논문 Table 1의 보고 항목은 아니다. Profiler가 multiply와 add를 각각 FLOP으로 세면 숫자가 약 2배로 보일 수 있다.

이 예에서는 FFN과 projection의 `ND²` 항이 attention의 `N²D` 항보다 크다. 그러나 resolution을 `224 -> 384`로 올리면 patch 수가 `196 -> 576`으로 약 2.94배가 되고 attention 행렬은 약 8.6배가 되어 quadratic 항과 memory 비중이 급격히 커진다.

### Parameter reasoning - 리뷰어 계산값

DeiT-S 한 block의 주요 weight는

```math
\#W_{attn}\approx4D^2,
\qquad
\#W_{ffn}\approx8D^2,
\qquad
\#W_{block}\approx12D^2
```

이므로 `D=384`에서 `1,769,472`개, 12 layers에서 `21,233,664`개다. Patch projection `768×384=294,912`, position/special token, LayerNorm, bias, classifier를 더하면 논문의 rounded `22M`에 대응한다.

Distilled DeiT-B가 `86M -> 87M`으로 늘어나는 주된 이유는 1000-class distillation classifier `768×1000`과 bias다. Special token vector 하나 자체는 768 parameter에 불과하다.

## 핵심 구현 pseudocode

```python
class DistilledDeiT(nn.Module):
    def __init__(self, patch_embed, blocks, norm, dim, num_classes):
        super().__init__()
        self.patch_embed = patch_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 198, dim))
        self.blocks = blocks
        self.norm = norm
        self.cls_head = nn.Linear(dim, num_classes)
        self.dist_head = nn.Linear(dim, num_classes)

    def forward_features(self, image):
        x = self.patch_embed(image)              # [B, 196, D]
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        dist = self.dist_token.expand(b, -1, -1)
        x = torch.cat([cls, x, dist], dim=1)     # [B, 198, D]
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, -1]

    def forward(self, image):
        cls, dist = self.forward_features(image)
        z_cls = self.cls_head(cls)
        z_dist = self.dist_head(dist)
        if self.training:
            return z_cls, z_dist
        return (z_cls.softmax(-1) + z_dist.softmax(-1)) / 2


def hard_distillation_loss(outputs, y, teacher_logits):
    z_cls, z_dist = outputs
    y_teacher = teacher_logits.argmax(dim=-1)
    return 0.5 * (
        F.cross_entropy(z_cls, y, label_smoothing=0.1)
        + F.cross_entropy(z_dist, y_teacher)
    )
```

실제 재현에서는 official code의 token 순서, position interpolation, label smoothing 적용 위치, teacher의 augmentation 입력을 그대로 확인해야 한다. 위 코드는 개념을 드러내기 위한 최소형이다.

## Training recipe

### 기본 설정 - 논문 Table 9

| 항목 | DeiT-B |
| --- | --- |
| Epochs | 300 |
| Batch size | 1024 |
| Optimizer | AdamW |
| Base learning rate | `5×10^-4 × batch_size/512` |
| LR schedule | cosine |
| Weight decay | 0.05 |
| Warmup | 5 epochs |
| Label smoothing | 0.1 |
| Dropout | 사용 안 함 |
| Stochastic depth | 0.1 |
| Repeated augmentation | 사용 |
| RandAugment | `9 / 0.5` |
| Mixup probability | 0.8 |
| CutMix probability | 1.0 |
| Random erasing probability | 0.25 |

Weight는 truncated normal로 초기화한다. Learning rate는 batch 512를 기준으로 선형 scaling한다. 논문은 learning rate `5×10^-4`, `3×10^-4`, `5×10^-5`와 weight decay `0.03`, `0.04`, `0.05`를 cross-validation했다.

Repeated augmentation는 같은 image의 서로 다른 augmentation을 한 batch/epoch 구성에 반복해 넣는다. 논문은 3 repetitions를 쓰므로 형식상 원본 전체를 한 번 순회하는 epoch와 다르며, 저자들은 effective training time 비교를 위해 이를 300 epochs라고 부른다.

### Higher-resolution fine-tuning

Patch size를 16으로 유지한 채 `224 -> 384`로 올리면 patch grid가 `14×14 -> 24×24`가 된다. Class/distillation token position은 별도로 유지하고 patch position embedding을 2D grid로 reshape해 bicubic interpolation한다.

저자들은 bilinear interpolation이 이웃 vector 평균 때문에 norm을 줄일 수 있어 bicubic을 사용했다고 설명한다. 그러나 interpolation만 한 model을 바로 쓰는 것이 아니라 이후 fine-tuning한다. Default high-resolution fine-tuning은 `384×384`, 25 epochs이며 단일 8-GPU node에서 약 `20시간`이다.

## 실험 결과

### Teacher architecture ablation - 정확한 논문 값

Student는 distilled DeiT-B이며 teacher만 바꾼다.

| Teacher | Teacher acc. | Student 224 | Student 384 |
| --- | ---: | ---: | ---: |
| DeiT-B | 81.8 | 81.9 | 83.1 |
| RegNetY-4GF | 80.0 | 82.7 | 83.6 |
| RegNetY-8GF | 81.7 | 82.7 | 83.8 |
| RegNetY-12GF | 82.4 | **83.1** | 84.1 |
| RegNetY-16GF | **82.9** | **83.1** | **84.2** |

Teacher 정확도만으로 student 결과가 결정되지 않는다. 동일 정확도 수준의 Transformer teacher보다 CNN teacher가 훨씬 유리하다. 저자들은 이를 CNN inductive bias가 distillation을 통해 전달되는 효과로 추정하지만, feature-level causality를 직접 증명하지는 않는다.

### Distillation method ablation - 정확한 논문 값

모두 ImageNet 300-epoch pretraining 결과다.

| Supervision | Ti 224 | S 224 | B 224 | B 384 |
| --- | ---: | ---: | ---: | ---: |
| No distillation | 72.2 | 79.8 | 81.8 | 83.1 |
| Usual soft distillation | 72.2 | 79.8 | 81.8 | 83.2 |
| Hard distillation, single output | 74.3 | 80.9 | 83.0 | 84.0 |
| Token distill, class head only | 73.9 | 80.9 | 83.0 | 84.2 |
| Token distill, dist head only | **74.6** | 81.1 | 83.1 | 84.4 |
| Token distill, late fusion | 74.5 | **81.2** | **83.4** | **84.5** |

DeiT-B 224에서 soft distillation은 `81.8`로 no-distillation과 같지만 hard distillation은 `83.0`, token late fusion은 `83.4`다. 이 논문 설정에서 개선의 핵심은 soft target 자체보다 hard teacher label과 역할이 분리된 token/head다.

### ImageNet, ReaL, V2 및 throughput

| Model | Params | Size | V100 image/s | ImageNet | ReaL | V2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeiT-Ti | 5M | 224 | 2536.5 | 72.2 | 80.1 | 60.4 |
| DeiT-S | 22M | 224 | 940.4 | 79.8 | 85.7 | 68.5 |
| DeiT-B | 86M | 224 | 292.3 | 81.8 | 86.7 | 71.5 |
| DeiT-B 384 | 86M | 384 | 85.9 | 83.1 | 87.7 | 72.4 |
| Distilled DeiT-B | 87M | 224 | 290.9 | 83.4 | 88.3 | 73.2 |
| Distilled DeiT-B, 1000 ep. | 87M | 224 | 290.9 | 84.2 | 88.7 | 73.9 |
| Distilled DeiT-B 384 | 87M | 384 | 85.8 | 84.5 | 89.0 | 74.8 |
| Distilled DeiT-B 384, 1000 ep. | 87M | 384 | 85.8 | **85.2** | **89.3** | **75.2** |

Distillation token의 throughput 감소는 DeiT-B에서 `292.3 -> 290.9 image/s`로 작다. 반면 resolution 증가가 `290.9 -> 85.8 image/s`로 훨씬 큰 비용이다.

### Resolution ablation

DeiT-B를 224에서 학습하고 서로 다른 크기로 fine-tune한 결과다.

| Fine-tune size | Throughput | ImageNet | ReaL | V2 |
| --- | ---: | ---: | ---: | ---: |
| 160 | 609.31 | 79.9 | 84.8 | 67.6 |
| 224 | 291.05 | 81.8 | 86.7 | 71.5 |
| 320 | 134.13 | 82.7 | 87.2 | 71.9 |
| 384 | 85.87 | **83.1** | **87.7** | **72.4** |

`224 -> 384`에서 ImageNet은 `+1.3%p`지만 throughput은 약 `70.5%` 감소한다. On-device라면 정확도만 보고 384를 선택하지 말고 latency와 small-object 요구를 함께 봐야 한다.

## Training ablation의 핵심 해석

Table 8은 default DeiT-B의 `81.8/83.1`을 기준으로 ingredient를 제거하거나 바꾼다. Hyperparameter는 각 변형에 다시 최적화하지 않았으므로, 특정 기법의 보편적 필요성을 증명하기보다 **이 recipe 내부의 상호작용**을 보여준다.

| 변경 | 224 top-1 | 384 top-1 |
| --- | ---: | ---: |
| Default DeiT-B | **81.8±0.2** | **83.1±0.1** |
| Pretrain optimizer를 SGD로 | 74.5 | 77.3 |
| RandAugment 제거 | 79.6 | 80.4 |
| RandAugment 대신 AutoAugment | 81.2 | 81.9 |
| Mixup 제거 | 78.7 | 79.8 |
| CutMix 제거 | 80.0 | 80.6 |
| Mixup과 CutMix 모두 제거 | 75.8 | 76.7 |
| Repeated augmentation 제거 | 76.5 | 77.4 |
| Dropout 추가 | 81.3 | 83.1 |
| EMA 추가 | 81.9 | 83.1 |

Mixup/CutMix과 repeated augmentation의 영향이 크고, EMA 이득은 fine-tuning 뒤 사라진다. 일부 regularization 제거 행은 `4.3%` 또는 `3.4%`로 학습이 붕괴했는데, 저자도 hyperparameter가 변형에 맞지 않기 때문일 수 있다고 표시한다. 이를 "그 기법 없이는 Transformer가 절대 학습되지 않는다"로 일반화하면 안 된다.

## 장점과 핵심 기여

- ViT architecture를 복잡하게 바꾸지 않고 ImageNet-1K-only 학습 격차를 크게 줄였다.
- 재현 가능한 optimizer, augmentation, regularization recipe와 광범위한 ablation을 제공했다.
- Distillation target을 별도 token에 배정해 attention representation에 teacher supervision이 참여하도록 했다.
- Soft/hard distillation, token/head별 inference, teacher architecture를 정량적으로 분리했다.
- ImageNet뿐 아니라 ReaL, V2, 여러 transfer dataset에서 generalization을 확인했다.
- Accuracy만이 아니라 같은 repository 구현에서 V100 throughput을 비교했다.

## 한계와 비판적 관점

### 1. Data-efficient가 training-efficient와 같은 뜻은 아니다

외부 data는 없지만 300-1000 epochs, 강한 augmentation, teacher inference가 필요하다. 1000-epoch 최고 결과의 총 training energy와 teacher training 비용을 포함하면 가볍지 않다.

### 2. Teacher 의존성

Best distillation은 84M-parameter RegNetY-16GF teacher를 사용한다. Student deployment에는 teacher가 필요 없지만 training pipeline과 재현 비용에는 포함해야 한다. Teacher bias와 오류도 hard label로 전달된다.

### 3. 모바일 latency와 peak memory 미보고

논문 throughput은 V100, 최대 batch다. `batch=1` CPU/GPU/NPU latency, p95, peak activation, energy, thermal throttling은 보고하지 않는다. DeiT-Ti의 높은 image/s가 모바일에서 같은 순위를 보장하지 않는다.

### 4. Global attention의 quadratic cost는 그대로다

Distillation은 학습법을 개선하지만 `N²` attention을 줄이지 않는다. 384 fine-tuning에서 throughput이 크게 떨어지는 결과가 이를 드러낸다.

### 5. Classification 중심

원 논문은 image-level classification과 transfer에 집중한다. FPN형 hierarchy, detection/segmentation feature pyramid, small-object 성능은 다루지 않는다.

### 6. Architecture와 recipe의 기여 분리가 어렵다

원형 ViT와의 격차에는 optimizer, augmentation, regularization, implementation 차이가 동시에 들어간다. 이 논문의 요지는 바로 recipe이지만, "DeiT architecture가 ViT보다 우수하다"는 표현은 부정확하다.

### 7. Hard distillation 우위의 범위

Hard target 우위는 이 teacher, dataset, temperature/λ 탐색 범위에서의 결과다. 더 잘 calibrated된 teacher나 다른 task에서도 soft target이 항상 열등하다고 결론낼 수 없다.

## 자주 헷갈리는 지점

### DeiT는 새로운 attention인가

아니다. 기본 MSA와 ViT encoder를 사용한다. 핵심은 training recipe와 distillation token이다.

### Distillation token은 teacher feature token인가

아니다. Teacher의 intermediate feature나 attention map을 입력받지 않는다. Teacher의 최종 class prediction을 target으로 받는 학습 가능한 student token이다.

### 추론 때 teacher가 필요한가

필요 없다. Student의 class head와 distillation head만 사용한다.

### Distillation token을 patch token 대신 추가하는가

아니다. `[CLS]`, 196 patch, `[DIST]`가 모두 들어가 sequence가 198이 된다.

### 두 head의 logits를 평균하는가, softmax를 평균하는가

논문은 두 classifier의 **softmax output을 더해** 예측한다고 설명한다. Logit averaging과 완전히 같은 연산은 아니다.

### Data-efficient는 작은 dataset에서 scratch가 항상 잘 된다는 뜻인가

아니다. CIFAR-10 scratch에서 distilled DeiT-B가 `98.5%`를 얻지만 ImageNet pretraining 후 transfer한 `99.1%`보다 낮다. Diversity가 큰 pretraining은 여전히 유리하다.

## 온디바이스 관점

### Model size보다 token 수를 먼저 본다

DeiT-Ti는 5M parameter로 작지만 224에서 197/198 token의 global attention을 12번 수행한다. NPU가 softmax와 transpose를 잘 지원하지 않으면 비슷한 MAC의 CNN보다 느릴 수 있다. 반대로 dense GEMM이 잘 최적화된 accelerator에서는 높은 utilization을 얻을 수 있다.

### Distillation은 deployment-free compression 수단이다

Teacher 비용은 training에만 있고 inference graph에는 들어가지 않는다. Distillation token과 head가 추가하는 비용도 작다. 따라서 architecture를 바꾸지 않고 accuracy를 올리는 방법으로 유용하다. 필요하면 추론 후 두 head를 유지할지 class head 하나만 사용할지 정확도-latency ablation을 할 수 있다.

### Operator checklist

- Patch Conv2D 또는 reshape+GEMM
- LayerNorm
- QKV projection과 head reshape/transpose
- batched matmul
- softmax
- GELU
- residual add
- position embedding add

특히 LayerNorm, softmax, transpose가 NPU에서 CPU fallback되는지 확인한다. 하나라도 fallback되면 graph partition과 tensor copy가 latency를 지배할 수 있다.

### Resolution trade-off

384의 정확도 이득은 분명하지만 sequence 길이가 577/578로 늘어난다. 동일 patch size에서 attention memory는 약 8.6배가 된다. 모바일에서는 224/256/320별 accuracy, p50/p95, peak memory를 직접 측정하고 동적 resolution 정책을 검토하는 편이 낫다.

## 재현 계획

### 1단계: architecture와 shape 검증

1. DeiT-Ti 또는 DeiT-S를 공식 config로 구현한다.
2. 224 입력에서 patch token 196, class-only sequence 197, distilled sequence 198을 assert한다.
3. 모든 head dimension이 64인지 확인한다.
4. Distillation head가 class head와 weight를 공유하지 않는지 확인한다.
5. 추론 late fusion이 softmax 뒤에서 수행되는지 확인한다.

### 2단계: training recipe 재현

1. ImageNet-1K preprocessing과 label mapping을 고정한다.
2. 300 epochs, AdamW, cosine, warmup 5, weight decay 0.05를 사용한다.
3. RandAugment, Mixup, CutMix, random erasing, stochastic depth, repeated augmentation를 official 값으로 맞춘다.
4. 같은 augmented image를 teacher와 student에 넣고 hard target을 생성한다.
5. No distillation, hard baseline, distillation token을 같은 seed budget으로 비교한다.

### 3단계: 필수 ablation

| 축 | 설정 |
| --- | --- |
| Teacher | 없음, DeiT, RegNetY |
| Target | soft, hard |
| Token | single CLS, CLS+DIST |
| Inference | class only, dist only, late fusion |
| Resolution | 224, 320, 384 |

Top-1뿐 아니라 calibration error, teacher-student disagreement, training GPU-hours도 기록한다.

### 4단계: on-device benchmark

- `batch=1`, fixed resolution
- FP32/FP16/INT8
- CPU/GPU/NPU backend
- p50/p95 latency와 peak memory
- operator별 fallback과 layout-copy time
- energy/image, 온도, 10분 지속 실행 후 throttling

논문의 V100 최대-batch throughput은 참고값으로만 두고, 타깃 장치의 결과와 같은 표에 직접 섞지 않는다.

## 구현 체크리스트

- [ ] Patch size와 stride가 모두 16인가?
- [ ] 224 입력의 patch 수가 196인가?
- [ ] Distilled model sequence가 198인가?
- [ ] Position embedding에 두 special token 위치가 포함되는가?
- [ ] FFN hidden dimension이 `4D`인가?
- [ ] Head dimension이 64로 유지되는가?
- [ ] Class와 distillation classifier가 별도 parameter인가?
- [ ] Teacher는 training 중 eval mode인가?
- [ ] Augmented input에 대해 teacher hard label을 생성하는가?
- [ ] Label smoothing, Mixup/CutMix target 처리 순서가 official code와 같은가?
- [ ] Resolution 변경 시 patch position만 2D bicubic interpolation하는가?
- [ ] Throughput과 batch=1 latency를 구분해 보고하는가?
- [ ] Peak activation을 paper-reported 값으로 오인하지 않는가?

## 로드맵에서의 위치와 후속 연결

MobileNetV2가 architecture에 강한 local prior와 activation-memory 효율을 내장했다면, DeiT는 원형 global ViT의 약한 prior를 **학습 recipe와 teacher supervision**으로 보완한다. 이 대비가 효율적 backbone 연구의 중요한 출발점이다.

- Swin Transformer는 recipe만이 아니라 window attention과 hierarchy로 dense vision의 구조적 문제를 해결한다.
- EfficientViT는 high-resolution global context를 linear attention으로 바꾸고 실제 edge hardware latency를 정면으로 측정한다.
- MobileNetV4는 convolution block과 mobile-friendly attention을 NAS search space 안에서 결합한다.
- 압축 단계에서는 distillation token이 architecture-neutral teacher-student 학습의 기준선이 된다.

온디바이스 연구에서는 DeiT의 핵심을 단순히 pretrained checkpoint 사용으로 끝내지 말고, convolution teacher가 작은 Transformer student의 정확도와 calibration, robustness에 어떤 bias를 전달하는지 ablation해야 한다.

## 최종 평가

DeiT는 "ViT에는 수억 장의 data가 필수"라는 당시의 실용적 장벽을 크게 낮췄다. Architecture 변경보다 학습 recipe가 model의 성패를 좌우할 수 있음을 명확히 보였고, distillation token이라는 간단한 interface로 CNN teacher와 Transformer student의 장점을 결합했다.

동시에 data-efficient라는 명칭 뒤에는 300-1000 epoch 학습, 강한 augmentation, teacher compute가 있다. Global attention의 high-resolution cost와 모바일 operator 문제도 해결하지 않는다. 따라서 이 논문은 경량 inference architecture의 완성형이라기보다, **동일 architecture를 공정하게 비교하려면 training recipe와 supervision budget까지 통제해야 한다는 기준**으로 읽는 것이 가장 가치 있다.

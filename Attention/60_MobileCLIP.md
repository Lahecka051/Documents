# 60. MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training

## 논문 정보

- 저자: Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel
- 소속: Apple
- 공개: arXiv:2311.17049v2, 2024년 4월 1일
- 원본 파일: `60_MobileCLIP.pdf`
- 논문 링크: https://arxiv.org/abs/2311.17049
- 공개 코드 및 모델: https://github.com/apple/ml-mobileclip

## 한눈에 보는 요약

MobileCLIP은 CLIP의 dual-encoder 구조를 모바일에 맞게 줄이는 동시에, 작은 모델의 정확도 하락을 보완하기 위해 teacher 지식을 데이터셋에 미리 저장하는 multi-modal reinforced training을 제안한다. 이미지 encoder에는 FastViT 기반의 hybrid CNN-transformer `MCi`, text encoder에는 convolution과 self-attention을 섞고 추론 시 branch를 접는 `MCt`를 사용한다.

논문의 기여는 architecture와 training system 두 축으로 나뉜다.

1. `MCi0/1/2` image encoder와 `MCt` text encoder를 조합한 S0, S1, S2, 그리고 표준 ViT-B/16 기반 B variant를 제공한다.
2. CoCa가 만든 synthetic caption과 두 개의 강한 CLIP teacher embedding을 DataComp에 미리 저장해 `DataCompDR`을 만든다.
3. 학생 학습 중 teacher와 caption generator를 실행하지 않고 저장된 embedding으로 affinity distillation을 수행한다.
4. iPhone 12 Pro Max, Core ML, batch=1에서 image encoder와 text encoder latency를 각각 측정한다.
5. MobileCLIP-S0는 3.1 ms, S2는 6.9 ms, B는 13.7 ms의 encoder 합산 latency를 보고한다. 다만 p95, peak memory, 전력, 온도, quantization은 보고하지 않는다.

이 논문의 핵심 메시지는 단순히 작은 backbone을 찾았다는 것이 아니다. 강한 teacher 계산을 데이터 생성 시점으로 이동해 작은 architecture 탐색을 빠르게 반복할 수 있게 만든 것이 더 큰 시스템적 기여다.

## CLIP에서 MobileCLIP으로

일반 CLIP은 image encoder $f_I$와 text encoder $f_T$를 같은 임베딩 공간에 정렬한다. 학습 batch가 $b$이면 $b\times b$ image-text similarity를 계산하고 diagonal pair를 정답으로 삼는다. inference에서는 두 encoder가 독립적이므로 text embedding이나 image gallery embedding을 캐시할 수 있다.

문제는 원래 CLIP의 encoder가 모바일에 크고 느리다는 점이다. 작은 encoder로 바꾸면 latency는 줄지만 web-scale noisy data에서 표현력이 크게 떨어진다. 그렇다고 architecture 후보마다 대형 caption model과 teacher ensemble을 online으로 실행하면 탐색 비용이 감당되지 않는다.

MobileCLIP의 답은 다음 순서다.

```text
한 번만 수행하는 dataset reinforcement:
  DataComp image-text pair
    -> 여러 image augmentation parameter 저장
    -> CoCa로 synthetic caption 여러 개 생성
    -> 강한 CLIP teacher ensemble로 image/text embedding 계산
    -> 원본과 reinforcement를 DataCompDR에 함께 저장

반복 가능한 student training:
  DataCompDR에서 image, real/synthetic caption, teacher embedding 로드
    -> 작은 student의 CLIP loss + affinity distillation
    -> teacher forward 없이 여러 architecture를 반복 실험
```

teacher compute가 사라진 것이 아니라 데이터셋 생성 단계로 이동했다. 이 비용은 같은 reinforced dataset으로 많은 architecture와 recipe를 실험할 때 상각된다.

## Multi-modal dataset reinforcement

원본 sample $i$는 image $x^{(i)}_{img}$와 real caption $x^{(i)}_{txt}$로 구성된다.

### 1. Synthetic captions

CoCa `ViT-L-14` caption model로 image당 5개의 synthetic caption $x^{(i,s)}_{syn}$을 생성한다. real caption은 구체적일 수 있지만 noisy하고, synthetic caption은 image의 시각 내용을 더 직접적으로 설명하는 경향이 있다. 둘 중 하나만 사용하는 것보다 함께 사용하는 것이 classification과 retrieval의 균형에 중요하다.

### 2. Reproducible image augmentations

augmentation 결과 image 자체를 모두 저장하지 않고, 다음 변환을 재현하는 parameter $a^{(i,j)}$를 저장한다.

$$
\hat{x}^{(i,j)}_{img}=A(x^{(i)}_{img};a^{(i,j)}).
$$

DataCompDR-12M은 image당 30개, DataCompDR-1B은 10개의 augmentation을 생성한다. training 때 하나를 무작위로 고르고 원본 image에 다시 적용한다.

### 3. Ensemble teacher embeddings

두 CLIP teacher는 다음 ViT-L/14 모델이다.

- OpenAI ViT-L/14
- `datacomp_xl_s13b_b90k` ViT-L/14

각 teacher의 768-D normalized embedding을 concatenate해 1,536-D teacher representation을 만든다. augmented image, real caption, synthetic caption 각각의 embedding을 BF16으로 저장한다. teacher의 inference ensemble 성능이 가장 높은 조합과 student distillation에 가장 좋은 teacher가 반드시 같지는 않았고, Table 15의 student ablation을 기준으로 이 두 모델을 선택했다.

### 4. 실제 저장 항목

sample 하나에는 대략 다음이 들어간다.

- 원본 image $x_{img}$와 real caption $x_{txt}$
- augmentation parameters $a^{(i,\cdot)}$
- 5 synthetic captions $x^{(i,\cdot)}_{syn}$
- augmentation별 teacher image embeddings
- real caption과 synthetic caption별 teacher text embeddings

개별 sample은 Pickle로 만들고 Gzip lossless compression을 적용한다. embedding은 BF16으로 저장한다. 논문 보고 전체 저장량은 DataComp-12M 0.9 TB에서 DataCompDR-12M 1.9 TB, DataComp-1B 90 TB에서 DataCompDR-1B 140 TB로 증가한다.

### 저장 메모리 reviewer 계산

DataCompDR-12M의 30 augmentation과 6 text captions를 기준으로 embedding payload만 계산하면 다음과 같다.

- image teacher embeddings: $30\times1536\times2$ bytes = 90 KiB
- text teacher embeddings: $6\times1536\times2$ bytes = 18 KiB
- 합계: sample당 약 108 KiB

이는 압축 전 embedding만의 계산이다. 원본 image, 문자열, augmentation parameter, 파일 metadata가 추가된다. 논문의 실제 1.9 TB를 12.8M samples로 나누면 sample당 약 148 KB다. 저장 형식과 압축률에 따라 차이가 생기므로 두 값은 모순이 아니다.

중요하게도 140 TB의 reinforced dataset은 모바일에 배포되는 asset이 아니다. 이것은 offline training infrastructure 비용이다. 최종 장치에는 student weight와 필요한 class/query embedding만 들어간다.

## 학습 objective

batch $B$에 $b$개의 image-text pair가 있다고 하자. teacher $k$의 image/text embedding matrix는 $\Psi^{(k)}_{img},\Psi^{(k)}_{txt}\in\mathbb{R}^{b\times d_k}$이고, student는 $\Phi_{img},\Phi_{txt}\in\mathbb{R}^{b\times d}$다.

row-wise affinity distribution을 다음처럼 정의한다.

$$
S_\tau(U,V)=\operatorname{softmax}_{row}\left(\frac{UV^\top}{\tau}\right)\in\mathbb{R}^{b\times b}.
$$

image-to-text distillation은 teacher affinity를 student affinity가 따라가도록 KL divergence를 최소화한다.

$$
\mathcal{L}^{I2T}_{Distill}(B)
=\frac{1}{bK}\sum_{k=1}^{K}
KL\left(
S_{\tau_k}(\Psi^{(k)}_{img},\Psi^{(k)}_{txt})
\parallel
S_{\hat{\tau}}(\Phi_{img},\Phi_{txt})
\right).
$$

text-to-image 항은 두 modality의 순서를 바꾼다.

$$
\mathcal{L}_{Distill}
=\frac{1}{2}\mathcal{L}^{I2T}_{Distill}
+\frac{1}{2}\mathcal{L}^{T2I}_{Distill}.
$$

표준 symmetric CLIP loss와 결합한 최종 항은 다음과 같다.

$$
\mathcal{L}_{Total}(B)
=(1-\lambda)\mathcal{L}_{CLIP}(B)
+\lambda\mathcal{L}_{Distill}(B).
$$

한 augmented image에 real caption을 붙인 $B_{real}$과 synthetic caption을 붙인 $B_{syn}$을 각각 만들고 두 loss를 더한다.

$$
\mathcal{L}_{final}
=\sum_{B\in\{B_{real},B_{syn}\}}\mathcal{L}_{Total}(B).
$$

### Shape 중심 학습 pseudocode

```python
 # batch size b, teacher count K=2
image, real_text, aug_params, synth_text, teacher_cache = loader(batch_ids)
aug_image = replay_augmentation(image, choose(aug_params))
synth_text = choose(synth_text)

student_img = normalize(student_image_encoder(aug_image))
student_real = normalize(student_text_encoder(real_text))
student_syn = normalize(student_text_encoder(synth_text))

loss = 0
for student_txt, cached_teacher_txt in [
    (student_real, teacher_cache.real_text),
    (student_syn, teacher_cache.synth_text),
]:
    clip = symmetric_clip_loss(student_img, student_txt)
    kd = symmetric_affinity_kl(
        cached_teacher_img=teacher_cache.image,
        cached_teacher_txt=cached_teacher_txt,
        student_img=student_img,
        student_txt=student_txt,
    )
    loss += (1 - lambda_kd) * clip + lambda_kd * kd
```

teacher encoder forward는 없지만 teacher embedding을 읽고 $b\times b$ teacher affinity를 계산하는 IO와 matrix 연산은 남는다. 논문의 wall-clock 결과가 보여주는 것은 이 비용이 online teacher model 실행에 비해 작아 standard CLIP training과 같은 1.3 hours/epoch였다는 것이다.

## Batch tensor와 training memory

DataCompDR large-scale training의 global batch는 65,536이다. similarity 원소 수는 다음과 같다.

$$
b^2=65,536^2=4,294,967,296.
$$

하나의 dense affinity logit matrix만 BF16으로 완전히 materialize하면 8.0 GiB다. FP32라면 16.0 GiB다. 학생 방향별 logits, teacher별 affinity, softmax/KL workspace까지 동시에 저장하면 더 커진다. 실제 distributed implementation은 장치별 shard 또는 chunking이 필요하다.

ablation의 batch 8,192도 $67,108,864$ logits이며 BF16 약 128 MiB다. 이 계산은 shape 기반 reviewer estimate이고, 논문은 similarity matrix의 실제 peak memory나 sharding 구현을 보고하지 않는다. batch=1 inference에는 이 $b^2$ 학습 병목이 없다.

## MobileCLIP architecture

### MCi image encoder

MCi는 FastViT 기반 hybrid vision transformer다. FastViT FFN의 expansion ratio 4를 3으로 줄이고, parameter 수를 비슷하게 유지하도록 depth를 늘린다. stage channel $C_s$와 block depth $L_s$는 다음과 같다.

| Variant | Channels $\{C_1,C_2,C_3,C_4\}$ | Depths $\{L_1,L_2,L_3,L_4\}$ |
|---|---|---|
| MCi0 | {64, 128, 256, 512} | {2, 6, 10, 2} |
| MCi1 | {64, 128, 256, 512} | {4, 12, 20, 4} |
| MCi2 | {80, 160, 320, 640} | {4, 12, 24, 4} |

ImageNet supervised 비교에서 MCi0/1/2는 각각 11.8M/21.9M/36.3M parameters, 2.4/4.7/7.8 GFLOPs, iPhone latency 1.5/2.5/3.6 ms, top-1 82.2/83.8/84.5%를 기록했다. CLIP table의 parameter 수가 11.4M/21.5M/35.7M으로 조금 다른 것은 최종 head와 task 구성 차이에 따른 표기다.

MCi의 full internal stride와 모든 activation shape는 논문 표에 제공되지 않는다. 따라서 batch=1 peak memory를 이 PDF만으로 정확히 복원할 수 없다. 입력 $[1,3,256,256]$ FP16 payload만 384 KiB이고, 실제 peak는 초반 high-resolution feature map, residual branch, token mixer workspace에 의해 훨씬 크다.

### MCt text encoder

순수 convolution text encoder는 작고 빠르지만 정확도가 크게 떨어졌다. MCt는 총 6개 token-mixing block 중 2개 Text-RepMixer와 4개 self-attention layer를 사용하는 hybrid다. context length는 77이다.

Text-RepMixer는 train-time에 identity/normalization/1D depthwise convolution branch를 사용하고, inference 전에 BatchNorm과 skip branch를 하나의 depthwise convolution으로 fold한다. ConvFFN도 linear layer에 depthwise 1D convolution을 더하고 BatchNorm을 fold한다. 구현은 tensor를 `[B, C, 1, S]`로 reshape하며 $S=77$, kernel size 11을 선택한다.

architecture ablation은 다음 trade-off를 보인다.

| Self-attention layers | Params | iPhone latency | IN-val |
|---:|---:|---:|---:|
| 6 | 44.5M | 1.9 ms | 60.9 |
| 4, MCt 선택 | 42.4M | 1.6 ms | 60.8 |
| 2 | 40.4M | 1.4 ms | 60.2 |
| 1 | 39.3M | 1.3 ms | 60.0 |
| 0 | 38.3M | 1.2 ms | 57.9 |

self-attention 하나만 넣어도 pure convolution보다 2.1 point 높아진다. 선택한 4-layer attention MCt는 full transformer 대비 5% 작고 15.8% 빠르며 정확도 차이가 0.1 point다. kernel을 3, 11, 31로 키우면 IN-val은 56.3, 57.9, 59.0으로 오르지만 latency가 1.0, 1.2, 5.4 ms가 되므로 kernel 11을 선택했다.

### Structural reparameterization의 의미

reparameterization은 layer를 삭제해 근사하는 pruning이 아니다. 학습 중 여러 linear branch의 convolution kernel과 BatchNorm affine을 algebraically 합쳐 inference 시 동일 함수를 더 단순한 graph로 실행한다. export 전에 반드시 fold를 수행하고, fold 전후 출력 오차를 확인해야 논문의 latency 이점을 재현할 수 있다.

## MobileCLIP variants

| 모델 | Image / text encoder | Params, image + text | iPhone latency, image + text | 합계 |
|---|---|---:|---:|---:|
| MobileCLIP-S0 | MCi0 / MCt | 11.4M + 42.4M | 1.5 + 1.6 ms | 3.1 ms |
| MobileCLIP-S1 | MCi1 / Base | 21.5M + 63.4M | 2.5 + 3.3 ms | 5.8 ms |
| MobileCLIP-S2 | MCi2 / Base | 35.7M + 63.4M | 3.6 + 3.3 ms | 6.9 ms |
| MobileCLIP-B | ViT-B/16 / Base | 86.3M + 63.4M | 10.4 + 3.3 ms | 13.7 ms |

S0/S1/S2 입력은 256×256, B 입력은 224×224다. 따라서 이 latency table은 각 모델의 권장 입력에서 잰 deployment trade-off이며, 동일 해상도 architecture-only 비교는 아니다. 사용자의 로드맵처럼 동일 입력 해상도 비교를 하려면 별도 benchmark가 필요하다.

## 학습 설정

### Small-scale ablation

- DataCompDR-12M, 12.8M pairs
- global batch 8,192
- 8×A100-80GB
- 30k에서 45k iterations
- 한 epoch 1,562 iterations

### Large-scale training

- DataCompDR-1B, 1.28B pairs
- 200k iterations, 약 13B seen samples
- global batch 65,536
- 256×A100
- AdamW, peak LR $10^{-3}$, minimum LR $10^{-6}$
- warm-up 2,000 iterations, cosine decay
- $\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.2
- BF16 mixed precision, EMA 0.9995
- MobileCLIP-B은 CLIP loss 0.25, KD 0.75, 즉 $\lambda=0.75$
- S0/S1/S2는 $\lambda=1.0$

seen sample 하나는 random augmented image, real caption, random synthetic caption의 triplet이다. 일반 image count나 unique pair count와 직접 동일시하면 안 된다.

## 핵심 ablation

### Reinforcement 구성 요소

ViT-B/16:Base를 DataCompDR-12M에서 30k iterations 학습한 Table 2는 다음 누적 효과를 보인다.

| 설정 | IN-val | Flickr30k |
|---|---:|---:|
| standard CLIP, reinforcement 없음 | 44.5 | 41.8 |
| synthetic caption 추가 | 51.9 | 69.3 |
| distillation만 사용 | 54.5 | 66.1 |
| strong augmentation 추가 | 59.3 | 70.5 |
| ensemble teacher 추가 | 61.7 | 72.0 |
| $\lambda=0.7$로 균형 조정 | 60.7 | 74.2 |

synthetic caption은 retrieval에 특히 큰 영향을 주고, strong augmentation은 IN-val +4.8, Flickr30k +4.4 point를 더했다. ensemble teacher는 IN-val을 2.4 point 더 올렸다. $\lambda=1$은 IN-val, $\lambda=0.7$은 Flickr30k에 유리해 하나의 scalar가 classification-retrieval trade-off를 조절한다.

### Real caption과 synthetic caption

distillation only에서 $B_{real}$만 쓰면 IN-val 56.4, Flickr30k 57.0이다. $B_{syn}$만 쓰면 49.8, 72.2다. 두 batch를 모두 loss에 넣으면 61.7, 72.0으로 균형이 가장 좋다. 둘 중 하나를 random 선택하는 방식은 57.3, 68.6으로 낮다.

### Augmentation와 caption 개수

45k iterations에서 성능은 약 5 augmentations와 2 synthetic captions부터 대체로 포화된다. 최대 설정 30 augmentations, 5 captions는 IN-val 64.74, Flickr30k 75.66이다. 저장량을 줄여야 하면 5/5 또는 2/2 설정이 합리적이다. Table 16에서 12.8k samples 기준 BF16 30/6 설정은 1.9 GB, 5/5는 1.2 GB, 2/2는 1.0 GB다.

### Online teacher 대비 training time

12.8M samples 한 epoch, 8×A100에서:

- standard real-caption CLIP: 1.3 hours
- teacher와 caption generation을 모두 online 수행: 21.1 hours
- synthetic caption만 저장하고 teacher online: 4.1 hours
- DataCompDR에서 둘 다 저장: 1.3 hours

따라서 paper가 말하는 no train-time overhead는 같은 hardware와 이 실험에서 epoch wall-clock이 standard CLIP과 같았다는 뜻이다. 최초 reinforcement 생성 시간과 1.9/140 TB 저장 시스템 비용은 포함되지 않는다.

## 주요 결과

### Large-scale accuracy와 latency

| 모델 | IN-val | IN-shift | Flickr T→I / I→T R@1 | COCO T→I / I→T R@1 | 38개 평균 |
|---|---:|---:|---:|---:|---:|
| MobileCLIP-S0 | 67.8 | 55.1 | 67.7 / 85.9 | 40.4 / 58.7 | 58.1 |
| MobileCLIP-S1 | 72.6 | 60.7 | 71.0 / 89.2 | 44.0 / 62.2 | 61.3 |
| MobileCLIP-S2 | 74.4 | 63.1 | 73.4 / 90.3 | 45.4 / 63.4 | 63.7 |
| MobileCLIP-B | 76.8 | 65.6 | 77.3 / 91.4 | 50.6 / 68.8 | 65.2 |

MobileCLIP-S0는 OpenAI ViT-B/16의 149.6M parameters와 14.8 ms에 비해 53.8M, 3.1 ms이며, 논문은 이를 약 3배 작고 5배 빠르면서 38개 평균 정확도는 같다고 요약한다. Table 7의 실제 비율은 parameter 약 2.78배, latency 약 4.77배다.

MobileCLIP-S2는 DataComp B/32-256보다 38개 평균이 2.8 point 높고, parameter 1.5배 작고, 1.4배 빠르다. abstract는 prior best ViT-B/16 기반 모델보다 2.3배 빠르면서 더 정확하다고 주장하지만, 정확한 비교 대상과 입력 해상도를 함께 확인해야 한다.

MobileCLIP-B는 SigLIP-B/16보다 38개 평균이 2.9 point 높고 parameter가 26.3% 적다. MobileCLIP-B long training은 39B seen samples에서 IN-val 77.2, 38개 평균 65.8로 13B schedule보다 각각 0.4, 0.6 point 높았다.

### Learning efficiency

DataCompDR-12M에서 MobileCLIP-B는 0.37B seen samples로 IN-val 65.3을 기록한 반면 같은 DataComp-12M의 standard training은 50.1이다. 1.48B seen samples에서는 reinforced 71.7, non-reinforced 55.7이다.

Figure 6에서 저자들은 다음을 보고한다.

- ImageNet-val: iteration efficiency 약 10배, data efficiency 100배 이상
- Flickr30k: iteration efficiency 약 18배, data efficiency 최대 1,000배

여기서 효율은 target metric에 도달하는 iteration 또는 dataset size 비교다. teacher pretraining과 DataCompDR 생성 비용을 포함한 end-to-end energy 효율은 아니다.

### Compositional retrieval, ARO

모두 ViT-B/16:Base인 비교에서 MobileCLIP-B는 다음을 기록했다.

| 모델 | IN-val | VG Relation | VG Attribute | COCO Order | Flickr30k Order |
|---|---:|---:|---:|---:|---:|
| OpenAI CLIP | 68.3 | 58.7 | 62.2 | 50.4 | 57.3 |
| SigLIP | 76.0 | 35.1 | 56.0 | 32.7 | 40.7 |
| DFN CLIP | 76.2 | 33.1 | 57.4 | 18.5 | 22.5 |
| MobileCLIP-B | 76.8 | 54.6 | 68.4 | 55.5 | 61.2 |

단순 zero-shot classification이나 noisy retrieval 최적화가 relation/order 이해를 보장하지 않는다는 점에서 의미가 있다. 다만 OpenAI CLIP의 VG Relation 58.7은 MobileCLIP-B 54.6보다 높으므로 모든 compositional 항목에서 최고라는 뜻은 아니다.

## iPhone latency를 정확히 읽는 법

논문의 실제 deployment 조건은 다음과 같다.

- iPhone 12 Pro Max
- iOS 17.0.3
- Core ML Tools 7.0으로 export
- batch size 1
- 각 method의 고유 입력 크기 사용
- MobileOne 논문의 protocol을 따름

text가 매 query마다 바뀌는 image-text retrieval이라면 `image + text` 합산 시간이 의미 있다. 반대로 ImageNet처럼 class prototype이 고정된 zero-shot classification에서는 text encoder를 사전에 한 번 실행해 캐시할 수 있으므로 steady-state frame latency는 image encoder 1.5, 2.5, 3.6, 10.4 ms에 더 가깝다. 이 해석은 dual-encoder 구조에서 도출한 배포 최적화이며 논문 table의 합산 수치를 바꾸는 것이 아니다.

또한 latency가 3.1 ms라고 해서 300 FPS sustained camera system을 보장하지 않는다. preprocess, camera copy, prototype matmul, postprocess, thermal throttling이 표에 포함됐는지 상세 breakdown이 없다.

## 논문의 한계와 비판적 관점

### 1. Architecture와 data recipe가 결합됨

MobileCLIP이 경쟁 모델보다 높은 이유에는 MCi/MCt뿐 아니라 DataCompDR의 synthetic caption, strong augmentation, teacher ensemble이 함께 작용한다. architecture-only 공정 비교에는 동일 데이터, 동일 seen samples, 동일 해상도의 재학습이 필요하다.

### 2. Offline 비용이 매우 큼

DataCompDR-1B은 140 TB이고 teacher ensemble 및 CoCa caption 생성이 필요하다. 많은 student를 반복 학습하면 유리하지만 단일 모델만 만드는 조직에는 reinforcement 생성 비용이 상각되지 않을 수 있다. teacher가 갱신되면 cache도 다시 만들어야 한다.

### 3. 저장된 teacher는 augmentation 범위를 고정함

offline embedding은 미리 생성한 augmentation에 묶인다. RangeAugment처럼 training 중 동적으로 바뀌는 변환은 teacher와 동일하게 적용할 수 없어 student에만 적용했다. 같은 RangeAugment를 양쪽에 적용하면 IN-val 56.6으로 student-only 55.9보다 높지만 reinforced workflow에서는 직접 사용할 수 없다.

### 4. 실제 모바일 memory와 tail latency가 없음

parameter와 평균성 latency는 있지만 peak activation memory, p50/p95, cold start, model load, power, temperature, sustained throttling이 없다. iPhone 12 Pro Max 한 장치 결과이므로 Android NPU나 저가 SoC로 일반화할 수 없다.

### 5. Quantization 결과가 없음

training과 stored embedding에는 BF16을 쓰지만 배포 model의 FP16/INT8 조건, quantization accuracy, file size는 보고하지 않는다. parameter 수만으로 device resident memory를 단정하면 안 된다.

### 6. Social bias와 dataset governance 평가가 부족함

web-scale DataComp, synthetic caption, teacher ensemble의 bias가 student에 전달될 수 있다. 논문은 latency와 benchmark accuracy에 집중하며 demographic bias, privacy, harmful retrieval, caption provenance를 체계적으로 평가하지 않는다.

## 온디바이스 구현 제안

### 고정 클래스 분류

```text
offline 또는 앱 초기화:
  class prompts -> text encoder -> normalized prototypes -> cache

frame마다:
  256 resize/crop -> MCi -> normalized image embedding
  -> prototype matmul -> calibrated top-k 또는 reject
```

S0에서 text encoder는 전체 parameter의 약 79%인 42.4M/53.8M이다. taxonomy가 고정이면 text encoder를 장치 graph에서 제거하고 prototype만 배포하는 구성이 model size를 크게 줄일 수 있다. 반면 자유 query retrieval에는 MCt가 필요하다.

### 자유 query retrieval

```text
image 등록:
  MCi -> embedding -> local ANN index

query 시:
  tokenizer -> MCt -> query embedding
  -> ANN search -> top-k result
```

이때 TTFT는 text encoder 1.6 또는 3.3 ms만이 아니라 tokenizer, model wake-up, ANN search, result serialization을 포함해야 한다. gallery embedding memory는 `개수 × embedding dimension × dtype`으로 별도 기록한다.

### 모델 선택 기준

- S0: detector 뒤 이벤트 분류처럼 최소 latency와 size가 우선일 때
- S1: 정확도와 latency의 중간점이 필요할 때
- S2: 7 ms 안팎 encoder budget에서 74.4 IN-val이 필요할 때
- B: 약 14 ms budget과 더 큰 memory를 허용하고 76.8 IN-val 및 높은 retrieval이 필요할 때

이 선택은 논문 장치에서의 기준이다. 실제 target SoC에서는 MCt depthwise convolution과 attention의 backend kernel 지원에 따라 순서가 바뀔 수 있다.

## 재현 체크리스트

### Reinforcement 생성

- [ ] 원본 DataComp subset과 filtering hash를 고정했다.
- [ ] image당 synthetic caption 수와 CoCa checkpoint를 기록했다.
- [ ] augmentation parameter를 정확히 replay할 수 있다.
- [ ] 두 teacher checkpoint, resolution, normalization을 고정했다.
- [ ] teacher embedding concat 순서와 BF16 변환을 검증했다.
- [ ] sample별 cache 누락과 corruption을 검사했다.
- [ ] reinforcement 생성 시간, GPU hours, storage를 별도 기록했다.

### Student 학습

- [ ] real/synthetic batch를 각각 loss에 넣었다.
- [ ] teacher와 student temperature를 구분했다.
- [ ] KL 방향을 teacher distribution에서 student distribution으로 설정했다.
- [ ] I2T와 T2I distillation을 모두 사용했다.
- [ ] $\lambda=1.0$과 0.75의 classification/retrieval trade-off를 재현했다.
- [ ] global batch similarity를 shard하고 peak memory를 측정했다.

### Export와 모바일 측정

- [ ] Text-RepMixer와 ConvFFN branch를 inference graph로 fold했다.
- [ ] fold 전후 numerical parity를 검사했다.
- [ ] 입력 224/256과 preprocess를 명시했다.
- [ ] image/text encoder를 각각 batch=1로 측정했다.
- [ ] prototype cache 사용 여부를 구분했다.
- [ ] p50/p95, cold/warm, peak memory, model size를 기록했다.
- [ ] 전력, 온도, 10분 이상 sustained throttling을 기록했다.
- [ ] FP16과 INT8에서 IN-val, retrieval R@K, calibration을 재측정했다.

## 로드맵에서의 위치

MobileCLIP은 CLIP 다음에 읽어야 할 직접적인 온디바이스 확장이다. CLIP에서 dual encoder와 contrastive objective를 이해했다면, 이 논문에서는 다음 세 질문에 집중하면 된다.

1. vision과 text encoder 중 어느 쪽을 줄일 것인가?
2. teacher compute를 online training에서 offline dataset으로 옮기는 것이 언제 경제적인가?
3. 고정 text prototype cache를 활용하면 실제 앱 graph에서 무엇을 제거할 수 있는가?

첫 구현 산출물로 MobileCLIP-S0와 S2를 같은 target device에서 비교하는 것이 좋다. 입력 해상도를 256으로 고정하고 image encoder만 실행하는 고정 class 설정과 image+text를 모두 실행하는 자유 query 설정을 나눠 측정한다. 정확도는 ImageNet 또는 자체 taxonomy top-1뿐 아니라 Flickr/COCO retrieval R@1, ARO relation/order, score calibration을 함께 기록해야 한다.

다음 BLIP-2와 LLaVA에서는 CLIP과 달리 visual embedding이 LLM token으로 변환되고 autoregressive decoder가 추가된다. MobileCLIP의 1.5에서 10.4 ms image latency는 그 VLM 파이프라인에서 vision encoder baseline으로 사용할 수 있지만, LLM prefill과 decode latency를 포함하지 않는다는 점을 유지해야 한다.

## 최종 평가

MobileCLIP은 작은 architecture만 제안한 논문보다 실용적이다. MCi와 MCt로 iPhone latency를 직접 낮추고, DataCompDR로 작은 student가 잃는 정확도를 회복하며, teacher forward를 반복 학습 경로에서 제거했다. 특히 image와 text encoder latency를 분리해 보고한 점은 고정 prototype cache가 가능한 실제 deployment 설계에 유용하다.

다만 140 TB reinforced dataset과 teacher/caption 생성 비용, 서로 다른 입력 해상도, mobile peak memory와 p95 및 전력의 부재를 함께 봐야 한다. 로드맵의 목표가 온디바이스 CV라면 논문의 평균 latency를 그대로 인용하는 데서 멈추지 말고, 같은 해상도와 batch=1에서 `image-only`, `image+text`, `cached prototype`, `FP16/INT8` 네 조건을 재측정하는 것이 핵심 재현 실험이다.

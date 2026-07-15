# 66. MobileCLIP2: Improving Multi-Modal Reinforced Training

## 논문 정보

- 원본 파일: `66_MobileCLIP2.pdf`
- 제목: **MobileCLIP2: Improving Multi-Modal Reinforced Training**
- 저자: Fartash Faghri, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Alexander Toshev, Oncel Tuzel, Hadi Pouransari
- 소속: Apple
- 발표: Transactions on Machine Learning Research, 2025년 8월
- 링크: [arXiv:2508.20691](https://arxiv.org/abs/2508.20691)
- 코드: [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip), [apple/ml-mobileclip-dr](https://github.com/apple/ml-mobileclip-dr)
- 핵심 키워드: CLIP, offline distillation, reinforced dataset, synthetic caption, DFNDR-2B, FastViT, mobile retrieval

## 한 문장 요약

MobileCLIP2는 더 좋은 DFN image-text data, 두 개의 강한 CLIP teacher, DFN에서 학습한 CoCa synthetic captioner를 offline reinforced dataset으로 결합해, iPhone 12 Pro Max에서 수 ms에서 수십 ms 범위의 image-text encoder로 높은 zero-shot 정확도를 얻는다.

## 이 논문의 핵심은 모델보다 학습 데이터 파이프라인이다

MobileCLIP2는 architecture 논문인 동시에 distillation data engineering 논문이다. 일반 knowledge distillation은 student step마다 teacher도 실행하므로 큰 teacher ensemble과 수십억 sample을 쓰기 어렵다. 이 논문은 teacher output을 미리 계산해 dataset에 저장한다.

```text
raw image-text pair
  -> multiple image augmentations
  -> two CLIP teachers
  -> teacher image/text embeddings 저장
  -> CoCa captioner
  -> five synthetic captions 저장
  -> reinforced dataset DFNDR-2B

student training
  -> 저장된 teacher embedding과 caption만 읽음
  -> online teacher forward 없음
  -> CLIP contrastive distillation
```

online step 시간은 일반 CLIP training과 거의 같아지지만 비용이 사라진 것은 아니다. teacher inference를 앞당겨 수행하고, 수십에서 수백 TB의 storage와 I/O로 바꾼 것이다. training compute와 dataset construction compute를 구분해서 읽어야 한다.

## CLIP 기본 구조

image encoder `f_I`와 text encoder `f_T`가 각각 embedding을 만들고 L2 normalize한다.

```math
u_i=\frac{f_I(x_i)}{\lVert f_I(x_i)\rVert_2},
\qquad
v_i=\frac{f_T(t_i)}{\lVert f_T(t_i)\rVert_2}
```

batch size가 `b`일 때 similarity logit은 다음과 같다.

```math
Z_{ij}=\frac{u_i^\top v_j}{\tau}
```

positive pair는 diagonal `(i,i)`이고 나머지는 in-batch negative다. image-to-text와 text-to-image cross entropy를 평균한 표준 CLIP loss를 사용한다.

```math
\mathcal{L}_{CLIP}
=\frac{1}{2b}\sum_i
\left[
-\log\frac{e^{Z_{ii}}}{\sum_j e^{Z_{ij}}}
-\log\frac{e^{Z_{ii}}}{\sum_j e^{Z_{ji}}}
\right]
```

inference에서는 image와 text encoder가 독립적이다. class name prompt를 미리 encoding하고 cache하면 매 camera frame마다 image encoder와 작은 similarity matrix만 실행하면 된다. 이 점이 autoregressive VLM과 가장 큰 차이다. TTFT와 tokens/s가 아니라 embedding latency, retrieval top-k latency, gallery memory가 핵심 지표다.

## Multi-modal reinforced distillation

student embedding을 `Phi_img, Phi_txt`, `k`번째 teacher embedding을 `Psi_img^(k), Psi_txt^(k)`라 하자. 둘 다 batch 차원을 포함한다.

```math
\Phi_{img},\Phi_{txt}\in\mathbb{R}^{b\times d},
\qquad
\Psi_{img}^{(k)},\Psi_{txt}^{(k)}\in\mathbb{R}^{b\times d_k}
```

row-wise distribution을 다음처럼 정의한다.

```math
S_\tau(U,V)=\mathrm{softmax}_{row}(UV^\top/\tau)
```

teacher가 만든 image-to-text 및 text-to-image similarity distribution을 student가 KL divergence로 따라간다.

```math
\mathcal{L}_{Distill}
=\frac{1}{2bK}\sum_{k=1}^{K}
\left[
D_{KL}\!\left(
S_{\tau_k}(\Psi_{img}^{(k)},\Psi_{txt}^{(k)})
\Vert
S_{\hat\tau}(\Phi_{img},\Phi_{txt})
\right)
\right.
```

```math
\left.
+D_{KL}\!\left(
S_{\tau_k}(\Psi_{txt}^{(k)},\Psi_{img}^{(k)})
\Vert
S_{\hat\tau}(\Phi_{txt},\Phi_{img})
\right)
\right]
```

전체 loss는 다음과 같다.

```math
\mathcal{L}_{Total}
=(1-\lambda)\mathcal{L}_{CLIP}
+\lambda\mathcal{L}_{Distill}
```

최종 MobileCLIP2 training recipe는 `CLIP loss weight=0`, `KD loss weight=1`, 즉 `lambda=1`이다. ground-truth caption과 synthetic caption의 weight는 각각 1이다. student는 hard one-hot pair보다 teacher가 알려 주는 batch 내 semantic relation을 학습한다.

## Temperature가 왜 중요한가

teacher similarity의 logit scale은 distribution entropy를 결정한다. 너무 큰 scale은 몇 개 pair에 확률이 몰리고, 너무 작으면 informative relation이 평평해진다. teacher가 자체 CLIP training에서 학습한 scale이 student KD에 항상 최적은 아니다.

| Teacher | KD logit scale | ImageNet val | Flickr30k | Avg. 38 |
| --- | ---: | ---: | ---: | ---: |
| DataComp-XL ViT-L/14 | 50 | 62.6 | 65.6 | 53.3 |
| DFN2B ViT-L/14 | 70 | 65.5 | 68.0 | 56.5 |
| DFN5B ViT-H/14 | 90 | 64.0 | 65.9 | 54.7 |
| DFN5B ViT-H/14@384 | 55 | 64.6 | 67.6 | 54.4 |
| DFN2B ViT-L/14-s39b | 60 | 65.2 | 67.5 | 54.8 |

최종 ensemble은 `DFN2B-CLIP-ViT-L-14-s39b`와 `DFN2B-CLIP-ViT-L-14`이며 scale은 각각 60과 70이다. 두 teacher의 optimal scale을 독립적으로 찾았고 joint tuning은 하지 않았다.

## Reinforced dataset 구성

최종 DFNDR-2B의 base는 약 1.9B image-text pair인 DFN-2B다.

- CLIP teacher 1: DFN2B-CLIP-ViT-L/14-s39b
- CLIP teacher 2: DFN2B-CLIP-ViT-L/14
- synthetic teacher: CoCa-ViT-L/14
- CoCa pretraining: DFN-2B
- CoCa fine-tuning: permissive license의 MSCOCO-38K
- synthetic caption: image당 5개
- full dataset image augmentation embedding: image당 2개

12.8M ablation subset은 image당 30 augmentation과 BF16 embedding을 저장하며 1.3TB다. full DFNDR-2B는 1.9B sample, image augmentation 2개, synthetic caption 5개이며 표에 기록된 크기는 162TB다. full-scale에서는 BF16 flag가 아니므로 저장 format과 compression을 공식 pipeline에서 확인해야 한다.

| Dataset | Samples | Teachers | Image augs | Captions | Size |
| --- | ---: | --- | ---: | ---: | ---: |
| DataComp-1B | 1.3B | 없음 | 없음 | 없음 | 90TB |
| DFN-2B | 1.9B | 없음 | 없음 | 없음 | 65TB |
| DataCompDR-1B | 1.3B | OpenAI + DataComp-XL | 10 | 5 | 140TB |
| DFNDR-2B | 1.9B | 두 DFN teacher | 2 | 5 | 162TB |

offline reinforcement는 재사용성이 높다. 여러 student architecture가 같은 teacher dataset을 반복 사용하면 초기 생성 비용을 상각할 수 있다. 반대로 teacher나 augmentation을 바꾸면 대규모 dataset을 다시 만들어야 한다.

## Synthetic caption ablation

DFN에서 pretrain한 CoCa를 고품질 caption data로 fine-tune하면 retrieval이 회복된다.

| CoCa base | Fine-tune | Context | IN-val | Flickr30k | Avg. 38 |
| --- | --- | ---: | ---: | ---: | ---: |
| LAION-2B | MSCOCO-123K | 77 | 65.4 | 75.8 | 56.2 |
| DFN-2B | 없음 | 77 | 65.9 | 68.7 | 55.9 |
| DFN-2B | MSCOCO-123K | 77 | 65.9 | 76.0 | 56.2 |
| DFN-2B | MSCOCO-38K | 77 | 65.9 | 75.4 | 56.5 |
| DFN-2B | DOCCI | 77 | 66.3 | 72.6 | 57.3 |
| DFN-2B | 5 models x 2 captions | 77 | 65.9 | 74.7 | 56.3 |
| DFN-2B | 10 models x 1 caption | 77 | 66.0 | 75.1 | 56.5 |

caption model을 10개로 늘려 diversity를 키워도 best와 one standard deviation 안이다. context length 255의 긴 caption도 일관된 이득이 없다. 생성 caption 수가 많을수록 무조건 좋아지는 것이 아니며, 첫 1-2개에서 대부분의 효과가 포화된다는 MobileCLIP 관찰과 일치한다.

## Architecture family

### Image encoder

S0, S2는 4-stage FastViT hybrid encoder `MCi0`, `MCi2`를 쓴다. S3와 S4는 5번째 transformer stage를 추가하고 그 전에 한 축 2배씩, 면적 4배 downsampling한다. 큰 channel과 attention을 더 작은 grid에서 실행해 고해상도 scaling을 개선한다.

image representation은 최종 spatial feature에 global average pooling을 적용하고 linear projection해 만든다. 따라서 CLIP retrieval inference에서 output은 수백 visual token이 아니라 image당 embedding 하나다.

### Text encoder

S0, S2, B는 12-layer standard Base text transformer를 사용한다. S3와 S4는 더 큰 pure transformer text encoder를 사용한다. context length는 77이다. image encoder만 빠르게 만들어도 text prompt가 매번 바뀌면 text latency가 전체의 큰 비중이 될 수 있다. 실제 앱에서는 class prompt embedding cache가 중요하다.

### 5-stage latency scaling

125M parameter로 맞춘 MCi2-Scaled와 MCi3를 비교하면 256에서 MCi3가 1.9배, 1024에서는 7.1배 빠르다. 고해상도 dense prediction에서 큰 layer를 작은 feature map으로 옮기는 효과가 커진다.

## Tensor shape와 batch memory

retrieval inference의 핵심 shape는 다음과 같다.

```text
image: B x 3 x H x W
image encoder: B x d
text ids: B x 77
text encoder: B x d
normalized embeddings: B x d
similarity: B x B during training
```

global batch 65,536이면 naive full similarity matrix는 다음 element 수다.

```math
65536^2=4,294,967,296
```

BF16 한 장만 약 8GiB이고 image-to-text와 text-to-image loss, gradient까지 고려하면 훨씬 크다. 실제 학습은 256 A100-40GB 또는 128 H100-80GB에 분산하고 communication-aware chunking을 사용해야 한다. 이 계산은 reviewer estimate이며 논문은 similarity implementation의 peak memory를 보고하지 않는다.

inference batch 1에서는 similarity가 class 수 `C`에 대해 `1 x C`다. class embedding을 cache하면 `C x d` matrix-vector product만 남는다. 예를 들어 embedding dimension을 실제 checkpoint에서 `d`라 하고 class가 1,000개면 FP16 text cache는 `1000 x d x 2` bytes다. 논문 표는 variant별 projection dimension을 명시하지 않으므로 임의 숫자를 넣지 말고 config에서 읽어야 한다.

## Training recipe

| Hyperparameter | S0 | S2 | B | S3 | S4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| input | 256 | 256 | 224 | 256 | 256 |
| iterations | 200K | 200K | 200K | 200K | 200K |
| warmup | 10K | 10K | 2K | 2K | 2K |
| global batch | 65,536 | 65,536 | 65,536 | 114,688 | 114,688 |
| max LR | `1e-3` | `1e-3` | `1e-3` | `1e-3` | `1e-3` |
| min LR | `1e-6` | `1e-6` | `1e-6` | 0 | 0 |

공통 설정은 AdamW, beta `(0.9,0.95)`, cosine decay, weight decay 0.2, gradient clipping 1.0, BF16, RandAugment다. 200K step과 batch 65,536은 약 13.1B seen samples이고, S3/S4의 batch 114,688은 약 22.9B다. 본문 표는 model별 seen samples를 13B로 표기하므로 실제 released recipe와 checkpoint metadata를 함께 확인해야 한다.

## Training pseudocode

```python
def reinforced_clip_step(batch, student, teacher_cache):
    image, gt_caption, aug_id = batch

    # 저장된 augmentation parameter로 teacher 생성 때와 같은 view 재현
    image_view = replay_augmentation(image, aug_id)
    captions = [gt_caption] + teacher_cache.synthetic_captions

    student_img = normalize(student.image_encoder(image_view))
    total = 0.0

    for caption in captions:
        student_txt = normalize(student.text_encoder(caption))
        student_i2t = softmax(student_img @ student_txt.T * student.logit_scale)
        student_t2i = softmax(student_txt @ student_img.T * student.logit_scale)

        for teacher in teacher_cache.two_clip_teachers:
            teacher_i2t = teacher.cached_i2t_distribution(caption, aug_id)
            teacher_t2i = teacher.cached_t2i_distribution(caption, aug_id)
            total += kl(teacher_i2t, student_i2t)
            total += kl(teacher_t2i, student_t2i)

    return total / normalization
```

실제 cache에는 pair별 embedding이 저장되며 batch similarity distribution은 training batch가 구성된 뒤 계산된다. augmentation parameter를 정확히 replay하지 않으면 teacher image embedding과 student input view가 달라져 distillation target이 불일치한다.

## 개선 단계 ablation

MobileCLIP-B architecture를 12.8M subset에서 30K step, 5회 반복해 비교한다.

| 설정 | IN-val | Flickr30k | Avg. 38 |
| --- | ---: | ---: | ---: |
| MobileCLIP recipe | 61.6 | 72.8 | 53.5 |
| base를 DFN-5B로 변경 | 63.1 +- 0.2 | 73.3 +- 0.6 | 54.1 +- 0.4 |
| DFN teacher ensemble | 65.4 +- 0.4 | 75.8 +- 0.3 | 56.2 +- 0.6 |
| MobileCLIP2, DFN CoCa + MSCOCO-38K | 65.9 +- 0.3 | 75.4 +- 0.2 | 56.5 +- 0.3 |
| 10 captioner x 1 caption | 66.0 +- 0.1 | 75.1 +- 0.6 | 56.5 +- 0.3 |

가장 큰 추가 이득은 base dataset과 teacher 교체에서 온다. caption generator 개선의 평균 이득은 더 작다. 10개 captioner ensemble은 비용 대비 뚜렷한 추가 이득이 없다.

학습 효율 곡선에서 DFNDR-12M은 같은 ImageNet 정확도까지 DataComp-12M보다 5배, DFN-12M보다 3.3배, DataCompDR-12M보다 1.3배 적은 sample을 본다. full scale에서는 DFNDR-2B가 DataCompDR-1B보다 약 1.7배 sample-efficient하고 13B seen sample에서 ImageNet이 3.2 point 높다. teacher dataset 생성 비용은 이 배수에 포함되지 않는다.

## MobileCLIP2 family 결과

latency는 image encoder와 text encoder의 합이며 iPhone 12 Pro Max에서 측정된다.

| Model | Image params | Text params | Image ms | Text ms | Total ms | IN-val | Flickr T2I/I2T | Avg. 38 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileCLIP2-S0 | 11.4M | 63.4M | 1.5 | 3.3 | 4.8 | 71.5 | 69.2 / 86.6 | 59.7 |
| MobileCLIP2-S2 | 35.7M | 63.4M | 3.6 | 3.3 | 6.9 | 77.2 | 74.8 / 90.4 | 64.1 |
| MobileCLIP2-B | 86.3M | 63.4M | 10.4 | 3.3 | 13.7 | 79.4 | 76.5 / 89.7 | 65.8 |
| MobileCLIP2-S3 | 125.1M | 123.6M | 8.0 | 6.6 | 14.6 | 80.7 | 77.3 / 91.6 | 66.8 |
| MobileCLIP2-S4 | 321.6M | 123.6M | 19.6 | 6.6 | 26.2 | 81.9 | 78.0 / 92.4 | 67.5 |

S4는 SigLIP-SO400M/14의 ImageNet 82.0과 사실상 같은 수준이면서 total parameter가 `445.2M` 대 `877.4M`로 약 2배 작다. DFN ViT-L/14의 image latency 57.9ms와 비교하면 S4 19.6ms는 약 3배 빠르지만 논문은 total latency 기준의 대표 비교를 2.5배로 요약한다.

DFNDR 모델은 classification과 Avg. 38에 강하지만 retrieval에서 항상 최고는 아니다. DataCompDR로 같은 architecture를 학습한 MobileCLIP-S4는 Flickr30k 79.5/94.9, COCO 52.1/70.3, Avg. 38은 68.1로 DFNDR S4보다 retrieval과 평균이 높다. dataset filtering bias가 downstream task 선택에 직접 영향을 준다.

## VLM backbone으로 쓸 때

ViT-B/16 backbone을 frozen하고 Qwen2-7B와 LLaVA-1.5 recipe로 평가한다.

| Pretraining | GQA | SQA | TextVQA | POPE | MMMU | MMB | VizWiz | VQAv2 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DataComp-1B | 59.6 | 71.5 | 50.5 | 81.8 | 42.6 | 59.1 | 51.8 | 70.7 | 61.0 |
| DFN-2B | 56.9 | 71.3 | 46.0 | 81.4 | 41.9 | 52.2 | 56.1 | 66.9 | 59.1 |
| DataCompDR-1B | 60.3 | 73.1 | 50.4 | 81.7 | 43.6 | 60.2 | 54.9 | 72.1 | 62.0 |
| DFNDR-2B | 60.4 | 72.9 | 49.9 | 83.3 | 45.2 | 61.9 | 54.5 | 72.4 | 62.6 |

DFNDR는 DFN보다 3.5 point, DataComp보다 1.6, DataCompDR보다 0.6 높다. 그러나 이 실험의 phone latency, visual token 수, TTFT, peak memory는 보고되지 않았다. CLIP retrieval latency를 Qwen2-7B VLM latency로 잘못 인용하면 안 된다.

## Dense prediction transfer

- COCO Mask R-CNN, ViT-B/16: box AP 47.0, mask AP 41.8
- ADE20K UperNet, ViT-B/16: mIoU 52.8, mAcc 64.0
- NYUv2 depth, ViT-B/16: RMSE 0.356
- MCi0 supervised 대비 MobileCLIP2 pretraining: COCO box +2.6, mask +1.6
- MCi2 supervised 대비 MobileCLIP2 pretraining: COCO box +2.5, mask +1.5
- ADE20K SemanticFPN: MCi0 44.8에서 47.0, MCi2 48.9에서 51.6

즉 global image-text objective가 retrieval뿐 아니라 localization transfer에도 유용하다. 다만 dense head, input 512, fine-tuning schedule의 비용은 mobile CLIP latency 표와 다른 조건이다.

## On-device memory와 operator 분석

parameter만 raw weight로 환산한 reviewer 계산은 다음과 같다.

| Model | Total params | FP16 | INT8 |
| --- | ---: | ---: | ---: |
| S0 | 74.8M | 약 142.7 MiB | 약 71.3 MiB |
| S2 | 99.1M | 약 189.0 MiB | 약 94.5 MiB |
| B | 149.7M | 약 285.5 MiB | 약 142.8 MiB |
| S3 | 248.7M | 약 474.4 MiB | 약 237.2 MiB |
| S4 | 445.2M | 약 849.2 MiB | 약 424.6 MiB |

scale, packing, activation, runtime workspace는 제외했다. 논문은 deployed precision, peak RAM, model file size를 보고하지 않는다.

FastViT 계열은 convolution, depthwise convolution, residual, global average pooling, 후기 attention으로 구성된다. mobile compiler에서 다음을 확인해야 한다.

1. train-time reparameterization branch가 inference 전에 fuse되는가
2. text transformer의 context 77이 static shape로 compile되는가
3. image와 text embedding normalize가 accelerator에서 fuse되는가
4. cached text embedding을 매 frame 다시 계산하지 않는가
5. top-k search가 gallery 크기에 따라 CPU bottleneck이 되지 않는가
6. large gallery는 exact GEMM, ANN index, quantized embedding 중 무엇을 쓰는가

## 논문이 보고하지 않은 것

- iPhone 측정의 run count, warm-up, p50/p95
- CPU, GPU, Neural Engine 중 실제 compute unit
- model precision과 quantization
- peak memory, power, energy, temperature
- image preprocessing과 tokenizer latency
- large gallery retrieval latency
- sustained camera-frame throughput
- VLM으로 연결했을 때 phone TTFT와 tokens/s

따라서 표의 4.8-26.2ms는 encoder benchmark로 읽어야 하며 full application frame time으로 읽으면 안 된다.

## 강점

1. offline teacher cache로 대규모 ensemble distillation을 반복 가능한 dataset으로 만든다.
2. base data, teacher, captioner, temperature를 분리한 다수 ablation과 5-run 분산을 제공한다.
3. iPhone 12 Pro Max latency와 parameter를 정확도 표에 함께 둔다.
4. S0부터 S4까지 폭넓은 latency operating point를 제공한다.
5. retrieval, zero-shot classification, VLM, detection, segmentation, depth로 representation을 검증한다.
6. data generation code와 checkpoint를 공개한다.

## 한계와 비판적 해석

### 1. 162TB dataset은 재현 장벽이 높다

online step은 효율적이지만 reinforced dataset 생성과 저장은 대규모 infrastructure를 요구한다. 작은 연구실에서는 12.8M subset부터 재현하는 것이 현실적이다.

### 2. Teacher bias를 student가 상속한다

두 DFN teacher와 DFN filtering은 ImageNet형 classification에 유리하지만 retrieval에는 불리할 수 있다. English-centric web data의 언어 및 문화 bias도 남는다.

### 3. Caption diversity 효과가 제한적이다

10개 captioner를 동원해도 one standard deviation 안이다. 복잡한 caption generation pipeline의 marginal return이 작다.

### 4. Mobile latency 설명이 충분하지 않다

device 이름은 명시하지만 execution backend, precision, variance와 energy가 없다. architecture 선택에는 이 정보가 필요하다.

### 5. VLM 효율은 별도 검증되지 않았다

frozen backbone 품질은 좋아졌지만 Qwen2-7B를 phone에서 실행한 결과는 아니다. retrieval model과 generative VLM의 시스템 주장을 분리해야 한다.

## 재현 체크리스트

- [ ] DFN-2B sample manifest와 license 정책 확인
- [ ] 두 teacher checkpoint와 logit scale 60/70 고정
- [ ] CoCa DFN-2B pretraining 및 MSCOCO-38K fine-tuning 확인
- [ ] sample당 synthetic caption 5개 확인
- [ ] augmentation parameter 저장과 replay 일치
- [ ] 12.8M subset과 full dataset의 augmentation 수 차이 확인
- [ ] embedding dtype, shard format, checksum 기록
- [ ] OpenCLIP evaluation version과 prompt template 고정
- [ ] global batch와 distributed similarity 구현 검증
- [ ] iPhone model, OS, runtime, compute unit, precision 기록
- [ ] text embedding cache on/off를 분리 측정
- [ ] p50/p95, peak RAM, energy, thermal test 추가
- [ ] VLM transfer에서는 visual token, TTFT, decode throughput 별도 측정

## 추천 구현 로드맵

1. released S0와 S2 checkpoint로 image-text retrieval을 먼저 재현한다.
2. 1K class prompt를 미리 encode하고 cache한 뒤 frame latency를 측정한다.
3. 12.8M DFNDR subset을 생성해 teacher scale과 caption 수 ablation을 재현한다.
4. INT8 image encoder와 FP16 embedding head의 정확도 및 latency를 비교한다.
5. detector 또는 VLM의 shared vision encoder로 사용할 때 feature stage와 token output을 명시한다.
6. 마지막에 target phone에서 cold start, sustained camera, energy까지 평가한다.

## Retrieval application의 실제 계산 경로

MobileCLIP2를 camera search에 사용할 때 text encoder를 매 frame 실행할 이유가 없다. query vocabulary가 `C`개라면 startup이나 query 변경 시 다음 cache를 만든다.

```math
Q=\mathrm{normalize}(f_T(T))\in\mathbb{R}^{C\times d}
```

frame마다 image embedding 하나를 만들고 matrix-vector similarity를 계산한다.

```math
u=\mathrm{normalize}(f_I(x))\in\mathbb{R}^{d},
\qquad
s=Qu\in\mathbb{R}^{C}
```

end-to-end frame time은 다음처럼 분해해야 한다.

```math
T_{frame}=T_{decode}+T_{resize}+T_{image\ enc}+T_{similarity}+T_{topk}
```

논문 표의 total latency는 `T_image_enc + T_text_enc`이며 camera decode, resize, similarity와 top-k는 포함하지 않는다. 반대로 cached query 앱에서는 `T_text_enc`가 steady-state frame마다 발생하지 않는다. 따라서 S2의 표기 total 6.9ms보다 실제 encoder core는 image 3.6ms에 가깝지만, 앱 overhead가 추가된다.

gallery retrieval은 방향이 반대다. `M`개 image embedding을 미리 저장하고 한 text embedding을 `M x d` gallery와 비교한다. gallery가 커지면 encoder보다 approximate nearest-neighbor index와 memory bandwidth가 병목일 수 있다. FP16 exact gallery raw size는 `2Md` bytes이며 INT8 embedding은 대략 절반이다. dimension `d`는 checkpoint config에서 읽어야 하며 논문에 없는 숫자를 가정해서는 안 된다.

## Offline reinforcement의 총비용 회계

논문이 말하는 5배 training efficiency는 student가 target accuracy까지 보는 sample 수를 비교한다. 전체 project cost는 다음을 함께 세야 한다.

```math
C_{total}
=C_{filter}
+C_{CLIP\ teacher}
+C_{captioner}
+C_{storage}
+C_{student\ train}
```

teacher embedding은 student architecture 여러 개에서 재사용되므로 student 수가 늘수록 sample당 amortized cost가 낮아진다. 반대로 teacher ensemble, logit scale, augmentation recipe를 자주 바꾸면 cache invalidation 때문에 큰 비용이 반복된다. 실험 기록에는 다음을 남기는 편이 좋다.

- source image당 teacher forward 횟수
- captioner decode token과 beam setting
- generated shard byte와 compression ratio
- object storage read throughput과 cache hit
- student당 reused dataset 비율
- failed sample 및 regeneration 비율

162TB를 읽는 I/O 자체도 무시할 수 없다. 10GB/s의 aggregate sustained read라도 전체를 한 번 순차 읽는 데 이론적으로 약 4.5시간이 걸리고, 실제 random access와 network overhead에서는 더 길다. 이는 reviewer 계산이며 논문 실측값이 아니다. 대규모 shard shuffle은 compute utilization을 떨어뜨릴 수 있으므로 local cache와 prefetch를 함께 benchmark해야 한다.

## Distillation batch의 통신 병목

global batch가 65,536이면 각 device의 local embedding만으로는 full in-batch negative를 만들 수 없다. distributed all-gather 또는 ring 방식으로 image/text embedding을 교환한다. communication byte는 embedding dimension과 device 수에 비례하고, full similarity는 batch square에 비례한다.

정확한 재현에서는 다음 두 구현을 구분해야 한다.

1. 모든 embedding을 all-gather하고 local row의 logits만 계산
2. chunked/ring similarity로 embedding을 순환하며 loss 누적

둘은 수학적으로 같을 수 있지만 peak memory, communication overlap, numerical reduction order가 다르다. teacher가 offline이라는 사실은 이 student-to-student distributed communication을 없애지 않는다. 12.8M ablation을 작은 batch로 축소하면 negative pool이 달라져 논문 accuracy를 그대로 기대하기 어렵다.

## Quantization 실험 설계

CLIP은 최종 embedding의 angular geometry가 중요하다. INT8 weight만 평가하지 말고 다음을 분리한다.

- image encoder INT8, projection과 L2 normalize FP16
- text encoder INT8, prompt cache FP16
- image와 text 모두 INT8, similarity FP16 accumulate
- embedding 자체 INT8 및 scale-aware dot product

zero-shot classification은 top-1뿐 아니라 logit margin과 calibration을, retrieval은 Recall@1/5/10을 기록한다. image와 text encoder의 quantization error가 서로 다른 방향으로 embedding space를 회전시키면 individual encoder feature error가 작아도 cross-modal alignment가 크게 나빠질 수 있다. paired calibration data를 사용하고 original FP32/FP16 embedding과 cosine drift를 측정해야 한다.

## App별 model 선택

| 사용 사례 | 우선 선택 | 이유 |
| --- | --- | --- |
| 항상 켜진 20-class camera classifier | S0 또는 S2 | image latency와 weight가 작고 text cache 가능 |
| 일반 image-text search | S2 또는 B | retrieval과 memory의 균형 |
| offline photo indexing, 충전 중 실행 | S3 또는 S4 | latency보다 representation 품질 우선 가능 |
| detection/segmentation backbone | MCi2/MCi3 feature stage | hierarchy와 high-resolution scaling 활용 |
| generative VLM | 별도 TTFT 검증 후 선택 | CLIP encoder latency만으로 VLM 효율 판단 불가 |

한 모델의 ImageNet 수치로 모든 app을 고르면 안 된다. DFNDR는 classification에 강하고 DataCompDR variant는 retrieval에서 더 좋은 경우가 있으므로 target validation을 먼저 고정해야 한다.

## 최종 평가

MobileCLIP2의 가장 중요한 교훈은 작은 architecture만으로 mobile foundation model이 만들어지지 않는다는 것이다. 강한 teacher의 similarity structure와 synthetic caption을 미리 저장하면 student는 매 step teacher를 실행하지 않고도 고품질 supervision을 반복해서 사용할 수 있다. 그 결과 S2는 iPhone 12 Pro Max 기준 image와 text 합 6.9ms에서 ImageNet 77.2를 기록하고, S4는 26.2ms에서 81.9를 기록한다.

대신 162TB reinforced dataset과 teacher bias라는 비용이 있다. 또한 iPhone latency의 backend, precision, peak memory와 전력이 충분히 공개되지 않았다. 따라서 온디바이스 연구에서는 `좋은 accuracy-latency curve`와 `비싼 offline data factory`를 함께 평가해야 한다. retrieval과 cached text query가 중심인 앱에는 매우 강한 baseline이며, generative VLM으로 확장할 때는 visual token과 TTFT를 새로 측정해야 한다.

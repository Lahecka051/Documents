# 49. Fully Convolutional Networks for Semantic Segmentation

## 논문 정보

- 제목: **Fully Convolutional Networks for Semantic Segmentation**
- 저자: Jonathan Long, Evan Shelhamer, Trevor Darrell
- 소속: UC Berkeley
- 공개: arXiv:1411.4038v2, 2015-03-08
- 발표: CVPR 2015
- 논문: [arXiv](https://arxiv.org/abs/1411.4038) / [PDF](https://arxiv.org/pdf/1411.4038)
- 원본 파일: `49_FCN.pdf`
- 구현/모델: [FCN project repository](https://github.com/shelhamer/fcn.berkeleyvision.org)
- 핵심 키워드: semantic segmentation, fully convolutional network, dense prediction, transposed convolution, skip fusion, whole-image training

## 한눈에 보는 요약

FCN은 image classifier를 patch마다 반복 호출하지 않고 **한 번의 fully convolutional forward로 모든 pixel의 class를 예측**할 수 있음을 체계화한 논문이다.

```text
classification CNN
 -> fully connected layer를 convolution으로 변환
 -> coarse spatial class score map
 -> in-network upsampling
 -> shallow fine feature와 deep semantic feature를 skip fusion
 -> input 크기의 semantic segmentation
```

핵심 구조는 세 variant로 발전한다.

- `FCN-32s`: 가장 깊은 stride-32 score를 한 번에 32배 upsample
- `FCN-16s`: stride-32 score를 2배 올려 pool4의 stride-16 score와 합침
- `FCN-8s`: 다시 2배 올려 pool3의 stride-8 score와 합침

FCN-8s는 PASCAL VOC 2012 test에서 mean IoU 62.2를 기록하고, 500×500 image inference를 약 175 ms에 수행한다. 이전 SDS의 mean IoU 51.6, 약 50초와 비교해 accuracy와 speed를 동시에 개선했다.

오늘날 관점에서 FCN-VGG16의 134M parameter와 coarse boundary는 무겁고 낡았지만, **classification backbone의 dense 변환, learned upsampling, multi-resolution skip connection, end-to-end pixel loss**라는 설계는 U-Net, DeepLab, FPN, SegFormer까지 이어진다.

## 이전 patchwise segmentation의 문제

초기 CNN segmentation은 pixel마다 주변 patch를 잘라 classifier에 넣었다.

```text
pixel 1 patch -> CNN
pixel 2 patch -> same CNN
pixel 3 patch -> same CNN
...
```

인접 pixel의 receptive field가 거의 겹치므로 같은 convolution을 반복한다. 큰 patch는 context가 좋지만 pooling으로 localization이 거칠어지고, 작은 patch는 위치는 정밀하지만 전체 object를 보지 못한다. Superpixel, proposal, random field와 별도 후처리도 pipeline을 복잡하게 했다.

FCN은 entire image를 한 번에 처리해 overlapping computation을 공유한다. 논문 예시에서 AlexNet은 227×227 image 한 장의 score에 1.2 ms가 걸리고, fully convolutional version은 500×500 image에서 10×10 output grid를 22 ms에 만든다. 독립 patch 100개를 효율적으로 batch 처리하는 것보다 5배 이상 빠르다고 보고한다.

## Fully convolutional network의 정의

Layer의 feature를 다음과 같이 둔다.

```math
x\in\mathbb{R}^{h\times w\times d}
```

Kernel size `k`, stride `s`인 local layer는 위치 `(i,j)`에서 다음 형태로 계산된다.

```math
y_{ij}=f_{ks}\left(
\{x_{si+\delta_i,sj+\delta_j}\}_{0\leq\delta_i,\delta_j\leq k}
\right)
```

`f`는 convolution, pooling, elementwise nonlinearity가 될 수 있다. 이런 local, translation-structured layer의 합성도 같은 형태를 유지한다.

```math
f_{ks}\circ g_{k's'}
=(f\circ g)_{k'+(k-1)s',\;ss'}
```

따라서 convolution과 pooling만으로 구성된 network는 입력 크기가 달라도 대응하는 spatial output을 낼 수 있다. 논문은 이를 nonlinear deep filter로 해석한다.

## Fully connected layer를 convolution으로 바꾸기

Classifier의 fully connected layer도 바로 앞 feature map 전체를 덮는 convolution으로 볼 수 있다.

예를 들어 input feature가 `7×7×512`, FC output이 4096이면 다음 두 계산은 같다.

```math
y=Wx+b
```

```math
Y=\operatorname{Conv}_{7\times7}(X),
\qquad C_{out}=4096
```

원 classifier는 고정 크기 input에서 spatial output 1개를 내지만, convolution으로 바꾼 network는 더 큰 input에서 score heatmap을 낸다. 같은 weight를 모든 위치에 적용하므로 overlapping patch computation이 공유된다.

이 변환은 parameter를 없애는 것이 아니다. VGG16의 `fc6`를 7×7 convolution으로 옮기면 weight 수는 reviewer 계산으로 다음과 같다.

```math
7\times7\times512\times4096
=102{,}760{,}448
```

이 layer 하나만 FP32 약 392 MiB다. FCN이 계산 방식은 효율적으로 만들었지만 원 VGG classifier의 큰 parameter burden은 그대로 물려받았다는 뜻이다.

## Spatial loss와 whole-image training

최종 score map의 각 위치에서 loss를 계산하고 합한다.

```math
\mathcal{L}(x;\theta)=\sum_{i,j}\mathcal{L}'(x_{ij};\theta)
```

Gradient도 각 spatial loss gradient의 합이다.

```math
\nabla_\theta\mathcal{L}
=\sum_{i,j}\nabla_\theta\mathcal{L}'_{ij}
```

따라서 whole-image FCN training은 output grid의 모든 receptive field patch를 한 minibatch처럼 학습하는 것과 같다. 차이는 convolution을 layer 단위로 한 번만 계산해 중복을 없앤다는 점이다.

Ground truth가 ambiguous/difficult로 표시된 pixel은 loss에서 무시한다. Classification은 per-pixel multinomial logistic loss를 사용한다.

## Coarse output 문제

VGG16 classifier는 pooling을 거치며 total stride 32가 된다. Fully convolutionalization만 하면 입력의 1/32 해상도 score map이 나온다.

```text
input 512×512
 -> stride 8:  64×64
 -> stride 16: 32×32
 -> stride 32: 16×16
```

이를 바로 32배 확대하면 object category는 맞아도 boundary가 거칠다. Semantic information은 깊은 layer에 강하고, edge와 위치 정보는 얕은 layer에 강하다는 `what vs where` tension이 생긴다.

## Shift-and-stitch와 filter rarefaction

Stride가 `f`인 network에 input을 `(x,y)`만큼 이동해 `f²`번 forward하고 coarse output을 interlace하면 dense prediction을 만들 수 있다. 논문은 이것이 stride를 줄이고 이후 filter에 0을 삽입해 rarefy하는 것과 동등하다고 설명한다.

원 filter `f_ij`를 stride `s`에 맞춰 확장하면 다음과 같다.

```math
f'_{ij}=
\begin{cases}
f_{i/s,j/s}, & s\text{가 }i,j\text{를 모두 나눌 때}\cr
0, & \text{그 외}
\end{cases}
```

이 방식은 receptive field를 유지하며 output을 촘촘하게 만들지만 계산량이 늘고, 원 filter가 더 fine한 input interaction을 새로 배우지는 못한다. FCN은 제한된 실험에서 skip fusion과 learned upsampling이 더 효율적이었다고 보고 shift-and-stitch를 최종 model에 쓰지 않는다.

## Upsampling과 transposed convolution

논문은 integer factor `f` upsampling을 input stride `1/f`인 convolution처럼 보고, convolution의 forward/backward 관계를 뒤집은 `backwards convolution`, 오늘날 transposed convolution이라고 부르는 layer로 구현한다.

```math
Y=\operatorname{ConvTranspose}(X;W,\text{stride}=f)
```

여기서 `deconvolution`이라는 옛 명칭은 convolution의 정확한 수학적 inverse라는 뜻이 아니다. Input 위치가 output patch에 weight를 scatter-add하는 linear operator다.

- Final upsampling filter: bilinear interpolation으로 고정
- Intermediate 2× upsampling: bilinear로 초기화한 뒤 학습

Pixelwise loss에서 gradient가 upsampling weight까지 전달되므로 별도 interpolation/post-processing 없이 end-to-end 학습할 수 있다.

## FCN-32s, FCN-16s, FCN-8s

### FCN-32s

Convolutionalized `conv7` 위에 class 수만큼 1×1 score convolution을 붙이고 32배 upsample한다.

```math
S_{32}\in\mathbb{R}^{B\times K\times H/32\times W/32}
```

### FCN-16s

Pool4에도 1×1 score layer를 붙인다. Deep score를 2배 upsample하고 pool4 score와 elementwise sum한다.

```math
S_{16}=\operatorname{Score}(P_4)
+\operatorname{Up}_2(S_{32})
```

그 뒤 16배 upsample해 input 크기로 복원한다. 새 pool4 score parameter는 0으로 초기화하고 FCN-32s weight에서 시작하므로 초기 output이 바뀌지 않는다. Learning rate는 100배 낮춘다.

### FCN-8s

Pool3 score를 한 번 더 합친다.

```math
S_{8}=\operatorname{Score}(P_3)
+\operatorname{Up}_2(S_{16})
```

최종 8배 upsample로 pixel score를 만든다. Max fusion은 gradient switching 때문에 학습이 어려워 sum fusion을 사용한다.

## Batch 1 tensor shape 계산 예

아래는 설명을 위한 **reviewer 계산**이다. `B=1`, input `512×512`, `K=21`, VGG stride와 same-padding을 단순 가정한다. 원 Caffe FCN은 padding, crop offset, arbitrary input 때문에 exact spatial size가 다를 수 있다.

| Tensor | Conceptual shape | FP32 payload |
| --- | ---: | ---: |
| pool3 | `1×256×64×64` | 4.00 MiB |
| pool4 | `1×512×32×32` | 2.00 MiB |
| pool5 | `1×512×16×16` | 0.50 MiB |
| score32 | `1×21×16×16` | 0.021 MiB |
| score16 | `1×21×32×32` | 0.082 MiB |
| score8 | `1×21×64×64` | 0.328 MiB |
| final logits | `1×21×512×512` | **21.0 MiB** |

Pool3/4 skip feature payload만 최소 6 MiB이며 decoder가 사용할 때까지 보존해야 한다. 최종 full-resolution FP32 logits은 21 MiB라서 class 수와 해상도가 커지면 output 자체가 peak memory에 크게 기여한다. FP16은 payload가 절반이지만 softmax가 FP32로 승격되거나 runtime workspace가 추가될 수 있다.

실제 FCN-VGG16의 conv7은 4096 channel이므로 coarse spatial map이어도 activation이 크다. Modern decoder가 256-channel bottleneck을 쓰는 이유를 보여준다.

## Crop과 alignment

Skip sum을 하려면 두 score map의 spatial position이 같은 input pixel을 가리켜야 한다. Stride, padding, valid convolution, transposed-conv kernel 때문에 shape만 같아도 receptive-field center가 어긋날 수 있다.

```text
upsampled deep score
 -> crop/alignment
 -> shallow score와 동일 H×W 및 동일 pixel center 확인
 -> sum
```

논문의 network는 필요한 위치를 crop해 맞춘다. 구현에서 단순 resize 후 같은 index끼리 더하면 1-몇 pixel shift가 생길 수 있으므로 impulse input이나 coordinate grid로 alignment를 unit test해야 한다.

## 해상도에 따른 activation scaling

Fully convolutional이라는 말은 input 크기에 관계없이 **같은 weight를 적용할 수 있다**는 뜻이지 memory가 크기에 무관하다는 뜻이 아니다. Spatial feature와 final logits는 pixel 수에 선형으로 증가한다.

```math
M_{logit}=BKHWD_{bytes}
```

COCO-Stuff처럼 `K=171`, FP16, batch 1이라고 가정하면 final logits payload는 해상도별로 다음과 같다. 이는 reviewer 계산이며 원 논문의 PASCAL 21-class setting이 아니다.

| Resolution | FP16 final logits |
| --- | ---: |
| 320×320 | 33.4 MiB |
| 512×512 | 85.5 MiB |
| 1024×1024 | 342.0 MiB |

Class가 많고 해상도가 높으면 backbone보다 output logits가 peak를 만들 수 있다. Tile inference는 이를 줄이지만 overlap halo와 tile seam 처리가 필요하다. Argmax를 tile별 device-side로 수행하면 host transfer는 줄지만, global softmax probability나 uncertainty map이 필요하면 full logits를 유지해야 한다.

Skip level을 stride 16에서 8로 추가할 때 score map 자체는 `K` channel이라 작지만, pool3 feature를 decoder 사용 시점까지 보관해야 한다. 위 512 example에서 pool3 FP32 4 MiB가 추가되며, backward training에서는 gradient와 saved activation 때문에 훨씬 커진다. 따라서 `FCN-16s -> FCN-8s`의 +0.3 mIoU가 target memory에서 가치가 있는지 별도 판단해야 한다.

## Parameter와 MAC를 혼동하지 않기

FCN-VGG16의 134M parameter는 weight memory를 설명하지만 input resolution별 compute는 설명하지 않는다. 같은 7×7 conv6 weight도 output spatial 크기만큼 반복된다. 단순화한 `H_o×W_o` output에서 conv6 MAC는 다음과 같다.

```math
\operatorname{MAC}_{conv6}
=H_oW_o(7\times7\times512)4096
```

Reviewer 예시로 `H_o=W_o=16`이면 약 26.3 billion MAC이다. 하지만 원 Caffe FCN의 actual `H_o,W_o`는 padding/crop과 input size에 따라 달라지므로 이 숫자를 논문 보고 연산량으로 인용하면 안 된다. 논문 자체는 total MACs를 제공하지 않는다.

이 예시는 fully connected를 convolution으로 바꾸면 parameter는 같아도 큰 input에서 spatial reuse 횟수가 늘어 compute가 커진다는 점을 보여준다. Arbitrary-size 지원과 cheap high-resolution inference는 같은 개념이 아니다.

## Model 선택과 parameter

논문 Table 1의 classifier 변환 결과는 다음과 같다. Timing은 500×500 input, NVIDIA Tesla K40c, 20회 평균이다.

| Backbone FCN | Mean IU | Forward | Conv layers | Params | Receptive field | Max stride |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AlexNet | 39.8 | 50 ms | 8 | 57M | 355 | 32 |
| VGG16 | **56.0** | 210 ms | 16 | 134M | 404 | 32 |
| GoogLeNet | 42.5 | 59 ms | 22 | 6M | 907 | 32 |

Classification accuracy가 비슷해도 segmentation transfer 결과는 크게 달랐다. 큰 receptive field만으로 localization과 dense representation quality가 보장되지 않는다.

FCN-VGG16 134M parameter의 순수 weight payload를 reviewer가 계산하면 FP32 약 511.2 MiB, FP16 약 255.6 MiB, INT8 약 127.8 MiB다. 논문은 MACs/FLOPs와 peak activation을 보고하지 않는다. Parameter만으로도 오늘날 mobile 상시 segmenter에는 매우 무겁다.

## Skip fusion ablation

Extra PASCAL training annotation을 사용한 VOC2011 validation subset 결과다.

| Model | Pixel acc | Mean acc | Mean IU | Frequency-weighted IU |
| --- | ---: | ---: | ---: | ---: |
| FCN-32s-fixed | 83.0 | 59.7 | 45.4 | 72.0 |
| FCN-32s | 89.1 | 73.3 | 59.4 | 81.4 |
| FCN-16s | 90.0 | 75.7 | 62.4 | 83.0 |
| FCN-8s | **90.3** | **75.9** | **62.7** | **83.2** |

전체 network를 fine-tune한 32s가 final classifier만 학습한 32s-fixed보다 mean IU가 14.0 높다. 32s에서 16s로는 +3.0, 16s에서 8s로는 +0.3으로 diminishing return이 나타난다. 논문은 더 낮은 layer fusion을 계속하지 않는다.

## Segmentation metric

`n_ij`를 true class `i`인 pixel 중 class `j`로 예측한 수, `t_i=sum_j n_ij`로 둔다.

Pixel accuracy:

```math
\frac{\sum_i n_{ii}}{\sum_i t_i}
```

Mean class accuracy:

```math
\frac{1}{n_{cl}}\sum_i\frac{n_{ii}}{t_i}
```

Mean IoU:

```math
\frac{1}{n_{cl}}
\sum_i
\frac{n_{ii}}
{t_i+\sum_jn_{ji}-n_{ii}}
```

Frequency-weighted IoU:

```math
\frac{1}{\sum_kt_k}
\sum_i
\frac{t_in_{ii}}
{t_i+\sum_jn_{ji}-n_{ii}}
```

Pixel accuracy는 background가 많은 dataset에서 높게 나올 수 있고, frequency-weighted IoU도 frequent class 영향이 크다. Mean IoU는 class를 균등 평균하지만 fine boundary quality를 충분히 반영하지 못한다.

논문 Appendix A는 ground truth를 downsample-upsample해 stride별 mIoU 상한을 근사한다.

| Downsample factor | Mean IU upper bound approximation |
| ---: | ---: |
| 128 | 50.9 |
| 64 | 73.3 |
| 32 | 86.1 |
| 16 | 92.8 |
| 8 | 96.4 |
| 4 | 98.5 |

Stride 32의 coarse map도 theoretical mean IU 86.1이 가능하다. 따라서 높은 mIoU가 pixel-perfect boundary를 의미하지 않는다. Boundary F-score나 trimap IoU를 함께 보는 것이 좋다.

## Training과 fine-tuning

- Optimizer: SGD with momentum 0.9
- Batch: 20 images
- Fixed learning rate: AlexNet `1e-3`, VGG16 `1e-4`, GoogLeNet `5e-5`
- Weight decay: `5e-4` 또는 `2e-4`
- Bias learning rate: weight의 2배
- Class score convolution: zero initialization
- Original classifier에 dropout이 있으면 유지
- Fine-tune all layers
- Framework/hardware: Caffe, single Tesla K40c

FCN-32s VGG fine-tuning은 약 3일, FCN-16s와 FCN-8s upgrade는 각각 약 1일 걸렸다. Scratch training은 classifier representation을 배우는 시간이 커서 시도하지 않았고 ImageNet pretrained network에서 시작한다.

## Whole-image와 patch sampling ablation

논문은 output spatial loss 중 일부만 확률 `p`로 유지해 patch sampling을 모사한다. Effective batch size를 맞추기 위해 image 수를 `1/p`배 늘린다.

```math
m_{ij}\sim\operatorname{Bernoulli}(p),
\qquad
\mathcal{L}_{sample}=\sum_{i,j}m_{ij}\mathcal{L}'_{ij}
```

Iteration 수 기준 convergence에는 큰 차이가 없지만 더 많은 image를 처리해야 해 wall time이 늘었다. 따라서 unsampled whole-image training을 채택한다. 이 결과는 모든 class-imbalanced dataset에서 sampling이 불필요하다는 뜻은 아니다. PASCAL label의 약 3/4이 background였지만 저자들의 setting에서는 class balancing도 필요하지 않았다.

Random mirroring과 ±32 pixel translation jitter도 뚜렷한 개선이 없었다. Modern strong augmentation과 동일한 실험은 아니다.

## 주요 결과

### PASCAL VOC

| Method | VOC2011 test mean IU | VOC2012 test mean IU | Inference |
| --- | ---: | ---: | ---: |
| R-CNN | 47.9 | - | - |
| SDS | 52.6 | 51.6 | 약 50 s |
| FCN-8s | **62.7** | **62.2** | 약 **175 ms** |

논문의 20% improvement는 VOC2012에서 `(62.2-51.6)/51.6`, 약 20.5%의 상대 향상이다. 10.6 percentage point 향상과 구분해야 한다. SDS 대비 convnet-only speedup은 114배, 전체 pipeline은 286배로 서술한다.

### NYUDv2

| Model | Pixel acc | Mean acc | Mean IU | FW IU |
| --- | ---: | ---: | ---: | ---: |
| FCN-32s RGB | 60.0 | 42.2 | 29.2 | 43.9 |
| FCN-32s RGBD early fusion | 61.5 | 42.4 | 30.5 | 45.5 |
| FCN-32s HHA | 57.1 | 35.2 | 24.2 | 40.4 |
| FCN-32s RGB-HHA late fusion | 64.3 | 44.9 | 32.8 | 48.0 |
| FCN-16s RGB-HHA | **65.4** | **46.1** | **34.0** | **49.5** |

Raw depth를 input channel로 붙인 early fusion보다 RGB와 HHA network의 prediction을 합친 late fusion이 좋다. Multi-modal input은 단순 channel concat만으로 충분하지 않을 수 있음을 보여준다.

### SIFT Flow

FCN-16s two-head model은 semantic과 geometric label을 동시에 예측한다. Pixel accuracy 85.2, mean accuracy 51.7, mean IU 39.5, frequency-weighted IU 76.1, geometry accuracy 94.3을 기록한다. 두 task를 별도 model로 학습하는 것과 비슷한 성능을 내면서 shared computation으로 학습과 inference 비용을 거의 한 model 수준으로 유지한다.

### PASCAL-Context

59-class task에서 FCN-32s/16s/8s mean IU는 각각 31.8/34.8/35.1이다. 16s에서 8s의 gain이 0.3으로 VOC와 같은 diminishing return을 보인다.

## 구현 pseudocode

```python
def fcn8s(image):
    pool3, pool4, pool5 = vgg16_encoder(image)
    conv6 = relu(conv7x7(pool5, out_channels=4096))
    conv7 = relu(conv1x1(conv6, out_channels=4096))

    score32 = conv1x1(conv7, out_channels=num_classes)

    up32_to16 = transposed_conv(score32, scale=2, init="bilinear")
    score_pool4 = conv1x1(pool4, out_channels=num_classes, init="zeros")
    score16 = align_and_crop(up32_to16, score_pool4) + score_pool4

    up16_to8 = transposed_conv(score16, scale=2, init="bilinear")
    score_pool3 = conv1x1(pool3, out_channels=num_classes, init="zeros")
    score8 = align_and_crop(up16_to8, score_pool3) + score_pool3

    logits = transposed_conv(score8, scale=8, init="bilinear", trainable=False)
    logits = crop_to_input(logits, image.shape[-2:])
    return logits
```

Loss는 ignore mask를 정확히 적용한다.

```python
loss_map = cross_entropy(logits, labels, reduction="none")
loss = loss_map[valid_and_not_difficult].sum()
```

Original Caffe code와 modern framework는 transposed-conv output padding convention이 다를 수 있으므로 checkpoint conversion 때 crop offset까지 검증해야 한다.

## 장점

- Image classifier를 dense predictor로 바꾸는 일반 원리를 명확히 정리했다.
- Patchwise 중복을 없애 학습과 inference를 크게 단순화했다.
- Learned in-network upsampling으로 pixel loss를 backbone까지 end-to-end 전달했다.
- `what`과 `where`를 결합하는 skip fusion을 정량 ablation으로 보여줬다.
- PASCAL, RGB-D, scene parsing, multi-task까지 같은 framework로 처리했다.
- Pixel accuracy와 여러 IoU metric을 모두 보고 metric 해석을 돕는다.

## 한계와 비판적 관점

### 1. VGG16 parameter가 지나치게 크다

134M parameter 대부분이 convolutionalized fc layer에 있다. Mobile memory뿐 아니라 DRAM bandwidth와 load time에 불리하다.

### 2. Boundary가 여전히 coarse하다

FCN-8s도 stride-8 shallow score까지만 결합하며 low-level edge를 직접 복원하지 않는다. Bilinear-like upsampling으로 가는 구조라 얇은 물체와 sharp boundary가 손실될 수 있다.

### 3. Transposed convolution과 crop의 배포 복잡성

Backend별 output padding과 coordinate convention이 다르고, dynamic crop이 graph compiler에 부담이 될 수 있다. Checkerboard artifact 가능성도 별도 분석하지 않는다.

### 4. 오래된 hardware timing

K40c의 500×500 latency를 현대 GPU/NPU 결과와 직접 비교할 수 없다. Batch, library, precision도 오늘날 setting과 다르다.

### 5. MACs와 peak memory가 없다

Parameter, forward time은 있지만 total operations, activation peak, power는 보고하지 않는다.

### 6. Fine boundary metric 부재

Appendix가 mIoU의 coarse-output 관용성을 지적하지만 boundary F-score를 정식 평가하지 않는다.

### 7. Pretraining 의존

Scratch training을 충분히 비교하지 않고 ImageNet classifier에 크게 의존한다. Biomedical이나 다른 domain에서 같은 transfer gain이 보장되지 않는다.

## 온디바이스 관점

FCN의 whole-image computation은 patch classifier보다 압도적으로 효율적이지만 원형 FCN-VGG16은 mobile model이 아니다. 주요 병목은 다음과 같다.

- 134M weight, 특히 7×7×512×4096 conv6
- Decoder가 쓸 pool3/pool4 activation retention
- Full-resolution class logits
- Transposed convolution 지원과 workspace
- Crop/alignment operator

Operator 관점에서는 Conv, ReLU, MaxPool, 1×1 score, add, upsample, softmax/argmax로 단순하다. 그러나 NPU에서 ConvTranspose가 느리면 다음처럼 바꿀 수 있다.

```text
ConvTranspose
 -> fixed bilinear Resize
 -> 3×3 또는 depthwise-separable refinement Conv
```

이는 원 checkpoint와 bit-exact 동등하지 않으므로 fine-tuning과 mIoU 재검증이 필요하다. 또 runtime이 final argmax를 device에서 수행하면 `K×H×W` logits를 host로 옮기지 않고 `H×W` label만 전송할 수 있다.

Modern on-device baseline으로 재구성할 때는 VGG16 대신 MobileNet/EfficientNet 계열 encoder, 128-256 channel decoder, output stride 16/8을 비교하는 것이 현실적이다. 그래도 FCN의 핵심 ablation인 `32s -> 16s -> 8s`는 activation-memory와 boundary-quality trade-off 실험으로 그대로 재현할 가치가 있다.

## 재현 체크리스트

- [ ] arXiv v2와 VOC corrected split을 사용했는가?
- [ ] VOC val과 추가 Hariharan annotation의 overlap을 제거했는가?
- [ ] VGG16 fully connected layer를 동일 weight의 convolution으로 변환했는가?
- [ ] Final class score layer를 zero-initialize했는가?
- [ ] FCN-16s/8s가 이전 coarse checkpoint에서 시작하는가?
- [ ] Intermediate upsampling은 bilinear init 후 trainable인가?
- [ ] Final upsampling filter는 fixed bilinear인가?
- [ ] Pool4/pool3 score와 upsampled score의 pixel center가 정렬되는가?
- [ ] Ignore/difficult pixel을 loss와 metric에서 제외했는가?
- [ ] Whole-image loss reduction이 원 구현과 같은 sum/normalization인가?
- [ ] Mean IoU에 background를 포함하는가?
- [ ] 500×500 timing과 실제 deployment resolution을 구분했는가?
- [ ] FP16/INT8 weight뿐 아니라 skip activation peak를 측정했는가?
- [ ] ConvTranspose가 target accelerator에 native로 배치되는가?
- [ ] Boundary metric과 mIoU를 함께 기록했는가?

## 로드맵에서의 연결

FCN은 segmentation 단계의 출발점이다.

```text
FCN: classifier를 dense predictor로 변환, additive skip
 -> U-Net: 대칭 decoder와 feature concatenation
 -> DeepLabv3+: dilated encoder + boundary-refining decoder
 -> FPN/Mask R-CNN: multi-scale hierarchy를 detection/instance mask에 확장
 -> SegFormer: hierarchical Transformer + lightweight MLP decoder
 -> Mask2Former: mask query 기반 universal segmentation
```

FCN review에서 손으로 계산할 핵심은 output stride가 32/16/8로 줄 때 skip activation과 final logits memory가 어떻게 바뀌는지다. 동일 mobile backbone에서 세 decoder를 비교하면 architecture 역사와 on-device constraint를 직접 연결할 수 있다.

## 최종 평가

FCN의 가장 큰 공헌은 특정 VGG16 score보다 **semantic segmentation을 whole-image, end-to-end dense prediction 문제로 재정의한 것**이다. Fully connected layer의 convolution 해석, transposed-conv upsampling, shallow-deep skip fusion은 이후 거의 모든 encoder-decoder segmentation model의 기본 문법이 됐다.

원형 model은 134M parameter, 큰 conv6, coarse stride와 old K40 timing 때문에 오늘날 온디바이스에 직접 쓰기 어렵다. 그러나 FCN-32s/16s/8s ablation은 여전히 중요한 설계 원리를 보여준다. 해상도를 높이는 것은 단순히 output을 확대하는 일이 아니라 **semantic context, spatial alignment, activation lifetime, boundary metric을 함께 조정하는 문제**다.

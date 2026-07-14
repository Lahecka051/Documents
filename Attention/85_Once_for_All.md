# 85. Once-for-All: Train One Network and Specialize It for Efficient Deployment

## 논문 정보

- 제목: Once-for-All: Train One Network and Specialize It for Efficient Deployment
- 저자: Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
- 소속: MIT, MIT-IBM Watson AI Lab
- 발표: ICLR 2020
- arXiv: [1908.09791](https://arxiv.org/abs/1908.09791)
- 코드: [mit-han-lab/once-for-all](https://github.com/mit-han-lab/once-for-all)
- 원본 파일: `85_Once_for_All.pdf`

## 한눈에 보는 요약

Once-for-All, OFA는 device와 latency constraint마다 architecture를 찾고 처음부터 다시 학습하는 hardware-aware NAS의 반복 비용을 줄이는 방법이다. 하나의 큰 supernet을 한 번 학습하고, 그 안에서 depth, width, kernel size, input resolution이 다른 sub-network를 골라 target device에 배포한다.

핵심은 training과 specialization을 분리하는 것이다.

- training stage: 공유 weight를 가진 OFA supernet을 progressive shrinking으로 한 번 학습
- specialization stage: target hardware의 latency predictor와 accuracy predictor로 constraint를 만족하는 sub-network를 search
- optional fine-tuning: 선택한 sub-network를 25 또는 75 epoch 추가 학습할 수 있으나 direct deployment에는 필수가 아님

논문은 약 $2\times10^{19}$개의 architecture와 25개 input resolution을 하나의 7.7M parameter supernet에 공유한다. 작은 network를 처음부터 동시에 학습하지 않고, 가장 큰 network를 먼저 학습한 뒤 kernel, depth, width를 순서대로 elastic하게 만드는 progressive shrinking을 제안한다.

대표 결과는 다음과 같다.

- Pixel1에서 추가 fine-tuning 없이 230M MAC, 58 ms, ImageNet top-1 76.0%
- 595M MAC mobile setting에서 top-1 80.0%
- Pixel1 143 ms에서 top-1 80.1%, EfficientNet-B2보다 2.6배 빠르며 0.3% 높음
- deployment scenario 40개 기준 OFA의 총 design cost는 1.2k GPU hours, CO2e 0.34k lb, AWS \$3.7k로 계산됨
- OFA supernet training은 32 V100에서 full model 180 epoch 후 progressive shrinking을 포함해 총 약 1,200 GPU hours

가장 중요한 해석은 다음과 같다.

> OFA가 줄이는 것은 각 sub-network의 inference cost가 아니라, 여러 target용 model을 반복해서 search하고 train하는 총 design cost다. 배포 시에는 선택된 하나의 일반 sub-network만 실행한다.

## 문제 설정

device마다 최적 architecture가 다르다. 같은 phone에서도 battery, workload, thermal state에 따라 latency constraint가 달라질 수 있다. 기존 hardware-aware NAS를 deployment scenario마다 다시 수행하면 scenario 수 $N$에 따라 search와 training cost가 선형 증가한다.

OFA는 model training과 architecture search를 분리해 weight training을 한 번만 수행한다. supernet weight를 $W_o$, architecture를 $a_i$, sub-network를 고르는 selection을 $C(W_o,a_i)$라고 하면 목표는 다음과 같다.

$$
\min_{W_o}\sum_{a_i}\mathcal{L}_{val}(C(W_o,a_i)).
$$

모든 architecture가 독립 weight를 갖는 것이 아니라 $W_o$의 일부를 공유한다. 이상적인 목표는 각 sub-network가 독립적으로 학습된 같은 architecture 수준의 정확도를 유지하는 것이다. 그러나 architecture 수가 $10^{19}$ 이상이므로 매 update마다 모두 enumerate할 수 없다. 무작위로 몇 개만 sampling하면 sub-network 사이 weight interference 때문에 정확도가 크게 낮아진다.

## Architecture space

OFA는 CNN을 feature map resolution이 줄고 channel이 늘어나는 여러 unit으로 나눈다. 각 unit의 첫 layer만 필요할 때 stride 2를 쓰고 나머지는 stride 1이다.

### Elastic resolution

입력 image size를 128부터 224까지 step 4로 선택한다.

$$
R\in\{128,132,\ldots,224\}.
$$

총 25개 resolution이다. resolution은 전체 training 동안 batch마다 sampling한다.

### Elastic depth

각 unit의 layer 수를 다음에서 고른다.

$$
D\in\{2,3,4\}.
$$

depth $D$를 선택할 때 임의의 $D$개 layer를 고르지 않고 앞의 $D$개를 유지하고 뒤의 $4-D$개를 skip한다. 이렇게 해야 한 depth setting이 하나의 deterministic path에 대응하고 앞 layer weight를 큰 model과 공유할 수 있다.

### Elastic width

MBConv expansion ratio를 다음에서 고른다.

$$
e\in\{3,4,6\}.
$$

full width를 먼저 학습한 뒤 channel importance를 weight의 L1 norm으로 계산해 내림차순 정렬한다. 작은 width는 앞의 중요한 channel prefix를 사용한다. 독립적인 arbitrary channel subset을 허용하지 않으므로 nested weight sharing이 가능하다.

### Elastic kernel size

depthwise convolution의 kernel size는

$$
k\in\{3,5,7\}
$$

에서 고른다. 7x7 kernel의 중앙 5x5, 그 중앙 3x3을 작은 kernel로 공유한다. 단순 center crop은 같은 weight가 큰 kernel 일부와 독립 작은 kernel의 두 역할을 해야 하므로 distribution 차이가 생긴다. 논문은 layer별 kernel transformation matrix를 둔다.

- 7x7에서 5x5: 25x25 matrix
- 5x5에서 3x3: 9x9 matrix
- layer당 추가 parameter: $25^2+9^2=706$

matrix는 layer 안에서 channel 사이 공유되므로 overhead가 작다.

## Search space 크기 손계산

각 unit에서 depth가 2, 3, 4일 수 있고 각 layer에 expansion 3개와 kernel 3개, 총 9개 choice가 있다고 단순화하자. unit 하나의 path 수는

$$
9^2+9^3+9^4=7{,}371
$$

이다. unit이 5개면

$$
7{,}371^5=21{,}758{,}655{,}492{,}572{,}485{,}851
\approx2.18\times10^{19}
$$

이다. 여기에 25개 resolution choice가 있다. 논문은 architecture space를 대략 $2\times10^{19}$로 표현하고 resolution 25개를 별도로 언급한다.

모든 model을 독립 저장하면 불가능하지만 nested weight sharing으로 supernet parameter는 7.7M이다. 다만 architecture 수가 많다는 사실 자체가 모든 sub-network 품질을 보장하지는 않는다. progressive shrinking과 sampling strategy가 필요하다.

## Progressive shrinking

### 왜 큰 network부터 시작하는가

작은 network와 큰 network를 처음부터 동시에 sampling하면 같은 weight가 서로 다른 objective의 gradient를 받는다. capacity가 작은 model의 요구가 큰 model 학습을 방해하고, 아직 좋은 representation이 없는 weight를 모두가 공유하면서 optimization이 불안정해진다.

progressive shrinking은 다음 순서를 따른다.

1. 최대 resolution 선택 가능 상태에서 최대 kernel, 최대 depth, 최대 width의 full network를 학습한다.
2. elastic kernel size를 추가한다.
3. elastic depth를 추가한다.
4. elastic width를 추가한다.
5. 큰 sub-network도 계속 sampling해 이전 능력을 유지한다.

작은 network는 잘 학습된 큰 network의 중요 weight로 초기화된다. 가장 큰 network는 teacher로 soft label을 제공하고 ground-truth loss와 knowledge distillation loss를 함께 사용한다.

### Network pruning과 차이

일반 pruning은 full model을 학습하고 width 또는 weight를 줄인 뒤 특정 하나의 작은 model을 fine-tune한다. progressive shrinking은 resolution, kernel, depth, width 네 축을 줄이고, 한 개가 아니라 큰 것과 작은 것을 포함한 전체 family를 유지한다. 그래서 generalized pruning으로 볼 수 있지만 objective는 하나의 compressed model이 아니라 reusable supernet이다.

### 세부 training schedule

원문 Appendix의 순서는 다음과 같다.

#### Full network

- optimizer: SGD, Nesterov momentum 0.9
- weight decay: $3\times10^{-5}$
- initial learning rate: 2.6
- schedule: cosine decay
- epoch: 180
- batch: 2048
- hardware: 32 V100 GPU

#### Elastic kernel

- $K\in[7,5,3]$
- update당 sub-network 1개 sampling
- 125 epoch
- initial learning rate 0.96

#### Elastic depth stage 1

- $D\in[4,3]$
- update당 sub-network 2개 gradient 합산
- 25 epoch
- initial learning rate 0.08

#### Elastic depth stage 2

- $D\in[4,3,2]$
- 125 epoch
- initial learning rate 0.24

#### Elastic width stage 1

- $W\in[6,4]$
- update당 sub-network 4개 gradient 합산
- 25 epoch
- initial learning rate 0.08

#### Elastic width stage 2

- $W\in[6,4,3]$
- 125 epoch
- initial learning rate 0.24

전체 one-time training cost는 약 1,200 V100 GPU hours다. 이 비용은 0이 아니며 단일 deployment만 있다면 작은 model 하나를 직접 학습하는 것보다 클 수 있다. OFA의 이점은 scenario가 많을 때 amortize된다.

## Progressive shrinking ablation

Figure 7은 resolution 224에서 여러 고정 $(D,W,K)$ sub-network를 비교한다. progressive shrinking을 쓰면 8개 설정 모두 top-1이 2.5-3.7% 높았다.

대표적으로 $(D=4,W=3,K=3)$, 226M MAC architecture는 다음 결과를 보였다.

- PS 없음: 71.5%
- PS 사용: 74.8%
- 개선: 3.3%

Table 1의 Pixel1 specialization에서는 OFA without PS가 72.4%, with PS가 76.0%로 3.6% 차이다. 따라서 OFA의 성과를 weight sharing 자체만으로 설명할 수 없고 training order가 핵심이다.

## Specialization stage

OFA supernet을 학습한 뒤 target hardware와 constraint에 맞는 architecture를 고른다. 이 단계에서는 후보를 매번 full training하지 않는다.

### Accuracy predictor

1. architecture와 resolution이 다른 sub-network 16,000개를 random sampling한다.
2. ImageNet training set에서 뽑은 validation image 10,000개로 각 sub-network accuracy를 측정한다.
3. 각 layer의 kernel size와 expansion ratio를 one-hot encoding한다.
4. skip된 layer는 zero vector로 나타낸다.
5. input resolution one-hot을 추가한다.
6. 전체 vector를 3-layer feed-forward network에 넣는다.

predictor는 hidden layer마다 400 unit을 사용한다. test set의 predicted accuracy와 estimated accuracy 사이 RMSE는 수렴 시 0.21%였다.

### Latency predictor

target hardware마다 operator latency lookup table을 만든다. architecture의 layer configuration으로 latency를 예측한다. MAC가 아니라 measured target latency를 constraint로 사용할 수 있다는 점이 중요하다.

### Evolutionary search

accuracy predictor와 latency lookup을 neural-network twins라고 부른다. target constraint 아래에서 evolutionary search가 후보를 mutate/crossover하고 predictor로 빠르게 평가한다.

```python
def specialize(supernet, target, latency_limit):
    latency_lut = profile_operator_configs(target)
    accuracy_net = train_accuracy_predictor(
        sample_subnets(supernet, count=16000),
        validation_images=10000,
    )

    population = random_architectures()
    for generation in range(G):
        scored = []
        for arch in population:
            latency = latency_lut.predict(arch)
            if latency <= latency_limit:
                accuracy = accuracy_net.predict(arch)
                scored.append((accuracy, arch))
        elites = top_k(scored)
        population = mutate_and_crossover(elites)

    return extract_shared_weights(supernet, best(elites))
```

16K accuracy data 수집 비용과 predictor/search를 포함한 search cost는 논문 표에서 40 GPU hours다. 이 cost는 architecture마다 training하는 비용과 분리해야 한다.

## 비용을 세 종류로 분리하기

OFA를 정확히 이해하려면 cost를 섞지 않아야 한다.

### 1. Supernet training cost

- 약 1,200 GPU hours
- 모든 deployment scenario가 공유
- progressive shrinking 전체 비용

### 2. Specialization/search cost

- 논문 표에서 40 GPU hours
- 16K sub-network accuracy 수집과 predictor 기반 search
- scenario 수에 대해 constant라고 모델링

논문은 accuracy predictor와 latency table을 재사용해 새 constraint search 비용을 negligible하게 본다. 하지만 완전히 새로운 hardware는 latency table profiling이 필요하다.

### 3. Optional sub-network fine-tuning cost

- direct deployment: 0 epoch, 추가 training 없음
- `#25`: 선택 sub-network당 25 epoch
- `#75`: 선택 sub-network당 75 epoch

scenario가 $N$개면 optional fine-tuning은 $25N$ 또는 $75N$으로 다시 선형 증가한다. OFA의 O(1) 주장은 direct deploy 또는 공유 training/search를 가리키며 optional fine-tuning까지 무조건 O(1)이라는 뜻이 아니다.

## Pixel1 비용 표

논문 Table 1은 deployment scenario $N=40$을 가정한다.

| 방법 | top-1 | MAC | Pixel1 | search cost | training cost | 총 GPU hours | CO2e | AWS cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 72.0 | 300M | 66 ms | 0 | $150N$ | 6k | 1.7k lb | \$18.4k |
| MnasNet | 74.0 | 317M | 70 ms | $40{,}000N$ | 미보고 | 1,600k | 453.8k lb | \$4,896.0k |
| FBNet-C | 74.9 | 375M | 미보고 | $216N$ | $360N$ | 23k | 6.5k lb | \$70.4k |
| ProxylessNAS | 74.6 | 320M | 71 ms | $200N$ | $300N$ | 20k | 5.7k lb | \$61.2k |
| MobileNetV3-Large | 75.2 | 219M | 58 ms | 미보고 | $180N$ | 7.2k | 1.8k lb | \$22.2k |
| OFA without PS | 72.4 | 235M | 59 ms | 40 | 1200 | 1.2k | 0.34k lb | \$3.7k |
| OFA with PS | 76.0 | 230M | 58 ms | 40 | 1200 | 1.2k | 0.34k lb | \$3.7k |
| OFA with PS #25 | 76.4 | 230M | 58 ms | 40 | $1200+25N$ | 2.2k | 0.62k lb | \$6.7k |
| OFA with PS #75 | 76.9 | 230M | 58 ms | 40 | $1200+75N$ | 4.2k | 1.2k lb | \$13.0k |

CO2와 AWS 비용은 당시 P3.16xlarge price와 cited carbon model로 계산한 값이다. 현재 cloud price나 grid carbon intensity와 같은 절대값으로 보면 안 된다. 비교 구조, 즉 repeated NAS가 scenario 수에 비례한다는 점이 핵심이다.

$N=40$에서 OFA CO2e는 ProxylessNAS보다 16배, FBNet보다 19배, MnasNet보다 1,300배 적다고 논문은 계산한다.

## ImageNet accuracy와 compute 결과

### MAC constraint

| OFA point | MAC | top-1 |
|---|---:|---:|
| small | 389M | 79.1% |
| medium | 482M | 79.6% |
| mobile max | 595M | 80.0% |

389M point는 비슷한 MAC의 EfficientNet-B0 76.3%보다 2.8% 높다. 595M point는 EfficientNet-B2 1,000M, 79.8%보다 0.2% 높고 MAC은 1.68배 적다.

### Pixel1 latency constraint

| OFA point | Pixel1 latency | top-1 |
|---|---:|---:|
| low | 78.7 ms | 78.7% |
| medium | 132 ms | 79.8% |
| high | 143 ms | 80.1% |

143 ms point는 EfficientNet-B2의 375 ms, 79.8%와 비교해 2.6배 빠르고 0.3% 높다. 같은 architecture를 scratch에서 train한 결과는 OFA shared pretrained weight보다 낮았다. 논문은 architecture뿐 아니라 inherited weight가 성능에 기여한다고 해석한다.

### MobileNetV3 trade-off

OFA는 Pixel1에서 20, 28, 40, 58 ms 부근의 후보가 71.4, 73.3, 74.9, 76.4%를 보였다. MobileNetV3의 네 고정 point보다 trade-off curve가 위에 있었다. 하나의 width multiplier만 바꾸는 방식보다 kernel, depth, width, resolution을 함께 조절하는 search space가 넓기 때문이다.

## Hardware별 측정 조건

논문은 latency를 한 장치에서만 측정하지 않았다.

### Cloud GPU

- NVIDIA 1080Ti, V100
- PyTorch 1.0 + cuDNN
- batch size 64

### Server CPU

- Intel Xeon E5-2690 v4
- MKL-DNN
- batch size 1

### Mobile phone

- Samsung, Google, LG phone
- TensorFlow Lite
- batch size 1

### Mobile GPU

- Jetson TX2
- PyTorch 1.0 + cuDNN
- batch size 16

### FPGA

- Xilinx ZU9EG, ZU3EG
- Vitis AI
- INT8 quantization
- batch size 1

GPU curve의 batch 64 latency와 phone batch 1 latency를 같은 의미로 비교하면 안 된다. hardware specialization이 각 조건에서 따로 search됐다는 것이 논문의 point다.

## Device별 architecture가 달라지는 이유

Figure 14는 같은 OFA space에서도 target에 따라 다른 sub-network가 선택됨을 보여 준다.

- ZU3EG FPGA, 4.1 ms, resolution 164: 모든 MBConv가 3x3 kernel
- Xeon CPU, 10.9 ms, resolution 144: 3x3, 5x5, 7x7을 혼합
- NVIDIA 1080Ti, 14.9 ms, batch 64: CPU보다 넓은 expansion을 더 많이 사용

FPGA의 depthwise engine과 pointwise engine pipeline은 3x3에서 throughput이 잘 맞고, 큰 kernel은 BRAM을 초과할 수 있다. CPU는 큰 kernel을 효율적으로 처리할 여지가 있고, GPU/FPGA는 parallel array를 채우기 위해 넓은 channel을 선호한다.

이 결과는 MAC가 같은 architecture라도 device별 latency가 다르며, hardware cost model을 search loop에 넣어야 함을 보여 준다.

## FPGA arithmetic intensity

논문은 memory access가 비싸고 compute가 상대적으로 싸다는 원리로 FPGA 결과를 분석한다. arithmetic intensity는 OPs/Byte다.

| width 수준 | MobileNetV2 | MnasNet | OFA |
|---|---:|---:|---:|
| 0.35 | 27.0 | 27.6 | 39.4 |
| 0.5 | 35.3 | 37.1 | 49.4 |
| 0.75 | 51.6 | 51.9 | 54.4 |
| 1.0 | 61.0 | 61.2 | 63.9 |

OFA는 target latency constraint 아래에서 memory footprint 대비 compute가 많은 architecture를 찾았다. 논문은 OFA arithmetic intensity가 MobileNetV2보다 48%, MnasNet보다 43% 높고, utilization/GOPS가 70-90% 향상됐다고 설명한다.

ZU3EG과 ZU9EG의 MnasNet 대비 roofline 분석에서는 arithmetic intensity가 각각 43%, 33%, GOPS/s가 92%, 72% 높다고 별도 figure caption에 정리한다. 비교 기준과 point가 다르므로 percentage를 하나로 합치면 안 된다.

## Weight, activation, data movement 손계산

이 절은 논문 표의 measured latency가 아니라 architecture choice의 비용을 이해하기 위한 별도 계산이다.

### Resolution scaling

입력 resolution이 224에서 160으로 줄면 초기 feature map element 수는 대략 area 비율로

$$
(160/224)^2\approx0.510
$$

이 된다. stride pattern과 channel이 같다고 단순화하면 대부분 spatial convolution MAC과 activation도 약 절반으로 줄 수 있다. 하지만 fixed overhead, later global pooling/FC, memory alignment 때문에 actual latency가 정확히 절반이 되지는 않는다.

### Kernel scaling

depthwise convolution $[B,C,H,W]$의 MAC은 $BCHWk^2$다. 같은 shape에서 7x7은 3x3보다

$$
49/9\approx5.44
$$

배 많은 MAC과 kernel weight를 가진다. 그러나 전체 MBConv에는 1x1 expansion/projection이 포함되어 있어 block total이 5.44배가 되는 것은 아니다. 큰 kernel은 receptive field를 늘리지만 FPGA BRAM과 depthwise pipeline에 불리할 수 있다.

### Width scaling

MBConv input/output channel과 expansion channel을 모두 width factor $\gamma$로 키운다고 단순화하면 1x1 pointwise MAC는 대략 $\gamma^2$, activation은 $\gamma$에 비례한다. parallel array가 큰 GPU/FPGA는 넓은 matrix에서 utilization이 좋아져 MAC 증가보다 latency 증가가 작을 수 있다. CPU나 memory-limited device는 반대일 수 있다.

### Supernet storage와 deployed storage

supernet 7.7M parameter를 FP32로 저장하면 단순히

$$
7.7\text{M}\times4\approx30.8\text{ MB}
$$

다. FP16이면 15.4 MB다. 그러나 deployment에는 선택된 sub-network weight만 export할 수 있으므로 supernet 전체를 phone에 탑재할 필요가 없다. runtime adaptation을 위해 여러 sub-network를 즉시 바꾸려면 shared supernet 또는 여러 extracted model과 normalization statistic 관리가 필요하다.

### Data movement

OFA objective에는 MAC 또는 measured latency가 들어가지만 activation peak와 DRAM byte가 명시적 primary objective는 아니다. 같은 predicted latency라도 activation이 큰 architecture는 memory pressure와 thermal behavior가 다를 수 있다. 온디바이스 확장에서는 latency predictor 외에 peak activation, bytes moved, energy predictor를 추가할 필요가 있다.

## 논문 증거와 별도 해석의 구분

### 논문이 직접 보고한 것

- 7.7M parameter로 $10^{19}$ 이상 sub-network 지원
- progressive shrinking schedule과 1,200 V100 GPU hours
- 16K architecture, 10K image accuracy predictor dataset, RMSE 0.21%
- search cost 40 GPU hours
- Pixel1, six mobile devices, GPU/CPU/FPGA latency curve
- OFA direct deploy와 #25/#75 fine-tuning 결과
- FPGA arithmetic intensity와 GOPS/s 개선

### 이 리뷰의 별도 계산

- search space $7{,}371^5=2.1759\times10^{19}$
- 224에서 160으로 resolution area 0.510배
- 7x7과 3x3 depthwise MAC 5.44배
- 7.7M parameter의 FP32/FP16 단순 storage

별도 계산은 architecture dimension의 영향 설명용이며 measured device cost가 아니다.

## 온디바이스 적용 설계

### Static specialization

앱 설치 시 device model을 식별하고 해당 sub-network 하나를 download/export한다. 가장 단순하며 runtime graph가 고정된다. 장치별 model artifact 관리가 필요하다.

### Runtime adaptive specialization

battery, temperature, workload에 따라 resolution 또는 sub-network를 바꾼다. 이 경우 다음을 고려해야 한다.

- model switch latency와 weight cache
- 각 sub-network의 BatchNorm statistic
- allocator가 최대 activation을 예약하는지 여부
- camera preprocessing resolution switch 비용
- thermal control loop의 안정성
- output quality가 frame마다 흔들리지 않는지

### Shared encoder 확장

detection, segmentation, VLM이 하나의 OFA vision encoder에서 width/depth/resolution을 선택하도록 확장할 수 있다. 그러나 task마다 최적 feature hierarchy와 normalization이 다르므로 ImageNet accuracy predictor만으로는 부족하다. multi-task metric과 activation memory를 cost model에 넣어야 한다.

## 실제 재현 계획

1. MobileNetV3 기반 OFA supernet 또는 공개 checkpoint를 준비한다.
2. target phone에서 모든 candidate block configuration의 batch=1 latency LUT를 만든다.
3. warm-up, thread 수, delegate, DVFS 조건을 고정한다.
4. accuracy predictor dataset을 architecture space 전체에서 균형 sampling한다.
5. predictor RMSE뿐 아니라 selected top candidate의 rank correlation을 본다.
6. MAC constraint search와 measured latency constraint search를 비교한다.
7. direct inherited weight, 25 epoch fine-tune, scratch training을 같은 architecture에서 비교한다.
8. p50/p95 latency, peak memory, power, temperature를 측정한다.
9. 10분 sustained run 후 throttling에서 constraint를 다시 검증한다.

## 장점과 기여

1. 여러 device용 NAS의 반복 training cost를 하나의 supernet으로 amortize했다.
2. depth, width, kernel, resolution을 하나의 nested architecture space로 묶었다.
3. progressive shrinking으로 sub-network interference를 크게 줄였다.
4. hardware별 measured latency를 specialization에 사용했다.
5. accuracy/latency twin으로 search 시 full training과 full evaluation을 줄였다.
6. mobile, CPU, GPU, FPGA에서 서로 다른 optimal architecture를 보였다.
7. architecture뿐 아니라 inherited weight가 성능에 기여함을 비교했다.
8. design cost를 GPU hour, carbon, dollar까지 확장해 평가했다.

## 한계와 비판적 관점

1. supernet training 1,200 GPU hours는 여전히 크며 deployment scenario가 적으면 amortization 이점이 작다.
2. O(1) design cost 주장은 optional per-subnet fine-tuning을 제외한 표현이다.
3. 새로운 hardware는 latency LUT profiling이 필요하므로 완전히 cost-free specialization은 아니다.
4. lookup table의 layer latency 합은 fusion, cache, concurrent execution을 완전히 반영하지 못할 수 있다.
5. accuracy predictor가 ImageNet top-1에 맞춰져 다른 task로 바로 일반화되지 않는다.
6. $10^{19}$개 모든 sub-network를 검증할 수 없으며 sampling bias가 있다.
7. channel L1 norm과 prefix sharing이 항상 최적 importance ordering이라는 보장은 없다.
8. BatchNorm statistic과 quantization을 sub-network마다 관리하는 실무 복잡성이 있다.
9. carbon/AWS cost는 당시 가정에 의존해 현재 절대값으로 재사용할 수 없다.
10. memory, energy, p95, thermal constraint는 latency만큼 직접 최적화하지 않았다.

## 구현 체크리스트

- [ ] full network를 먼저 충분히 학습했는가
- [ ] resolution을 training 전체에서 sampling했는가
- [ ] elastic kernel, depth, width 순서를 지켰는가
- [ ] 작은 kernel이 큰 kernel 중앙을 올바르게 공유하는가
- [ ] layer별 kernel transform 706 parameter를 적용했는가
- [ ] depth sub-network가 앞 $D$개 layer prefix를 사용하는가
- [ ] channel을 L1 norm으로 정렬하고 width prefix를 사용하는가
- [ ] large sub-network와 small sub-network를 함께 sampling하는가
- [ ] teacher soft label과 ground-truth loss를 올바르게 결합했는가
- [ ] sub-network별 normalization statistic을 재계산했는가
- [ ] accuracy predictor data가 architecture space를 고르게 덮는가
- [ ] target device에서 batch와 runtime 조건을 맞춰 latency LUT를 만들었는가
- [ ] LUT prediction과 end-to-end measured latency 오차를 확인했는가
- [ ] supernet training, search, optional fine-tuning cost를 분리했는가
- [ ] MAC, p50/p95 latency, peak memory, power, thermal을 함께 기록했는가

## 후속 연구와의 연결

OFA는 weight-sharing NAS, slimmable network, dynamic network의 교차점에 있다. 후속 연구에서는 BatchNorm calibration, zero-cost predictor, transformer supernet, quantization-aware supernet, multi-objective hardware search로 확장할 수 있다.

현재 on-device VLM에는 다음 연결이 중요하다.

- vision encoder의 resolution/depth/width를 device별로 specialize
- visual token 수를 latency와 KV cache objective에 포함
- detector, segmenter, VLM이 공유하는 encoder의 multi-task accuracy predictor
- FP16/INT8/W4A16 precision choice까지 architecture gene에 포함
- compiler 측정값을 LUT가 아니라 end-to-end cost model로 사용

## 최종 평가

Once-for-All의 핵심 기여는 하나의 가장 좋은 mobile network를 제시한 것이 아니라, 여러 hardware와 constraint에 맞는 network family를 한 번의 weight training으로 제공하는 deployment methodology를 만든 것이다. progressive shrinking은 방대한 sub-network가 공유 weight를 쓰면서도 품질을 유지하게 하는 실질적인 핵심이며, latency/accuracy twin은 specialization을 저비용 search로 바꾼다.

다만 OFA의 비용 절감 주장은 supernet training, specialization, optional fine-tuning을 분리해 읽어야 한다. 또한 실제 장치에서는 LUT 예측만 믿지 말고 extracted sub-network의 p95 latency, peak activation, power, thermal을 다시 측정해야 한다. 이 검증을 추가하면 OFA는 다양한 phone과 edge accelerator를 지원해야 하는 제품에서 model 설계 비용을 크게 줄이는 강력한 기반이 된다.

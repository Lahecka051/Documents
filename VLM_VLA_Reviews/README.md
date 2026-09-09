# VLM / VLA 논문 상세 리뷰

VLM과 VLA의 구조, 시각 토큰 효율화, 행동 생성, 실시간 제어를 공부하기 위한 한국어 논문 해설 26편입니다. 기존 10편과 신규 16편을 취합했으며, 각 논문은 Markdown 한 파일 및 주 PDF 한 파일에 대응합니다. 초록 요약이 아니라 motivation, 저자의 주장, 섹션별 기술, 기호와 텐서 차원, 수식, forward pass, 실험 조건 및 한계를 풀어 설명합니다.

총 **주 PDF 26개 + 보충자료·추가 출판본 PDF 6개**, 원문 발췌 PNG 545개를 포함합니다. PDF는 리뷰에서 사용한 버전을 그대로 보존했으며, 최신판으로 임의 교체하거나 다시 압축하지 않았습니다. 각 리뷰 상단과 아래 목록에서 원문을 열 수 있습니다.

## 논문 목록

| 번호 | 논문 | 발표·기준 | 상세 해설 | 주 PDF | 중심 주제 |
|---|---|---|---|---|---|
| 01 | VLA-Cache | NeurIPS 2025 | [리뷰](01_VLA_Cache_Detailed_Review_KO.md) | [PDF](papers/01_VLA_Cache.pdf) | 연속 관측에서 시각 토큰의 KV 재사용 |
| 02 | HiRED | AAAI 2025 | [리뷰](02_HiRED_Detailed_Review_KO.md) | [PDF](papers/02_HiRED.pdf) | 고해상도 이미지의 토큰 예산 배분 |
| 03 | FEATHER | ICCV 2025 | [리뷰](03_FEATHER_Detailed_Review_KO.md) | [PDF](papers/03_FEATHER.pdf) | 시각 토큰 pruning의 공간적 편향과 개선 |
| 04 | FastV | ECCV 2024 | [리뷰](04_FastV_Detailed_Review_KO.md) | [PDF](papers/04_FastV.pdf) | LLM 내부 시각 토큰 pruning |
| 05 | Prismatic VLMs | ICML 2024 | [리뷰](05_Prismatic_VLMs_Detailed_Review_KO.md) | [PDF](papers/05_Prismatic_VLMs.pdf) | 시각 표현·전처리·학습 설계의 통제 비교 |
| 06 | X-VLA | ICLR 2026 | [리뷰](06_X_VLA_Detailed_Review_KO.md) | [PDF](papers/06_X_VLA.pdf) | Soft prompt 기반 cross-embodiment 정책 |
| 07 | Fast-in-Slow | NeurIPS 2025 | [리뷰](07_Fast_in_Slow_Detailed_Review_KO.md) | [PDF](papers/07_Fast_in_Slow.pdf) | 느린 추론과 빠른 행동 실행의 결합 |
| 08 | FASTer | ICLR 2026 | [리뷰](08_FASTer_Detailed_Review_KO.md) | [PDF](papers/08_FASTer.pdf) | Neural action tokenization과 블록 단위 디코딩 |
| 09 | CogVLA | NeurIPS 2025 | [리뷰](09_CogVLA_Detailed_Review_KO.md) | [PDF](papers/09_CogVLA.pdf) | 지시 기반 시각·언어·행동 계산 효율화 |
| 10 | VLA-Adapter | AAAI 2026 | [리뷰](10_VLA_Adapter_Detailed_Review_KO.md) | [PDF](papers/10_VLA_Adapter.pdf) | 작은 VLM의 표현과 행동 정책 연결 |
| 11 | SigLIP 2 | arXiv 2025, v1 | [리뷰](11_SigLIP2_Detailed_Review_KO.md) | [PDF](papers/11_SigLIP2.pdf) | 다국어 시각·언어 정렬과 dense feature 학습 |
| 12 | Meta CLIP 2 | NeurIPS 2025 / arXiv v3 | [리뷰](12_MetaCLIP2_Detailed_Review_KO.md) | [PDF](papers/12_MetaCLIP2.pdf) | 전 세계 웹의 개념 분포와 데이터 선별 |
| 13 | FAST | RSS 2025 / arXiv v1 | [리뷰](13_FAST_Detailed_Review_KO.md) | [PDF](papers/13_FAST.pdf) | DCT·양자화·BPE 기반 행동 토큰화 |
| 14 | OpenVLA-OFT | RSS 2025 / arXiv v2 | [리뷰](14_OpenVLA_OFT_Detailed_Review_KO.md) | [PDF](papers/14_OpenVLA_OFT.pdf) | 병렬 행동 생성과 효율적인 fine-tuning |
| 15 | BEAST | NeurIPS 2025 / arXiv v3 | [리뷰](15_BEAST_Detailed_Review_KO.md) | [PDF](papers/15_BEAST.pdf) | B-spline 기반 행동 시퀀스 표현 |
| 16 | RTC | NeurIPS 2025 / arXiv v2 | [리뷰](16_RTC_Detailed_Review_KO.md) | [PDF](papers/16_RTC.pdf) | 지연을 고려한 flow 정책의 실시간 chunk 실행 |
| 17 | Training-Time Action Conditioning | arXiv 2025, v2 | [리뷰](17_Training_Time_Action_Conditioning_Detailed_Review_KO.md) | [PDF](papers/17_Training_Time_Action_Conditioning.pdf) | 실행할 행동 prefix를 학습 조건으로 사용 |
| 18 | FASTER: Real-Time Flow VLAs | arXiv 2026, v3 | [리뷰](18_FASTER_RealTime_Flow_Detailed_Review_KO.md) | [PDF](papers/18_FASTER_RealTime_Flow.pdf) | Horizon-aware schedule과 streaming 실행 |
| 19 | SmolVLA | arXiv 2025, v1 | [리뷰](19_SmolVLA_Detailed_Review_KO.md) | [PDF](papers/19_SmolVLA.pdf) | 소형 VLA와 비동기 추론·실행 |
| 20 | Teaching Tiny VLA | arXiv 2026, v3 | [리뷰](20_Teaching_Tiny_VLA_Detailed_Review_KO.md) | [PDF](papers/20_Teaching_Tiny_VLA.pdf) | 소형 VLA의 시각·행동 증류 |
| 21 | FlashVLA | arXiv 2026, v1 | [리뷰](21_FlashVLA_Detailed_Review_KO.md) | [PDF](papers/21_FlashVLA.pdf) | Streaming action decoding과 비동기 실행 |
| 22 | Learnable VLA Caching | arXiv 2026, v2 | [리뷰](22_Learnable_VLA_Caching_Detailed_Review_KO.md) | [PDF](papers/22_Learnable_VLA_Caching.pdf) | 학습 가능한 시각 토큰 캐싱 |
| 23 | Speculative Decoding | ICML 2023 / arXiv v2 | [리뷰](23_Speculative_Decoding_Detailed_Review_KO.md) | [PDF](papers/23_Speculative_Decoding.pdf) | 작은 모델의 제안과 확률적으로 정확한 검증 |
| 24 | CLIP | ICML 2021 / arXiv v1 | [리뷰](24_CLIP_Detailed_Review_KO.md) | [PDF](papers/24_CLIP.pdf) | 대조학습과 zero-shot 시각 전이 |
| 25 | FlashAttention-2 | ICLR 2024 / arXiv v1 | [리뷰](25_FlashAttention_2_Detailed_Review_KO.md) | [PDF](papers/25_FlashAttention_2.pdf) | Exact attention의 GPU 병렬화·작업 분할 |
| 26 | PagedAttention | SOSP 2023 / arXiv v1 | [리뷰](26_PagedAttention_Detailed_Review_KO.md) | [PDF](papers/26_PagedAttention.pdf) | KV cache 메모리 관리와 LLM 서빙 |

발표 연도와 제목은 첨부 PDF 본문을 우선합니다. Prismatic VLMs는 ICML 2024 논문이며, FASTer는 첨부 최종본의 제목인 *Toward Efficient Autoregressive Vision Language Action Modeling via Neural Action Tokenization*을 사용합니다.

13의 FAST, 08의 FASTer, 18의 FASTER는 서로 다른 논문입니다. arXiv 표기는 리뷰에서 고정한 기준본을 뜻하며, 확인하지 않은 학회 채택을 추정하지 않습니다. 출판본과 arXiv의 페이지·수식 번호가 다른 경우 각 리뷰의 버전 규칙을 따라야 합니다.

## 별도 보충자료와 추가 출판본

| 대응 리뷰 | 추가 PDF | 포함 이유 |
|---|---|---|
| 03 FEATHER | [ICCV supplementary](papers/03_FEATHER_Supplement.pdf) | 별도 5쪽 보충자료 |
| 04 FastV | [ECCV supplementary](papers/04_FastV_Supplement.pdf) | 별도 3쪽 attention map 자료 |
| 10 VLA-Adapter | [arXiv v2 전체본](papers/10_VLA_Adapter_arXiv_v2_With_Appendix.pdf) | 출판본이 가리키는 부록과 버전별 수식 대조 |
| 12 Meta CLIP 2 | [NeurIPS 2025 출판본](papers/12_MetaCLIP2_NeurIPS_2025.pdf) | arXiv v3와 구분한 추가 실험·기술 부록 |
| 24 CLIP | [ICML supplementary](papers/24_CLIP_Supplement.pdf) | 부록 재배열과 추가 설정 대조 |
| 25 FlashAttention-2 | [ICLR 2024 출판본](papers/25_FlashAttention_2_ICLR_2024.pdf) | 알고리즘 표기 교정과 추가 decode 실험 대조 |

원문 URL, 버전, 페이지 수, 파일 크기, SHA-256 및 리뷰 대응은 [PDF manifest](papers/manifest.json)에 기록했습니다. 모든 PDF는 리뷰에 기록된 원본 해시와 대조했습니다. 큰 PDF의 GitHub 미리보기가 열리지 않으면 해당 파일 화면의 다운로드 기능을 이용해 주세요.

## Figure와 수식을 읽는 방법

총 545개의 원문 발췌 PNG를 포함합니다. 기존 10편의 Figure·수식 210개와 신규 16편의 Figure·Table·수식·Algorithm 등 335개입니다. 이미지 한 장에 복수의 식이나 문장 내 정의가 포함될 수 있으므로, 이미지 수와 원문 식 번호의 개수는 다릅니다.

- 핵심 Figure와 수식은 PDF에서 직접 추출한 PNG로 관련 해설 옆에 배치합니다. 원문의 축, 범례, 기호와 식 번호를 확인하는 용도입니다.
- 편집 가능한 LaTeX와 단계별 해설을 함께 제공합니다. LaTeX를 지원하지 않는 뷰어에서도 원문 수식 이미지는 확인할 수 있습니다.
- 원문에 식 번호가 없는 식은 비번호 식으로 구분합니다. 리뷰에서 추가한 설명용 수식은 원문 수식으로 취급하지 않습니다.
- 캡션의 `PDF p.N`은 각 리뷰에서 정의한 실제 PDF 페이지 기준입니다. 별도 보충자료를 사용한 경우 출처를 구분합니다.
- 논문별 `assets/<번호_논문>/publication_assets.json`에는 출처와 이미지 추출 위치를 기록합니다.

## 해석과 검증 범위

저자 보고, 공개 코드 확인, 수치 재계산, 리뷰어 해석, 논문 미기재 사항을 구분합니다. 이 리뷰 작업에서 GPU 학습·추론이나 실제 로봇 실험을 새로 수행하지 않았습니다. FLOPs·토큰 감소, CUDA latency, 전체 응답 지연, action chunk와 제어 주파수는 서로 다른 지표입니다. Jetson Thor 관련 제안은 후속 검증 대상이며, Thor 이식이나 TensorRT 가속을 검증했다는 뜻이 아닙니다.

Markdown, 이미지, 저장소 PDF 링크는 이 폴더를 기준으로 한 상대 경로입니다. 로컬 분석 코드, 전체 페이지 진단 이미지, 작업 배정·브라우저 정보와 임시 파일은 포함하지 않습니다. 논문별 검산 자료와 검증 JSON은 해당 리뷰 작업의 기록이며, 모델 학습·실행 또는 GitHub 실서비스 검증을 수행했다는 추가 주장은 아닙니다.

## 출처와 권리

이 문서들은 원 논문을 이해하기 위한 AI 보조 한국어 해설입니다. 인용 시에는 각 리뷰의 공식 원문 링크를 통해 해당 논문과 버전을 확인해 주세요. 원문 PDF와 발췌한 Figure·수식의 권리는 각 원저자 또는 권리자에게 있으며, 이 저장소는 해당 자료의 라이선스를 변경하거나 새로운 이용 허락을 부여하지 않습니다. 재사용 시에는 각 원문의 이용 조건을 확인해야 합니다. 논문별 출처는 각 문서, asset manifest 및 PDF manifest에 표시합니다.

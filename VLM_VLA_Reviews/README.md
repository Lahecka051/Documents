# VLM / VLA 논문 상세 리뷰

VLM과 VLA의 구조, 시각 토큰 효율화, 행동 생성, 실시간 제어를 공부하기 위한 한국어 논문 해설 10편입니다. 각 논문은 Markdown 한 파일에 대응합니다. 초록 요약이 아니라 motivation, 저자의 주장, 섹션별 기술, 기호와 텐서 차원, 수식, forward pass, 실험 조건 및 한계를 풀어 설명합니다.

## 논문 목록

| 번호 | 논문 | 발표 | 상세 해설 | 중심 주제 |
|---|---|---|---|---|
| 01 | VLA-Cache | NeurIPS 2025 | [한국어 리뷰](01_VLA_Cache_Detailed_Review_KO.md) | 연속 관측에서 시각 토큰의 KV 재사용 |
| 02 | HiRED | AAAI 2025 | [한국어 리뷰](02_HiRED_Detailed_Review_KO.md) | 고해상도 이미지의 토큰 예산 배분 |
| 03 | FEATHER | ICCV 2025 | [한국어 리뷰](03_FEATHER_Detailed_Review_KO.md) | 시각 토큰 pruning의 공간적 편향과 개선 |
| 04 | FastV | ECCV 2024 | [한국어 리뷰](04_FastV_Detailed_Review_KO.md) | LLM 내부 시각 토큰 pruning |
| 05 | Prismatic VLMs | ICML 2024 | [한국어 리뷰](05_Prismatic_VLMs_Detailed_Review_KO.md) | 시각 표현·전처리·학습 설계의 통제 비교 |
| 06 | X-VLA | ICLR 2026 | [한국어 리뷰](06_X_VLA_Detailed_Review_KO.md) | Soft prompt 기반 cross-embodiment 정책 |
| 07 | Fast-in-Slow | NeurIPS 2025 | [한국어 리뷰](07_Fast_in_Slow_Detailed_Review_KO.md) | 느린 추론과 빠른 행동 실행의 결합 |
| 08 | FASTer | ICLR 2026 | [한국어 리뷰](08_FASTer_Detailed_Review_KO.md) | Neural action tokenization과 블록 단위 디코딩 |
| 09 | CogVLA | NeurIPS 2025 | [한국어 리뷰](09_CogVLA_Detailed_Review_KO.md) | 지시 기반 시각·언어·행동 계산 효율화 |
| 10 | VLA-Adapter | AAAI 2026 | [한국어 리뷰](10_VLA_Adapter_Detailed_Review_KO.md) | 작은 VLM의 표현과 행동 정책 연결 |

발표 연도와 제목은 첨부 PDF 본문을 우선합니다. Prismatic VLMs는 ICML 2024 논문이며, FASTer는 첨부 최종본의 제목인 *Toward Efficient Autoregressive Vision Language Action Modeling via Neural Action Tokenization*을 사용합니다.

## Figure와 수식을 읽는 방법

총 210개의 원문 발췌 PNG(Figure 105개, 수식 이미지 105개)를 포함합니다. 수식 이미지 한 장에 복수의 식이나 문장 내 정의가 포함될 수 있으므로, 이미지 수와 원문 식 번호의 개수는 다릅니다.

- 핵심 Figure와 수식은 PDF에서 직접 추출한 PNG로 관련 해설 옆에 배치합니다. 원문의 축, 범례, 기호와 식 번호를 확인하는 용도입니다.
- 편집 가능한 LaTeX와 단계별 해설을 함께 제공합니다. LaTeX를 지원하지 않는 뷰어에서도 원문 수식 이미지는 확인할 수 있습니다.
- 원문에 식 번호가 없는 식은 비번호 식으로 구분합니다. 리뷰에서 추가한 설명용 수식은 원문 수식으로 취급하지 않습니다.
- 캡션의 `PDF p.N`은 각 리뷰에서 정의한 실제 PDF 페이지 기준입니다. 별도 보충자료를 사용한 경우 출처를 구분합니다.
- 논문별 `assets/<번호_논문>/publication_assets.json`에는 출처와 이미지 추출 위치를 기록합니다.

## 해석과 검증 범위

저자 보고, 공개 코드 확인, 수치 재계산, 리뷰어 해석, 논문 미기재 사항을 구분합니다. 이 리뷰 작업에서 GPU 학습·추론이나 실제 로봇 실험을 새로 수행하지 않았습니다. FLOPs·토큰 감소, CUDA latency, 전체 응답 지연, action chunk와 제어 주파수는 서로 다른 지표입니다. Jetson Thor 관련 제안은 후속 검증 대상이며, Thor 이식이나 TensorRT 가속을 검증했다는 뜻이 아닙니다.

Markdown과 이미지 링크는 이 폴더를 기준으로 한 상대 경로입니다. 원본 PDF, 로컬 분석 코드, 전체 페이지 진단 이미지와 임시 파일은 이 리뷰 모음에 포함하지 않습니다.

## 출처와 권리

이 문서들은 원 논문을 이해하기 위한 AI 보조 한국어 해설입니다. 인용 시에는 각 리뷰의 공개 원문 링크를 통해 해당 논문과 버전을 확인해 주세요. 발췌한 Figure와 수식의 권리는 각 원저자 또는 권리자에게 있으며, 이 저장소의 해설 문서가 원문 이미지의 라이선스를 변경하거나 재허가하지 않습니다. 논문별 출처는 각 문서와 asset manifest에 표시합니다.

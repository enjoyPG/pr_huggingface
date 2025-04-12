# DETR (facebook/detr-resnet-50) 모델 소개

[DETR (DEtection TRansformer)](https://huggingface.co/facebook/detr-resnet-50)는 Facebook AI Research (FAIR)가 발표한 객체 탐지 모델로, 기존의 CNN 기반 방식과는 **완전히 다른 Transformer 기반 아키텍처**를 사용합니다.

---

## ✅ DETR의 핵심 개념

DETR은 객체 탐지를 위해 **Transformer 구조**를 사용한 첫 모델로, 기존 객체 탐지기의 복잡한 구조를 단순화하여 End-to-End 학습이 가능합니다.

---

## 🧠 핵심 특징

| 특징 | 설명 |
|------|------|
| **Transformer 기반** | CNN + Transformer 구조 사용 (이미지에 self-attention 적용) |
| **End-to-End 구조** | Region Proposal 없이 직접 객체 박스 + 라벨 예측 |
| **Fixed number of predictions** | 항상 일정한 수의 예측 (ex. 100개), 그 중 유효한 것만 필터링 |
| **Set-based Loss** | Hungarian matching을 활용한 1:1 예측-정답 매칭 방식 |

---

## 🔧 `detr-resnet-50` 구성

| 구성 요소 | 설명 |
|-----------|------|
| **Backbone** | ResNet-50 (이미지 특징 추출용 CNN) |
| **Transformer** | 위치 정보를 기반으로 객체 간 관계 파악 |
| **Object queries** | 학습 가능한 벡터, 각각 하나의 객체를 예측 |
| **Output** | 각 query → `[bounding box + class label]` 예측 |

---

## ✅ 기존 객체 탐지기와 비교

| 항목 | 기존 방식 (ex. Faster R-CNN) | DETR (`detr-resnet-50`) |
|------|------------------------------|--------------------------|
| 구조 | 다단계 (Region Proposal → Classification) | 단일 End-to-End |
| Anchor box | 필요함 (수작업 튜닝 필수) | 불필요 |
| 학습 속도 | 빠름 | 느림 (더 많은 epoch 필요) |
| 소형 객체 탐지 | 우수 | 상대적으로 불리함 |
| 후처리 (Post-processing) | 복잡 (NMS 등) | 불필요 또는 간단 |
| 구현 복잡도 | 높음 | 간결함 |

---

## 💡 요약

| 항목 | 내용 |
|------|------|
| **모델 이름** | `facebook/detr-resnet-50` |
| **장점** | 구조 단순, Anchor-free, 정확도 우수 |
| **단점** | 소형 객체 탐지 약점, 학습 시간 길어짐 |
| **사용 예시** | 고양이, 사람, 자동차 등 객체 탐지 |
| **대체 모델** | YOLOv5, YOLOS, Faster-RCNN 등 |

---

## 🔗 참고

- Hugging Face 모델 페이지: [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50)
- 원 논문: ["End-to-End Object Detection with Transformers"](https://arxiv.org/abs/2005.12872)


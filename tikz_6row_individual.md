# Architecture Diagram — 6행 개별 렌더링

> 소스: [rows/](file:///DATA/ST-DiT2/thesis/thesis_2026/tikz_scratch/rows)
> 전체 합본: [tikz_6row_rendered.md](file:///home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/tikz_6row_rendered.md)

## (a) Overview — 전체 2-Stage 파이프라인

![Row A: Overview](/home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/row_a_overview.png)

---

## (a') Stage 1 Overview — CDiTUNet U-Net 구조

![Row A': Stage 1 U-Net Overview](/home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/row_a2_stage1_unet_v6.png)

---

## (b) Encoder — Stem → ResBlock×3 levels → 64²

![Row B: Encoder](/home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/row_b_encoder.png)

---

## (c) DiT Bottleneck — Patchify → 1024 tokens → DiT×4 → Unpatchify

![Row C: DiT Bottleneck](/home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/row_c_dit_bottleneck.png)

---

## (c') ST-DiT Block Detail — adaLN and Self-Attention / MLP

![Row C': ST-DiT Block Detail](/home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/row_c2_stdit_block.png)

---

## (d) Decoder — ↑2 + skip concat → ResBlock×3 levels → 2ch output

![Row D: Decoder](/home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/row_d_decoder.png)

---

## (e) Stage 2 Overview — SPADE-Refiner 전체 흐름

![Row E: Stage 2 Overview](/home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/row_e_stage2_overview.png)

**각 블록 설명:**
- **Frozen CDiTUNet**: 1단계에서 학습된 예측 모델로, 입력 데이터를 통해 초기 기상 예측 맵($\hat{\mathbf{x}}_0$)과 불확실성($\log \hat{\sigma}$)을 출력합니다. 2단계 학습 시 가중치가 고정(Frozen)됩니다.
- **Context Map Extraction**: 1단계의 중간 피처 맵($h_{\text{dec}}$)과 최종 예측결과, 고도(Topo) 등의 추가 정보를 결합하여 SPADE 모듈에서 사용할 공간적 맥락 맵($\mathbf{m}_{\text{ctx}}$)을 생성합니다.
- **SPADE ResBlock**: 기상 데이터의 노이즈 $\mathbf{z}_T$를 입력으로 받아 해상도를 유지하며 디코딩을 수행합니다. 이때 $\mathbf{m}_{\text{ctx}}$를 조건으로 사용하여 픽셀 단위로 스케일($\gamma$)과 시프트($\beta$)를 조절합니다.
- **Output**: 정제된 최종 기상 예측 $\tilde{\mathbf{x}}_0$를 생성하여 1단계의 해상도 저하 및 흐림 현상(blur)을 극복하고 고주파수(디테일)를 복원합니다.

---

## (f) SPADE Block Detail — per-pixel modulation

![Row F: SPADE Block Detail](/home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/row_f_spade_detail_refined.png)

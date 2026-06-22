# Stage 1 CRPS Loss Function — 코드베이스 기준 설계 문서

> **소스 파일**: [gaussian.py](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py) / [config.py](file:///DATA/ST-DiT2/ST-DiT/stdit/config.py)
> **활성화 조건**: `USE_CRPS_LOSS=True` → `PREDICTION_TYPE=x0` 및 `LEARN_VARIANCE=True` 자동 강제

---

## 1. 네트워크 출력: Dual-Channel Head

CDiTUNet의 최종 출력은 **2채널** 텐서로 분리됩니다.

> **코드 참조**: [gaussian.py L351–353](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L351-L353)

```python
learn_variance = model_output_full.shape[1] == x_start.shape[1] * 2
if learn_variance:
    model_output, model_var_values = torch.split(model_output_full, x_start.shape[1], dim=1)
```

| 채널 | 코드 변수명 | 수학 기호 | Shape | 의미 |
|:---:|---|:---:|:---:|---|
| `[:, 0, :, :]` | `model_output` | $\hat{\mathbf{x}}_0$ (= $\mu$) | $(B, 1, H, W)$ | **점 예측** — 깨끗한 영상의 조건부 평균 |
| `[:, 1, :, :]` | `model_var_values` | $\log\hat{\sigma}_{\text{raw}}$ | $(B, 1, H, W)$ | **로그 불확실성** — 픽셀별 예측 불확실성의 원시값 |

> [!NOTE]
> `USE_CRPS_LOSS=True` 설정 시 [config.py L817–820](file:///DATA/ST-DiT2/ST-DiT/stdit/config.py#L817-L820)에서 `LEARN_VARIANCE=True`가 자동 강제되어, 네트워크의 `out_channels`이 2로 설정됩니다. $\log\hat{\sigma}$를 직접 출력하여 양수성($\hat{\sigma} = e^{\log\hat{\sigma}} > 0$)과 수치 안정성을 동시에 보장합니다.

---

## 2. σ Anchoring (Design B): 불확실성 추정의 출발점 설정

네트워크가 출력한 $\log\hat{\sigma}_{\text{raw}}$을 최종 $\log\hat{\sigma}_{\text{eff}}$로 변환하는 단계입니다.

> **코드 참조**: [gaussian.py L373–379](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L373-L379)

### Mode A: `crps_sigma_anchor = "none"` (기본값)

$$\log\hat{\sigma}_{\text{eff}} = \log\hat{\sigma}_{\text{raw}}$$

네트워크가 출력한 원시 로그 표준편차를 **그대로** 사용합니다. 네트워크가 $\hat{\sigma}$의 절대 스케일을 처음부터 스스로 학습해야 합니다.

### Mode B: `crps_sigma_anchor = "posterior_std"`

$$\log\hat{\sigma}_{\text{eff}} = \underbrace{\log\!\left(\frac{\sqrt{1-\bar{\alpha}_\tau}}{\sqrt{\bar{\alpha}_\tau}}\right)}_{\text{Prior: }\log\sigma_{\text{post}}(\tau)} \;+\; \alpha_{\text{mod}} \cdot \log\hat{\sigma}_{\text{raw}}$$

| 항 | 의미 |
|---|---|
| $\log\sigma_{\text{post}}(\tau)$ | 디퓨전 타임스텝 $\tau$에서의 **posterior 표준편차** — 노이즈 스케줄로부터 결정적으로 계산 |
| $\alpha_{\text{mod}}$ | 네트워크 보정의 강도 계수 (config: `CRPS_SIGMA_MOD_STRENGTH`, 기본값 1.0) |
| $\log\hat{\sigma}_{\text{raw}}$ | 네트워크가 출력한 **잔차적 보정값** |

> [!TIP]
> 이 모드의 설계 의도: $\hat{\sigma}$의 "기본값(anchor)"을 디퓨전 프로세스의 물리량인 posterior std로 잡아두고, 네트워크는 그 **편차(보정값)만 학습**하도록 부담을 줄입니다. 학습 초기 안정성이 향상됩니다.

---

## 3. 핵심 수식: `gaussian_crps_loss()`

> **코드 참조**: [gaussian.py L37–55](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L37-L55)

### 가우시안 CRPS 닫힌 형태 (Gneiting & Raftery, 2007)

$$\mathrm{CRPS}\!\left(\mathcal{N}(\hat{\mathbf{x}}_0,\, \hat{\sigma}^2),\; \mathbf{x}_{\text{gt}}\right) = \hat{\sigma} \left[ z \cdot (2\Phi(z) - 1) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right]$$

### 노테이션 정의

| 기호 | 코드 구현 | 수식 정의 | 의미 |
|:---:|---|:---:|---|
| $\hat{\mathbf{x}}_0$ | `mu` | — | 네트워크의 점 예측 (채널 0) |
| $\hat{\sigma}$ | `sigma = exp(log_sigma).clamp(min=1e-6)` | $e^{\log\hat{\sigma}_{\text{eff}}}$ | 예측 표준편차 (양수 보장) |
| $\mathbf{x}_{\text{gt}}$ | `y` | — | 실제 관측값 (Ground Truth) |
| $z$ | `z = (y - mu) / sigma` | $\dfrac{\mathbf{x}_{\text{gt}} - \hat{\mathbf{x}}_0}{\hat{\sigma}}$ | **표준화 잔차** — 예측 오차를 불확실성 단위로 정규화 |
| $\phi(z)$ | `exp(-0.5*z*z) * (1/√(2π))` | $\dfrac{1}{\sqrt{2\pi}} e^{-z^2/2}$ | 표준정규분포 **PDF** |
| $\Phi(z)$ | `0.5 * (1 + erf(z/√2))` | $\dfrac{1}{2}\!\left[1 + \mathrm{erf}\!\left(\dfrac{z}{\sqrt{2}}\right)\right]$ | 표준정규분포 **CDF** |

### 안전장치

```python
log_sigma = torch.clamp(log_sigma, max=5.0)   # σ ≤ e^5 ≈ 148 (폭주 방지)
sigma = torch.exp(log_sigma).clamp(min=1e-6)   # σ ≥ 1e-6 (0 나눗셈 방지)
```

### 출력

- **Shape**: 입력과 동일한 $(B, 1, H, W)$ — **픽셀별(per-element) 스칼라 손실**
- Caller(`p_losses`)가 reduction(공간 평균 등)을 결정

---

## 4. CRPS의 통계적 직관: 3가지 균형 메커니즘

CRPS 수식 내부에서 $\hat{\sigma}$(불확실성)와 $z$(표준화 잔차)가 어떻게 상호작용하는지 정리합니다.

| 시나리오 | $\hat{\sigma}$ | $|z|$ | CRPS 반응 | 해석 |
|---|:---:|:---:|---|---|
| **정확하고 확신** | 작음 ✓ | 작음 ✓ | **낮음** ✓ | 이상적 상태 — 예측이 맞고 불확실성도 작음 |
| **부정확하지만 확신** | 작음 | **큼** ✗ | 괄호 항 $z(2\Phi-1)$ **폭발** | 과신(overconfident) 패널티 |
| **정확하지만 불확신** | **큼** ✗ | 작음 | 선행 계수 $\hat{\sigma}$ **증폭** | 과소확신(underconfident) 패널티 |

> [!IMPORTANT]
> **결정론적 극한** ($\hat{\sigma} \to 0$):
> $$\lim_{\hat{\sigma} \to 0} \mathrm{CRPS} = |\mathbf{x}_{\text{gt}} - \hat{\mathbf{x}}_0| \quad (= \mathrm{MAE})$$
> 즉, CRPS는 **MAE의 확률적 일반화**입니다. 점 예측 정확도(MAE/MSE 방향)와 불확실성 보정(calibration)을 하나의 목적함수로 동시 최적화합니다.

---

## 5. t-Conditional Weighting (Design C): 타임스텝별 가중치

> **코드 참조**: [gaussian.py L382–391](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L382-L391)

디퓨전 타임스텝 $\tau$에 따라 CRPS에 추가 가중치를 곱하는 선택적 기능입니다.

$$\mathrm{CRPS}_{\text{weighted}} = \eta(\tau) \cdot \mathrm{CRPS}$$

| Config 값 (`CRPS_T_EXPONENT`) | $\eta(\tau)$ | 효과 |
|---|:---:|---|
| `"none"` (기본값) | $1$ | 가중치 없음 — 모든 타임스텝 동등 |
| `"snr_power"` | $\mathrm{SNR}(\tau)^{\beta}$ | 높은 SNR(쉬운 스텝) 강조 |
| `"inv_snr_power"` | $(1/\mathrm{SNR}(\tau))^{\beta}$ | 낮은 SNR(어려운 스텝) 강조 |

여기서 $\beta$는 `CRPS_T_EXP_BETA` (기본값 0.0).

```python
eta = eta / eta.mean().clamp(min=1e-8)  # 배치 내 평균=1 정규화 → loss 스케일 보존
eta = eta.view(-1, 1, 1, 1)             # (B,) → (B,1,1,1) broadcast
```

---

## 6. 공간 평균 → Min-SNR Reweighting

### Step 1: 픽셀별 → 샘플별 평균

> **코드 참조**: [gaussian.py L428](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L428)

$$\ell_b = \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} \mathrm{CRPS}_{b,c,h,w} \quad \text{Shape: } (B,)$$

### Step 2: Min-SNR Reweighting

> **코드 참조**: [gaussian.py L432–445](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L432-L445)

$$\ell_b^{\text{weighted}} = w_{\text{SNR}}(\tau_b) \cdot \ell_b$$

$$w_{\text{SNR}}(\tau) = \min\!\left(\mathrm{SNR}(\tau),\; \gamma_{\text{SNR}}\right), \quad \gamma_{\text{SNR}} = 5$$

| Prediction Type | 가중치 $w$ |
|---|---|
| `x0` (본 연구) | $\min(\mathrm{SNR}(\tau),\, 5)$ |
| `epsilon` | $\min(\mathrm{SNR}(\tau),\, 5) \;/\; \mathrm{SNR}(\tau)$ |
| `v` | $\min(\mathrm{SNR}(\tau),\, 5) \;/\; (\mathrm{SNR}(\tau) + 1)$ |

> [!NOTE]
> `x0`-prediction에서는 SNR 값 자체가 가중치가 됩니다. SNR이 큰(쉬운) 스텝은 $\gamma=5$에서 잘려 과도한 기여가 억제됩니다.

### Step 3: 미니배치 평균

> **코드 참조**: [gaussian.py L496](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L496)

$$\mathcal{L}_{\text{Stage1}} = \frac{1}{B}\sum_{b=1}^{B} \ell_b^{\text{weighted}}$$

---

## 7. 최종 손실 함수 — 통합 수식

$$\boxed{\mathcal{L}_{\text{Stage1}} = \frac{1}{B}\sum_{b=1}^{B} \underbrace{\min\!\left(\mathrm{SNR}(\tau_b),\, 5\right)}_{w_{\text{SNR}}(\tau_b)} \cdot \underbrace{\frac{1}{HW}\sum_{h,w} \hat{\sigma}_{b,h,w}\!\left[z_{b,h,w}\!\left(2\Phi(z_{b,h,w})\!-\!1\right) + 2\phi(z_{b,h,w}) - \frac{1}{\sqrt{\pi}}\right]}_{\text{Spatial-mean Gaussian CRPS}}}$$

$$z_{b,h,w} = \frac{(\mathbf{x}_{\text{gt}})_{b,h,w} - (\hat{\mathbf{x}}_0)_{b,h,w}}{\hat{\sigma}_{b,h,w}}, \qquad \hat{\sigma}_{b,h,w} = \exp\!\left((\log\hat{\sigma}_{\text{eff}})_{b,h,w}\right)$$

---

## 8. 전체 데이터 흐름 다이어그램

```
x_noisy (B,1,512,512) + x_ctx (B,5,512,512) + τ + season + lead
                │
                ▼
        ┌───────────────────────┐
        │   CDiTUNet (~22M)     │
        │   out_channels = 2    │  ← LEARN_VARIANCE=True 자동 강제
        │   prediction_type=x0  │  ← USE_CRPS_LOSS=True 자동 강제
        └───────────┬───────────┘
                    │  (B, 2, 512, 512)
                    ▼
        ┌───────────────────────┐
        │  torch.split(…, 1)    │
        ├───────────┬───────────┤
        │  채널 0   │  채널 1   │
        │  μ = x̂₀   │ log σ_raw │
        │(B,1,H,W)  │(B,1,H,W) │
        └─────┬─────┴─────┬─────┘
              │           │
              │     [σ Anchoring]
              │     ├─ "none": log σ_eff = log σ_raw
              │     └─ "posterior_std": log σ_eff = log σ_post(τ) + α_mod · log σ_raw
              │           │
              │     σ = exp(clamp(log σ_eff, max=5.0)).clamp(min=1e-6)
              │           │
              ▼           ▼
        ┌─────────────────────────────────────────┐
        │        gaussian_crps_loss(μ, σ, x_gt)   │
        │                                         │
        │  z = (x_gt − μ) / σ                     │
        │  φ(z) = (1/√2π) · exp(−z²/2)            │
        │  Φ(z) = ½ · (1 + erf(z/√2))             │
        │  CRPS = σ · [z·(2Φ−1) + 2φ − 1/√π]     │
        │                                         │
        └───────────────────┬─────────────────────┘
                            │  (B, 1, H, W) per-pixel loss
                            ▼
                  [Design C: t-Conditional Weighting]
                    CRPS × η(τ)  (optional)
                            │
                            ▼
                    mean(dim=[1,2,3])
                            │  (B,) per-sample loss
                            ▼
                  [Min-SNR Reweighting]
                    ℓ_b × min(SNR(τ_b), 5)
                            │
                            ▼
                        mean()
                            │  scalar
                            ▼
                     𝓛_Stage1  ← 최종 학습 손실
```

---

## 9. 전체 Config 키 요약

| Config 키 | 기본값 | 의미 |
|---|:---:|---|
| `USE_CRPS_LOSS` | `False` | CRPS 손실 활성화 (True 시 x0-pred + dual-head 자동 강제) |
| `CRPS_SIGMA_ANCHOR` | `"none"` | σ 앵커링 모드 (`"none"` / `"posterior_std"`) |
| `CRPS_SIGMA_MOD_STRENGTH` | `1.0` | 앵커링 시 네트워크 보정의 강도 계수 $\alpha_{\text{mod}}$ |
| `CRPS_T_EXPONENT` | `"none"` | 타임스텝별 가중치 모드 (`"none"` / `"snr_power"` / `"inv_snr_power"`) |
| `CRPS_T_EXP_BETA` | `0.0` | 타임스텝 가중치 지수 $\beta$ |
| `LOSS_WEIGHTING` | `"min_snr"` | Min-SNR reweighting 활성화 |
| `MIN_SNR_GAMMA` | `5.0` | SNR 상한 $\gamma_{\text{SNR}}$ |

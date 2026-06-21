# 슬라이드 14 — CRPS 손실 함수: 예상 질문 & 답변 전략

> 대상: [defense_outline.md 슬라이드 14](file:///home/user/.gemini/antigravity-ide/brain/72adefda-27b8-44e9-8133-d2231788679c/defense_outline.md)
> 코드: [gaussian_crps_loss()](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L37-L55)
> 논문: [03_methodology.tex §3.3](file:///DATA/ST-DiT2/thesis/thesis_2026/thesis_chapters/03_methodology.tex#L99-L147)

---

## Q1. "CRPS가 뭔지 한 문장으로 설명해 보세요."

**답변**:
> "CRPS는 예측 분포의 CDF와 관측값의 경험적 CDF (step function) 사이의 적분 제곱 거리를 측정하는 **proper scoring rule**입니다."

**보충 (꼬리 질문 대비)**:
- Proper scoring rule이란? → "기대 CRPS가 **진짜 분포일 때만 최소화**되는 성질. 즉 모델이 거짓말하면 손실이 올라갑니다."
- 수식: $\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} [F(x) - \mathbf{1}(x \ge y)]^2 \, dx$

---

## Q2. "왜 MSE나 MAE 대신 CRPS를 손실 함수로 썼나요?"

**답변**:
> "MSE/MAE는 **점 예측(point estimate)**만 평가합니다. 기상 예측에서는 '이 예측이 얼마나 확신할 수 있는가'라는 **불확실성 정보**가 매우 중요합니다. CRPS는 예측 정확도와 불확실성 보정(calibration)을 **하나의 목적함수로 동시에 최적화**할 수 있는 유일한 선택지였습니다."

**보충**:
- "CRPS의 결정론적 극한($\hat{\sigma} \to 0$)은 MAE로 수렴합니다. 즉 CRPS는 MAE의 **확률적 일반화**입니다."
- "기상학에서 CRPS는 **표준 평가 지표**(Gneiting & Raftery, 2007)이지만, 딥러닝 **학습 손실**로 사용한 사례는 드뭅니다. 이것이 저희 연구의 차별점 중 하나입니다."

---

## Q3. "가우시안 분포를 가정한 근거는? 실제 오차가 가우시안인지 검증했나요?"

**답변**:
> "각 픽셀의 예측 오차는 — 입력 컨텍스트와 diffusion timestep에 조건부(conditioned)인 상태에서 — **대략 단봉(unimodal)이고 대칭**입니다. 가우시안은 이 조건을 만족하는 가장 자연스러운 1차 분포 모델입니다."

**보충 (방어 포인트)**:
- "더 복잡한 분포(mixture, heavy-tail 등)를 쓸 수도 있지만, 가우시안의 장점은 **닫힌 형태(closed-form) CRPS**가 존재하여 학습이 안정적이고 역전파가 효율적이라는 점입니다."
- "논문 Table의 CRPS 값이 좋은 성능을 보이는 것 자체가 가우시안 가정이 크게 빗나가지 않았음을 간접적으로 보여줍니다."
- "Future work으로 mixture Gaussian이나 quantile regression CRPS를 시도해 볼 수 있습니다."

---

## Q4. "CRPS 수식의 각 항이 의미하는 바를 설명해 보세요."

**수식**: $\mathrm{CRPS} = \hat{\sigma}\left[z\,(2\Phi(z)-1) + 2\phi(z) - \frac{1}{\sqrt{\pi}}\right], \quad z = \frac{y - \hat{x}_0}{\hat{\sigma}}$

**답변**:
> "세 가지 요소가 경쟁합니다."

| 요소 | 역할 |
|---|---|
| **앞의 $\hat{\sigma}$** | 전체 스케일. $\hat{\sigma}$가 클수록 손실 자체가 비례 증가 → **과도하게 넓은 분포 억제** |
| **$z \cdot (2\Phi(z) - 1)$** | 표준화 잔차 $z$가 클수록(=예측이 빗나갈수록) 급격히 증가 → **점 예측 정확도** 패널티 |
| **$2\phi(z) - 1/\sqrt{\pi}$** | $z=0$에서 최대, $|z|$ 커지면 감소. 분포의 **중심(sharpness)** 보상 |

> "$\hat{\sigma}$가 너무 작으면 $|z|$가 폭발하고, 너무 크면 앞의 $\hat{\sigma}$ 자체가 손실을 키웁니다. **정확할 때 확신하고, 불확실할 때 솔직한 예측**이 최적입니다."

---

## Q5. "네트워크가 $\hat{\sigma}$를 직접 출력하나요? 수치적으로 안정한가요?"

**답변**:
> "아닙니다. 네트워크는 **$\log\hat{\sigma}$를 출력**하고, 손실 계산 시 $\hat{\sigma} = \exp(\log\hat{\sigma})$로 변환합니다. 이렇게 하면 $\hat{\sigma} > 0$이 자동으로 보장되고, 매우 작은 $\hat{\sigma}$에서도 gradient가 안정적입니다."

**코드 근거** ([gaussian.py L48-49](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L48-L49)):
```python
log_sigma = torch.clamp(log_sigma, max=5.0)  # σ_max ≈ 148, 발산 방지
sigma = torch.exp(log_sigma).clamp(min=1e-6)  # σ_min = 1e-6, 0 방지
```

> "추가로 `log_sigma_max=5.0`으로 상한을 걸어 학습 초기에 $\hat{\sigma}$가 발산하는 것을 방지합니다."

---

## Q6. "Min-SNR 재가중치는 왜 필요한가요? CRPS만으로는 부족한가요?"

**답변**:
> "Diffusion 모델 학습에서 timestep $\tau$에 따라 gradient의 크기가 **수십 배** 차이가 납니다. 높은 SNR(낮은 노이즈) 구간의 gradient가 지배적이 되면 학습이 불안정해집니다. Min-SNR(Hang et al., 2024)은 $w(\tau) = \min(\mathrm{SNR}(\tau),\, \gamma)$로 **고SNR 구간의 기여를 상한**하여 모든 timestep에서 균등하게 학습되도록 합니다."

**수식**: $\mathcal{L}_{\text{S1}} = w_{\text{SNR}}(\tau) \cdot \mathrm{CRPS}, \quad w_{\text{SNR}}(\tau) = \min(\mathrm{SNR}(\tau),\; 5)$

> "이것은 CRPS 자체의 문제가 아니라, **확산 모델 학습의 보편적인 문제**에 대한 처방입니다. MSE 손실을 쓰더라도 동일하게 적용됩니다."

---

## Q7. "학습된 $\hat{\sigma}$를 실제로 활용하나요? 추론 시에는 어떻게 쓰이나요?"

**답변**:
> "추론 시에는 **점 예측 $\hat{x}_0$만 사용**하고, $\hat{\sigma}$는 학습 시 CRPS 최적화를 위한 도구입니다. 다만 $\hat{\sigma}$는 두 가지로 활용 가능합니다:
> 1. **신뢰도 맵(confidence map)**: 높은 $\hat{\sigma}$ 영역은 모델이 불확실해하는 곳 → 구름 경계, 급변 지역
> 2. **앙상블 가중치**: 다중 샘플 생성 시 $\hat{\sigma}$로 가중 평균하면 성능 향상 가능
>
> 본 논문에서는 Stage 2 SPADE-Refiner의 입력으로 $\hat{x}_0$만 전달하지만, $\hat{\sigma}$ 맵을 Context Encoder에 추가 채널로 넣는 것은 자연스러운 확장입니다."

---

## Q8. "x₀-prediction을 채택한 이유는? v-prediction이나 ε-prediction과 비교했나요?"

**답변**:
> "CRPS 손실은 **모델 출력이 $x_0$여야** 닫힌 형태로 계산됩니다. $\epsilon$-prediction이나 v-prediction에서는 $x_0$를 역산해야 하는데, 이 과정에서 **numerically unstable**한 나눗셈($\div \sqrt{\bar{\alpha}_t}$)이 발생하여 CRPS gradient가 불안정해집니다.

> 코드에서도 `use_crps_loss=True`이면 `prediction_type=x0`를 강제합니다."

**코드 근거** ([gaussian.py L128-129](file:///DATA/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L128-L129)):
```python
if self.use_crps_loss and prediction_type != "x0":
    raise ValueError(f"use_crps_loss requires prediction_type=x0")
```

---

## Q9. "CRPS가 MSE 대비 실질적으로 성능이 좋았나요? 실험적 비교가 있나요?"

**답변**:
> "Ablation에서 MSE 학습 vs CRPS 학습을 비교했을 때, CRPS는 **MSE/MAE 지표에서는 동등하거나 약간 우수**했고, **SSIM에서는 명확한 개선**을 보였습니다. 이는 CRPS가 불확실성 보정을 통해 모델이 '확신 없는 영역에서 무리한 예측'을 하지 않도록 유도하여, 결과적으로 구조적 일관성(structural coherence)이 향상되었기 때문입니다."

**방어 포인트**: 만약 수치가 정확히 기억나지 않으면 →
> "핵심은 성능 수치의 미세한 차이보다, CRPS가 제공하는 **부가 정보(불확실성 맵)**와 **이론적 정당성(proper scoring rule)**에 있습니다."

---

## Q10. "CRPS는 픽셀별 독립으로 계산하는데, 공간적 상관성은 무시되는 것 아닌가요?"

**답변**:
> "맞습니다. CRPS는 per-pixel loss이므로 공간적 상관(spatial correlation)을 직접 모델링하지 않습니다. 하지만 이 문제는 **두 가지 메커니즘으로 보완**됩니다:
> 1. **CDiTUNet 자체의 공간적 귀납 편향**: Conv Encoder의 지역적 수용장 + DiT의 전역 Self-Attention이 공간 구조를 학습
> 2. **Stage 2 SPADE-Refiner**: 공간적으로 적응적인(spatially adaptive) 보정을 수행하여 per-pixel loss의 한계를 보완
>
> 공간적 손실(e.g., SSIM loss, perceptual loss)을 CRPS에 추가하는 것은 유효한 future work입니다."

---

## 보너스: 예상 꼬리 질문

### "Proper scoring rule과 strictly proper scoring rule의 차이는?"
> "Strictly proper는 **유일하게 진짜 분포에서만** 기대 손실이 최소. CRPS는 strictly proper입니다. 반면 log-score(NLL)도 strictly proper이지만 outlier에 극도로 민감합니다."

### "NLL(Negative Log-Likelihood) 대신 CRPS를 쓴 이유는?"
> "NLL은 outlier 하나에 $-\log(\text{작은 값}) = \infty$로 발산할 수 있습니다. CRPS는 **bounded**하고, MAE 스케일이라 해석이 직관적이며, 기상학 커뮤니티의 표준입니다."

### "확산 모델과 CRPS를 함께 쓴 선행 연구가 있나요?"
> "Diffusion 모델에서 CRPS를 학습 손실로 직접 쓴 사례는 거의 없습니다. 대부분은 $\epsilon$-prediction MSE를 사용합니다. 저희는 $x_0$-prediction + dual-head (μ, log σ) 설계로 이를 가능하게 했으며, 이것이 본 연구의 방법론적 기여 중 하나입니다."

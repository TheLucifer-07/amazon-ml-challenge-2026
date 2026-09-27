# 📊 Confusion Matrix & Classification Measures Report
### Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

## 1. Pairwise Confusion Matrix
Evaluated on the held-out validation set (49,600 candidate pairs) at the optimal threshold **0.660**:

| Actual \ Predicted | Predicted Match (Positive) | Predicted Non-Match (Negative) | Total Actual |
|:---|:---:|:---:|:---:|
| **Actual Match (True Positive Link)** | **TP = 2,830** | **FN = 279** | **3,109** |
| **Actual Non-Match (Negative Link)** | **FP = 37** | **TN = 46,454** | **46,491** |
| **Total Predicted** | **2,867** | **46,733** | **49,600** |

---

## 2. Quantitative Performance Measures

| Measure | Formula | Value | Percentage | Challenge Context |
|:---|:---|:---:|:---:|:---|
| **Macro $F_{0.5}$** | $\frac{1.25 \times P \times R}{0.25 \times P + R}$ | **0.9434** | **94.34%** | **Primary Competition Metric** |
| **Precision (PPV)** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | **0.9871** | **98.71%** | 2× heavier penalty on False Merges |
| **Recall (Sensitivity)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | **0.9103** | **91.03%** | Captures 91% of all positive match links |
| **Specificity (TNR)** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | **0.9992** | **99.92%** | Near-zero false alarms among negative candidates |
| **Singleton Accuracy** | $\frac{\text{Correct Singletons}}{\text{Total Singletons}}$ | **0.9658** | **96.58%** | **254 / 263** true singletons correctly predicted as `""` |
| **Pairwise Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{Total}}$ | **0.9936** | **99.36%** | Overall pairwise decision accuracy |
| **False Positive Rate (FPR)** | $\frac{\text{FP}}{\text{FP} + \text{TN}}$ | **0.000796** | **0.080%** | Rate of incorrect link proposals |
| **False Discovery Rate (FDR)** | $\frac{\text{FP}}{\text{TP} + \text{FP}}$ | **0.0129** | **1.29%** | Only 1.29% of predicted matches are false alarms |
| **False Negative Rate (FNR)** | $\frac{\text{FN}}{\text{FN} + \text{TP}}$ | **0.0897** | **8.97%** | Missed matches rate |

---

## 3. Analysis & Key Takeaways
1. **Asymmetric Precision Alignment:**
   Because Macro $F_{0.5}$ applies a 2× heavier penalty on False Merges (FP) than Missed Matches (FN), our optimal threshold sweep shifted the decision threshold from $0.50$ to **$0.660$**. This suppressed False Positives to just **37 cases** ($0.079\%$ FPR).
2. **Singleton Handling:**
   A false match prediction on a true singleton yields an entity score of $0.0$, whereas predicting empty string `""` yields $1.0$. The model achieved **96.58% accuracy** on true singletons.

# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolution Baseline Team  
**Team Members:** Baseline Model Submitter  
**Submission Date:** September 2026

---

## 1. Executive Summary
We present an offline-first, compliant Entity Resolution (ER) system linking Source 1 reference business entities with Source 2 and Source 3 candidate records. Our solution introduces a vectorized multi-pass inverted index blocking engine that slashes the pairwise comparison space by >99.99%, a 32-dimensional pairwise similarity feature extractor (RapidFuzz edit distances, token Jaccard overlap, address digit matching, and open-set country representations), and a LightGBM Gradient Boosted Decision Tree. By optimizing the decision threshold to 0.660 against the challenge's asymmetric Macro $F_{0.5}$ metric (which penalizes false merges 2× more heavily than missed matches) and enforcing strict singleton handling, the system achieves a validation Macro $F_{0.5}$ of 0.9434, 98.71% match precision, and 96.58% singleton accuracy.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights discovered during EDA and dataset profiling:
- **Scale & Cardinality:** Training dataset contains 2,206,821 Source 1 reference records, 5,034,616 Source 2 records, and 5,285,603 Source 3 records (10,320,219 total candidate records). Test set contains 1,732,544 Source 1 entities.
- **Singleton Distribution:** Ground truth analysis reveals 123,247 singletons (5.58%) with zero matching records. Under Macro $F_{0.5}$, predicting empty string `""` on true singletons yields 1.0, while predicting any false candidate yields 0.0.
- **Open-Set Country Domain Shift:** Training data covers exclusively `US` and `India`, whereas the test set introduces an open-set country label: `France` (`FR`), representing ~15% of test records. The normalization pipeline was engineered to dynamically handle arbitrary ISO country strings without hardcoded lookup failures.
- **Noise Patterns:** Observed high diacritic variance (`é`, `ü`, `è`), corporate legal suffix variations (`Inc.`, `LLC`, `Corp`, `Limited`, `Pvt Ltd`, `GmbH`, `SARL`), and address structural discrepancies (abbreviated street types, landmark references, and omitted postal digits).

### 2.2 Solution Strategy
- **Approach Type:** Vectorized Multi-Pass Inverted Index Blocking + Pairwise Gradient Boosted Decision Tree Classifier + Asymmetric Macro $F_{0.5}$ Threshold Optimization.
- **Core Innovation:** Leakage-free `GroupShuffleSplit` partitioning strictly grouped by `source1_entity_id`, fast vectorized multi-pass inverted index retrieval, 32-dimensional RapidFuzz-accelerated string/token/digit similarity feature matrix, and precision-heavy thresholding.

---

## 3. Candidate Generation (Blocking)
To reduce the $O(N \times M)$ comparison space, candidate generation utilizes a 3-pass inverted index blocking strategy:
- **Blocking keys used:**
  1. *Pass 1 (Exact Normalized Name):* Exact Unicode-normalized NFKD business name combined with normalized country (`norm_country::norm_name`).
  2. *Pass 2 (Clean Name):* Business name stripped of all corporate legal entity suffixes (`norm_country::clean_name`).
  3. *Pass 3 (Alphanumeric Prefix Key):* First 4 characters of cleaned business name (`norm_country::clean_prefix_4`), queried selectively when earlier passes yield $<20$ candidates.
- **Candidate pairs generated:** Mean candidate volume of 24.8 candidates per entity (Median: 3, P95: 121, upper cap enforced at 100 per entity).
- **How you ensured true matches were not lost:** Multi-pass union with fallback logic ensures exact and clean match variants are captured before selective prefix expansion.

---

## 4. Matching Model

**Features used (32 distinct numerical features):**
- **Name features:** Exact raw/normalized/clean match indicators, Jaro-Winkler similarity (full & clean name), Levenshtein normalized similarity (full & clean name), token Jaccard similarity, shared token count, token overlap ratio, token count difference, token length ratio, character length difference.
- **Address features:** Exact raw/normalized address match indicators, address Jaro-Winkler similarity, address Levenshtein similarity, address token Jaccard, shared address tokens, address token overlap ratio, shared extracted numeric digits (postal/building digits), digit Jaccard, digit count difference, address character length difference.
- **Country & Missingness features:** Exact country match flag, one-hot indicators for US, India, and France (open-set), missing address indicator flags (`source_addr_missing`, `candidate_addr_missing`, `both_addr_missing`).

**Model type:** LightGBM Classifier (`LGBMClassifier`, 300 estimators, learning rate 0.05, max depth 6, num leaves 63, subsample 0.8, colsample_bytree 0.8).  
**Threshold selection method:** Global grid sweep across $[0.05, 0.95]$ with step $0.01$ evaluated against the official competition Macro $F_{0.5}$ metric on held-out validation entities, identifying an optimal decision threshold of **0.660**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9434** on held-out validation entities (Match Precision: **98.71%**, Match Recall: **91.03%**, Singleton Accuracy: **96.58%** / 254 of 263 correct).
- **Pairwise Confusion Matrix (Validation set, 49,600 candidate pairs @ threshold 0.660):**
  - True Positives (TP): 2,830
  - False Positives / False Merges (FP): 37 (FPR: 0.079%, FDR: 1.29%)
  - False Negatives / Missed Matches (FN): 279 (FNR: 8.97%)
  - True Negatives (TN): 46,454
- **Common false positives (wrong merges):** Multi-location franchise chains or branches sharing identical brand names situated at different street addresses where numeric building digits were partially missing or unparsed.
- **Common false negatives (missed matches):** Severe typographical corruption in address fields combined with non-standard acronym variations in business titles.

---

## 6. Conclusion
The developed solution demonstrates that combining vectorized multi-pass inverted index blocking with high-dimensional pairwise string/token/digit similarity features and asymmetric metric-aligned thresholding achieves robust entity resolution performance (0.9434 Macro $F_{0.5}$, 98.71% precision). The pipeline is fully offline-compliant, reproducible, and strictly satisfies all submission formatting rules.

---

## Appendix

### A. Code Artefacts
Complete runnable code structure:
- `src/` — modular libraries: `data` (streaming loaders, validators), `preprocessing` (Unicode NFKD, address/legal normalizers), `blocking` (inverted index generators), `pairs` (candidate pair builders), `features` (32-D RapidFuzz feature extractors), `models` (GroupShuffleSplit, LightGBM training), `evaluation` (Macro $F_{0.5}$, singleton scoring, threshold optimizer), and `inference` (aggregator, TSV writers).
- `scripts/` — pipeline execution entry points `01_inspect_data.py` through `10_validate_submission.py`.
- `output/` — final submission outputs: `matching_results.tsv` (scored matches) and `candidate_pairs.tsv` (blocking candidate set).
- `requirements.txt` & `README.md` — pinned dependency environment and reproduction guidelines.

### B. Additional Results
Top 5 most important features by LightGBM split gain:
1. `addr_char_len_diff` (1,257) — Address character length difference
2. `name_jaro_winkler` (1,162) — Business name Jaro-Winkler similarity
3. `addr_levenshtein` (1,008) — Address Levenshtein distance
4. `name_levenshtein` (992) — Business name Levenshtein distance
5. `addr_jaccard` (983) — Address token Jaccard similarity

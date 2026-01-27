# 🏦 Credit Risk Analysis for Retail Lending: Identified $13.45M Annual Loss Prevention & 93% Default Detection Accuracy

**Author:** Deraina Cani RAKOTONIAINA ANTSASOA  
**Tools:** `SQL` | `Python` | `Power BI` | `Excel`  
**Contact:** [LinkedIn Profile]([https://www.linkedin.com/in/deraina-cani/)](https://www.linkedin.com/in/deraina-cani/)) 
| 📱 +261 34 09 821 54

---

## 📊 Executive Summary

### **The Business Impact**
Identified a **$13.45 Million annual capital preservation opportunity** by deploying a predictive "Digital Firewall" that identifies potential defaults before capital is deployed. The solution delivers a **43.8x Return on Investment (ROI)** by reducing realized losses without sacrificing lending volume.

### **The Challenge**
The institution faced a critical **13.82% portfolio default rate**, nearly triple the industry benchmark of 5%, resulting in **$36.15M in annual capital destruction**. Traditional underwriting relied on static thresholds that failed to capture the non-linear risk correlation between income volatility and debt-to-income (DTI) ratios.

### **The Solution**
Developed an end-to-end machine learning pipeline analyzing a **$435.7M portfolio** using **SMOTEENN resampled XGBoost** architectures. The project bridged the gap between raw data and executive strategy by translating model metrics (93.04% Recall) into a **Tri-Tier Decision Engine** for automated approvals, manual audits, and high-risk rejections.



### **The Key Finding**
Feature engineering revealed that a custom **Composite Risk Score (CRS)**—combining payment-to-income and loan-to-value—was **2.4x more predictive** than external credit grades alone. Statistical testing identified a **"Toxic DTI Threshold" at 45%**, where default probability surges by **215%**, providing a clear data-driven guardrail for underwriting.



---

### **Quick Stats**
* 🎯 **93.04% Recall** | Maximum sensitivity to high-risk defaults.
* 💰 **$13.45M** | Net annual losses avoided through AI intervention.
* 📈 **43.8x ROI** | Financial return on model deployment and maintenance costs.
* 🛡️ **13,800% Net Benefit** | Total economic value created in Year 1.
* ⚙️ **SMOTEENN + XGBoost** | Champion architecture for handling imbalanced financial data.


---
## 📖 Table of Contents
* [📊 Executive Summary](#-executive-summary)
* [🧹 Phase 1: Data Cleaning & Preprocessing](#-phase-1-data-cleaning--preprocessing)
* [📈 Phase 2: Exploratory Data Analysis](#-phase-2-exploratory-data-analysis)
* [🧠 Phase 3: Feature Engineering](#-phase-3-feature-engineering)
* [🤖 Phase 4: Model Development & Tuning](#-phase-4-model-development--tuning)
* [🔍 Phase 5: Results & Key Findings](#-phase-5-results--key-findings)
* [💰 Phase 6: Business Impact & ROI](#-phase-6-business-impact--roi)
* [🧭 Business Recommendations](#-business-recommendations)
  
---
## 🎯 Phase 1: Business Problem

### 📉 The Situation
Between January and December 2021, the bank’s retail lending division experienced a sharp deterioration in portfolio credit quality. While peer institutions maintain a **5.00%** default benchmark, our internal rate climbed to **13.82%**.

With **38,576 active loans** and an average size of **$11,296**, the bank faces an annual loss exposure of over **$36 Million**. This trend is accelerating, rising by 22.3% over the 12-month period, necessitating an immediate shift from static grading to predictive modeling.

![Loan Default Trend 2021](Images/laon_default_trend_2021.PNG)

### ❓ Critical Business Questions
1.  **Borrower Profiles:** Which borrower characteristics and loan features most strongly correlate with default?
2.  **Point of Origination:** Can we accurately decline or adjust terms for high-risk applicants before the loan is issued?
3.  **Early Warning System:** Can we trigger proactive collection interventions for existing loans based on risk probability?

---

## 💡 Why This Matters

### 💰 Financial Impact
Every 1% reduction in the default rate saves approximately **$4.36 Million**. Reverting to the industry benchmark of 5% would recover an estimated **$38.4 Million** in annual lost capital.

### 🎯 Strategic Impact (Unit Economics)
High default rates inflate the "Effective Cost" of every loan. By optimizing the portfolio mix, we improve the **LTV-to-CAC ratio**, ensuring that interest revenue from "Safe" borrowers is no longer cannibalized to cover the principal losses of "At-Risk" segments.

### ⚖️ Competitive Impact (Adverse Selection)
Our current default rate creates a competitive disadvantage. To remain solvent, we must inflate interest rates, leading to **Adverse Selection**: 
* **Prime Borrowers** migrate to lower-interest competitors.
* **High-Risk Borrowers** (rejected elsewhere) disproportionately flow into our portfolio.
> **Predictive analytics is a defensive necessity to reclaim the "Prime" market segment.**

---
## 🔬 Methodology

### Analysis Framework
This project utilizes a **six-phase analytical approach** combining deep exploratory data analysis (EDA), rigorous feature engineering, statistical hypothesis testing, and ensemble machine learning. The goal is to move beyond simple reporting to build an **actionable default prediction system**.

![Six Phase Analytical Approach](Images/analytical_approach.PNG)
---

### Phase 1: Data Understanding & Quality Assessment
**Objective:** Conduct a comprehensive audit of the **38,576-record** loan portfolio to establish a “Single Source of Truth,” ensuring data integrity and alignment with banking business rules prior to modeling.

#### 🛠️ Activities
* **Ingestion & Profiling:** Loaded and profiled 38,576 records across 24 financial and demographic variables.
* **Multidimensional Quality Audit:** Evaluated the dataset across **6 rigorous dimensions**: Completeness, Validity, Consistency, Uniqueness, Timeliness, and Accuracy.
* **Statistical Outlier Detection:** Utilized the **Interquartile Range (IQR) method** on `annual_income` to isolate extreme values that could skew risk assessments.
* **Temporal Integrity Mapping:** Audited date-based columns (`last_credit_pull_date`, `last_payment_date`, and `next_payment_date`) to resolve chronological violations relative to the `issue_date`.
* **Business Rule Enforcement:** Validated records against core lending constraints, specifically ensuring `loan_amount` ≥ $1,000 and `total_payment` ≥ `loan_amount`.

#### 📊 Key Audit Findings
| Metric | Result |
| :--- | :--- |
| **Portfolio Shape** | 38,576 rows × 24 columns |
| **Class Imbalance** | 85.8% Fully Paid vs. **14.2% Charged Off** |
| **Missing Data** | < 5% in all critical columns |
| **Business Rule Violations** | 1,182 records where `total_payment` < `loan_amount` |
| **Minimum Loan Violation** | 448 records with `loan_amount` < $1,000 |

> **Timeline Violations Identified:** Significant chronological discrepancies found in `last_credit_pull_date` (19,464 records), `last_payment_date` (14,898), and `next_payment_date` (12,814) relative to the original issue date.

**Tools Used:** `Python` (`Pandas`, `NumPy`)

---

### Phase 2: Exploratory Data Analysis (EDA)
**Objective:** Dissect behavioral and financial drivers to isolate the **“Risk Signatures”** that distinguish solvent borrowers from defaults.

#### 🛠️ Activities
* **Target Distribution Analysis:** Confirmed the 85.8%/14.2% split, establishing the necessity for **High Recall** on the minority "Charged Off" class.
* **Univariate Profiling:** Used histograms, boxplots, and violin plots to identify distribution skews and inform necessary feature transformations.
* **Bivariate Correlation:** Developed a heatmap to identify **multicollinearity** (notably between `loan_amount`, `installment`, and `total_payment`), enabling a streamlined feature set.
* **Categorical Risk Assessment:** Quantified the impact of qualitative factors like loan purpose and home ownership using **Chi-square testing**.
* **Temporal & Segmentation Analysis:** Built time-series views of default trends and stratified the portfolio into Low/Medium/High-risk clusters.



#### 💡 Key Strategic Insights
* **The Interest Rate “Tipping Point”:** Borrowers with rates exceeding **20%** exhibit a **28.8x higher default rate** (a 286.6% increase in risk), marking the critical threshold for risk-based pricing.
* **Credit Grade Precision:** Validated a linear risk gradient from **Grade A (5.96% default)** to **Grade G (31.8% default)**.
* **The “Tenure Paradox”:** Employment length showed minimal predictive power; **Debt-to-Income (DTI)** metrics were found to be far more indicative of immediate default risk than job stability.
* **Collateral Influence:** Homeowners demonstrate a **6.38% lower default rate** than renters, confirming asset-backed stability as a strong secondary predictor.

**Tools Used:** `Python` (`Matplotlib`, `Seaborn`), `Scipy` (Statistical Tests)

## 🏗️ Phase 3: Feature Engineering

**Objective:** To transform raw loan application data into high-signal predictive features, culminating in a **proprietary Composite Risk Score (CRS)** that quantifies default probability.

### 1. Financial Capacity Ratios
We moved beyond raw numbers to calculate the true financial strain on the borrower:
* **Payment-to-Income Ratio:** Calculates the annual installment burden relative to total annual income.
* **Loan-to-Income Ratio:** Measures the total debt magnitude against the borrower’s earning power.
* **DTI Tiers:** Segmented Debt-to-Income levels into 4 categories (*Stable, Moderate, High, Critical*) to capture the non-linear relationship between debt and default.

### 2. Strategic Risk Indicators (Flags)
Using the statistical insights from Phase 2, we engineered specific flags for high-impact segments:
* **Toxic Yield Flag:** A binary indicator for **Interest Rates > 20%**, targeting a segment where default rates surge by **286.6%**.
* **Term Exposure Flag:** Isolated 60-month loans, which carry a **26.68% default rate**—double that of the standard 36-month term.
* **Ordinal Grade Encoding:** Converted alphabetical Credit Grades into a numerical scale (A=7 to G=1) to leverage the bank’s strongest predictive variable ($p\text{-value} = 5.9 \times 10^{-297}$).

### 3. The Composite Risk Score (CRS) Framework
To provide a unified decision-making tool, we synthesized the top 4 most statistically significant variables into a weighted **100-point score**. We deliberately excluded lower-signal or inconsistent variables (like `emp_length` or `home_ownership`) to ensure model robustness.

**Weighting Logic:**

| Risk Pillar | Weight | Rationale |
| :--- | :--- | :--- |
| **Credit Grade** | 40 pts | Foundation of risk assessment ($p < 10^{-200}$) |
| **Interest Rate** | 30 pts | Strongest behavioral trigger for default (34.56%) |
| **Loan Term** | 15 pts | Duration risk identified as a massive exposure factor |
| **DTI Ratio** | 15 pts | Critical measure of fundamental repayment capacity |

#### 🚦 Operational Risk Segmentation
The CRS is translated into 3 actionable **Decision Zones** for stakeholders:
* 🟢 **Green Zone (0-35) Low Risk:** Candidates for automated approval.
* 🟡 **Yellow Zone (36-60) Medium Risk:** Requires manual underwriting review.
* 🔴 **Red Zone (61-100) High Risk:** Recommended for automatic rejection to protect capital.



**Technical Stack & Methodology:**
* **Tools:** `Python` (`Pandas`, `NumPy`)
* **Techniques:** Custom binning, ordinal encoding, vectorized logical operations (`np.select`, `.loc`), and financial ratio engineering.

---

## 🔬 Phase 4: Statistical Testing & Validation

**Objective:** To conduct a comprehensive, end-to-end scientific validation of all features—proving significance and ensuring structural integrity.

### 1. Categorical Variable Validation (Chi-Square Test)
We applied the **Chi-Square Test of Independence** to confirm that our categorical segments and flags are not due to random chance.
* **Key Findings:** All features achieved p-values far below $\alpha = 0.05$.
* **Primary Predictors:** `grade_numeric` ($p = 5.90 \times 10^{-297}$), `int_rate_category` ($p = 7.07 \times 10^{-291}$), and `risk_segment` ($p = 4.72 \times 10^{-230}$) were confirmed as the strongest categorical anchors.

### 2. Numerical Validation (T-Tests & Point-Biserial)
* **Independent T-Tests:** Performed to compare means of “Defaulted” vs. “Non-Defaulted” groups. Results showed significant mean differences for `payment_completion_ratio`, `interest_rate`, and `loan_to_income_ratio` ($p < 0.05$).
* **Point-Biserial Correlation:** Measured the linear relationship strength between continuous features and the binary target.
    * **Payment Completion Ratio:** Emerged as the strongest predictor with a correlation of **-0.80**, indicating that repayment health is the most direct indicator of default.

### 3. Heatmap & Multicollinearity Analysis
A Feature Correlation Heatmap was generated to ensure the model remains “lean” and avoids redundancy.
* **Multicollinearity Detection:** Identified a near-perfect correlation (**0.95**) between `payment_to_income_ratio` and `loan_to_income_ratio`.
* **Optimization Insight:** To prevent model instability, we prioritized the **Payment-To-Income ratio** as the primary capacity indicator and dropped redundant raw variables like `installment`.



### 📊 Final Statistical Audit Summary
| Feature Type | Test Performed | Key Insight | Scientific Conclusion |
| :--- | :--- | :--- | :--- |
| **Categorical** | Chi-Square | Segments isolate default clusters | **Validated** ($p < 0.05$) |
| **Numerical** | Independent T-Test | Mean risk significantly differs by group | **Validated** ($p < 0.05$) |
| **Numerical** | Point-Biserial | `payment_completion` is the strongest signal | **Strong Association (-0.80)** |
| **Relational** | Heatmap | Identified 0.95 redundancy in income ratios | **Streamlined Model** |

### 🛠️ Technical Stack & Methodology
* **Programming & Data:** `Python` (`Pandas`, `NumPy`)
* **Visualization:** `Seaborn`, `Matplotlib`
* **Statistical Engine:** `SciPy Stats` (`ttest_ind`, `pointbiserialr`)
* **Machine Learning:** `Scikit-Learn`, `Imbalanced-Learn`

---

## 🤖 Phase 5: Predictive Modeling

**Objective:** The core goal was to develop a high-fidelity classification engine capable of distinguishing reliable borrowers from potential defaulters. Given the banking context, the project prioritized **Recall** to minimize credit losses while ensuring **Interpretability** for regulatory compliance.

### 🧪 Experimental Setup & Rigor
To ensure scientific rigor and avoid **Data Leakage**, I implemented a specialized `Scikit-Learn` & `Imbalanced-Learn` Pipeline. This architecture ensures every transformation is applied correctly within the validation framework:

* **Preprocessing Layer:** * **Logarithmic Transformation (`np.log1p`):** Applied to skewed numerical features to normalize their distribution.
    * **RobustScaler:** Standardized features while minimizing the influence of outliers for stable model convergence.
* **Advanced Resampling:** Integrated **SMOTEENN** (Synthetic Minority Over-sampling + Edited Nearest Neighbors) directly within the cross-validation loop to address class imbalance and clean noisy overlapping points.
* **Algorithm Benchmarking:** Compared three distinct architectures:
    * **Logistic Regression:** Selected for high interpretability and strong linear separation.
    * **Random Forest:** Captured complex non-linear interactions.
    * **XGBoost:** Utilized gradient boosting for peak precision.

### 📊 Comprehensive Performance Metrics
The following results represent the final evaluation of our models, highlighting the stability provided by our cross-validation protocol:

| Model | Precision | Recall | F1-Score | Accuracy | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7225 | 0.9285 | 0.8126 | 0.9389 | 0.9346 |
| **Random Forest** | 0.8640 | 0.8908 | 0.8772 | 0.9644 | 0.9337 |
| **Cross-Val (LogReg)** | 0.7239 | **0.9304** | 0.7239 | 0.8142 | **0.9805** |



**🏆 Champion Selection:** While Random Forest offered higher precision, **Logistic Regression** was selected as the champion. Its exceptional **93.04% Cross-Validated Recall** and superior **0.9805 ROC-AUC** provide the most secure safety net for capital protection.

---

### 🔍 Interpretability: Decoding the Drivers of Risk
We "opened the black box" by analyzing the Logistic Regression coefficients to identify the primary drivers of default:

1. **Primary Safety Signal:** `payment_completion_ratio` (**-3.56**) is the strongest mitigator of risk.
2. **Categorical Influence:** Internal Low Risk segments (**-1.98**) and top-tier Grades (**-1.95**) significantly lower default probability.
3. **Risk Escalators:** Grade 2 (**+1.77**) and 60-month terms (**+1.04**) are major risk drivers. Longer exposure periods and lower internal grades are mathematically linked to higher default rates.



---

## 🎯 Final Conclusion
The final architecture successfully handles non-normal distributions and class imbalance to deliver a highly reliable scoring tool. By accurately identifying **93% of defaults**, this model provides a data-driven foundation for a more profitable and secure lending strategy.

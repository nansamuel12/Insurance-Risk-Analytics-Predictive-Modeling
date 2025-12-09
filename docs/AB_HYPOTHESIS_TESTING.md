# A/B Hypothesis Testing - Implementation Summary

## Overview
This document provides a comprehensive summary of the A/B Hypothesis Testing implementation for the Insurance Risk Analytics Predictive Modeling project.

## Implementation Details

### Module: `src/ab_hypothesis_testing.py`

A comprehensive statistical analysis module that performs hypothesis testing to identify significant differences between groups in the insurance dataset.

### Statistical Tests Implemented

#### 1. Chi-Square Test of Independence
- **Purpose**: Tests whether two categorical variables are independent
- **Null Hypothesis (H0)**: The two categorical variables are independent
- **Alternative Hypothesis (H1)**: The two categorical variables are dependent
- **Effect Size Metric**: Cramér's V
- **Tests Performed**:
  - Smoker vs Sex
  - Smoker vs Region

#### 2. Independent Samples T-Test
- **Purpose**: Compares means between two independent groups
- **Null Hypothesis (H0)**: Mean of group A = Mean of group B
- **Alternative Hypothesis (H1)**: Mean of group A ≠ Mean of group B
- **Effect Size Metric**: Cohen's d
- **Tests Performed**:
  - Insurance charges: Smokers vs Non-smokers
  - Insurance charges: Female vs Male
  - BMI: Smokers vs Non-smokers

#### 3. One-Way ANOVA
- **Purpose**: Compares means across multiple groups
- **Null Hypothesis (H0)**: All group means are equal
- **Alternative Hypothesis (H1)**: At least one group mean is different
- **Effect Size Metric**: Eta-squared (η²)
- **Tests Performed**:
  - Insurance charges across regions
  - BMI across regions

## Key Features

### 1. Automated Statistical Analysis
- Performs 7 comprehensive hypothesis tests automatically
- Calculates test statistics, p-values, and effect sizes
- Determines statistical significance at α = 0.05 level

### 2. Effect Size Calculations
- **Cohen's d**: Measures standardized difference between two means
  - Small: 0.2, Medium: 0.5, Large: 0.8
- **Cramér's V**: Measures association between categorical variables
  - Small: 0.1, Medium: 0.3, Large: 0.5
- **Eta-squared (η²)**: Proportion of variance explained
  - Small: 0.01, Medium: 0.06, Large: 0.14

### 3. Comprehensive Reporting
- **JSON Output** (`reports/ab_hypothesis_tests.json`): Machine-readable results
- **Text Report** (`reports/ab_hypothesis_report.txt`): Human-readable detailed report
- **Visualizations**:
  - P-value comparison chart
  - Effect size comparison chart
  - Group comparison charts for t-tests

### 4. Data-Driven Interpretations
Each test includes a detailed interpretation explaining:
- Test statistic and p-value
- Decision to reject or fail to reject null hypothesis
- Practical significance (effect size)
- Group statistics (means, standard deviations)

## Results Summary

### Significant Findings

#### 1. Smoking Status and Insurance Charges ✓ SIGNIFICANT
- **Test**: Independent T-Test
- **Statistic**: t = 4.8866
- **P-value**: 0.0001 (p < 0.001)
- **Effect Size**: Cohen's d = 2.73 (Very Large)
- **Interpretation**: Smokers have significantly higher insurance charges than non-smokers
- **Group Means**:
  - Smokers: $30,285.72 ± $10,256.98
  - Non-smokers: $7,994.10 ± $7,672.62
- **Business Impact**: Smoking status is a critical risk factor for insurance pricing

### Non-Significant Findings

#### 2. Smoking Status and Sex ✗ NOT SIGNIFICANT
- **Test**: Chi-Square Test
- **Statistic**: χ² = 0.0000
- **P-value**: 1.0000
- **Effect Size**: Cramér's V = 0.00
- **Interpretation**: Smoking status is independent of sex

#### 3. Smoking Status and Region ✗ NOT SIGNIFICANT
- **Test**: Chi-Square Test
- **Statistic**: χ² = 3.5714
- **P-value**: 0.3116
- **Effect Size**: Cramér's V = 0.42
- **Interpretation**: Smoking status is independent of region

#### 4. Sex and Insurance Charges ✗ NOT SIGNIFICANT
- **Test**: Independent T-Test
- **Statistic**: t = 0.5609
- **P-value**: 0.5818
- **Effect Size**: Cohen's d = 0.26 (Small)
- **Interpretation**: No significant difference in charges between males and females

#### 5. Smoking Status and BMI ✗ NOT SIGNIFICANT
- **Test**: Independent T-Test
- **Statistic**: t = 0.8873
- **P-value**: 0.3867
- **Effect Size**: Cohen's d = 0.50 (Medium)
- **Interpretation**: No significant difference in BMI between smokers and non-smokers

#### 6. Region and Insurance Charges ✗ NOT SIGNIFICANT
- **Test**: One-Way ANOVA
- **Statistic**: F = 0.5158
- **P-value**: 0.6773
- **Effect Size**: η² = 0.09
- **Interpretation**: No significant difference in charges across regions

#### 7. Region and BMI ✗ NOT SIGNIFICANT
- **Test**: One-Way ANOVA
- **Statistic**: F = 2.2491
- **P-value**: 0.1220
- **Effect Size**: η² = 0.30
- **Interpretation**: No significant difference in BMI across regions

## Business Insights

### Primary Risk Factor Identified
**Smoking Status** is the most significant predictor of insurance charges:
- Smokers pay approximately **3.8x more** than non-smokers
- Very large effect size (Cohen's d = 2.73) indicates strong practical significance
- This finding should inform:
  - Premium pricing strategies
  - Risk assessment models
  - Health intervention programs

### Factors with No Significant Impact
The following factors showed no significant impact on insurance charges:
- **Sex**: Male vs Female pricing can be similar
- **Region**: Geographic location doesn't significantly affect charges
- **BMI**: While important for health, BMI differences between groups aren't statistically significant in this dataset

## Technical Implementation

### Code Quality Features
1. **Object-Oriented Design**: `ABHypothesisTester` class encapsulates all testing logic
2. **Type Hints**: Full type annotations for better code clarity
3. **Dataclasses**: Structured result storage using `HypothesisTestResult`
4. **Error Handling**: Robust error handling for data loading and processing
5. **Documentation**: Comprehensive docstrings for all methods
6. **Modularity**: Separate methods for each test type
7. **Visualization**: Automated generation of publication-quality plots

### Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scipy.stats`: Statistical tests
- `matplotlib`: Plotting
- `seaborn`: Statistical visualizations

### Integration with DVC Pipeline
The A/B hypothesis testing is integrated into the DVC pipeline:
```yaml
ab_hypothesis_testing:
  cmd: python src/ab_hypothesis_testing.py
  deps:
    - data/raw/data.csv
    - src/ab_hypothesis_testing.py
  outs:
    - reports/ab_hypothesis_tests.json
    - reports/ab_hypothesis_report.txt
```

## Usage

### Running the Tests
```bash
# Run directly
python src/ab_hypothesis_testing.py

# Or through DVC pipeline
dvc repro ab_hypothesis_testing
```

### Output Files
1. **reports/ab_hypothesis_tests.json**: Structured JSON with all test results
2. **reports/ab_hypothesis_report.txt**: Detailed text report with interpretations
3. **reports/figures/hypothesis_test_pvalues.png**: P-value visualization
4. **reports/figures/hypothesis_test_effect_sizes.png**: Effect size visualization
5. **reports/figures/hypothesis_test_group_comparisons.png**: Group comparison charts

## Statistical Rigor

### Methodology
- **Significance Level**: α = 0.05 (95% confidence)
- **Two-Tailed Tests**: All tests are two-tailed for conservative estimates
- **Effect Sizes**: Reported alongside p-values for practical significance
- **Assumptions**: Tests assume:
  - Independence of observations
  - Normality for parametric tests (t-test, ANOVA)
  - Homogeneity of variance for t-tests

### Interpretation Guidelines
- **P-value < 0.05**: Reject null hypothesis (statistically significant)
- **P-value ≥ 0.05**: Fail to reject null hypothesis (not statistically significant)
- **Effect Size**: Provides context for practical significance beyond statistical significance

## Future Enhancements

### Potential Improvements
1. **Non-Parametric Tests**: Add Mann-Whitney U test, Kruskal-Wallis test for non-normal data
2. **Post-Hoc Tests**: Implement Tukey HSD for ANOVA follow-up
3. **Assumption Testing**: Add Shapiro-Wilk test for normality, Levene's test for homogeneity
4. **Multiple Comparison Correction**: Implement Bonferroni or FDR correction
5. **Power Analysis**: Calculate statistical power for each test
6. **Confidence Intervals**: Add confidence intervals for effect sizes
7. **Interactive Visualizations**: Create interactive plots with Plotly

## Conclusion

The A/B Hypothesis Testing implementation provides a robust, automated framework for statistical analysis of insurance risk factors. The key finding—that smoking status significantly impacts insurance charges—provides actionable insights for business decision-making while maintaining statistical rigor and reproducibility.

---
**Author**: nansamuel12  
**Date**: December 2025  
**Project**: Insurance Risk Analytics Predictive Modeling

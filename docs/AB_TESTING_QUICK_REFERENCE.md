# A/B Hypothesis Testing - Quick Reference

## What is A/B Hypothesis Testing?

A/B Hypothesis Testing is a statistical method used to compare two or more groups to determine if there are significant differences between them. In the context of insurance risk analytics, it helps identify which factors significantly impact insurance charges and risk assessment.

## Tests Performed

### 1. Chi-Square Test (χ²)
**Purpose**: Tests if two categorical variables are related  
**Example**: Is smoking status related to sex or region?  
**Result**: No significant relationship found

### 2. Independent T-Test
**Purpose**: Compares average values between two groups  
**Examples**:
- Do smokers pay more than non-smokers? ✓ **YES** (p < 0.001)
- Do males pay more than females? ✗ No (p = 0.58)
- Do smokers have higher BMI? ✗ No (p = 0.39)

### 3. ANOVA (Analysis of Variance)
**Purpose**: Compares average values across multiple groups  
**Examples**:
- Do charges differ across regions? ✗ No (p = 0.68)
- Does BMI differ across regions? ✗ No (p = 0.12)

## Key Finding 🔍

**Smokers pay 3.8x more than non-smokers!**
- Smokers: $30,286 ± $10,257
- Non-smokers: $7,994 ± $7,673
- Statistical significance: p < 0.001
- Effect size: Very Large (Cohen's d = 2.73)

## How to Read the Results

### P-Value
- **p < 0.05**: Statistically significant (reject null hypothesis)
- **p ≥ 0.05**: Not statistically significant (fail to reject null hypothesis)

### Effect Size
Measures the magnitude of the difference:
- **Small**: Detectable but minor difference
- **Medium**: Moderate difference
- **Large**: Substantial difference

## Output Files

1. **reports/ab_hypothesis_tests.json**: All results in JSON format
2. **reports/ab_hypothesis_report.txt**: Detailed text report
3. **reports/figures/hypothesis_test_pvalues.png**: P-value chart
4. **reports/figures/hypothesis_test_effect_sizes.png**: Effect size chart
5. **reports/figures/hypothesis_test_group_comparisons.png**: Group comparisons

## Running the Tests

```bash
# Option 1: Run directly
python src/ab_hypothesis_testing.py

# Option 2: Run through DVC pipeline
dvc repro ab_hypothesis_testing
```

## Business Implications

### ✅ Action Items
1. **Smoking Status**: Primary risk factor for pricing
   - Implement smoking status verification
   - Offer smoking cessation programs
   - Adjust premiums based on smoking status

### ℹ️ No Action Needed
2. **Sex**: No significant difference in charges
3. **Region**: Geographic location doesn't affect pricing
4. **BMI**: Not a significant differentiator in this dataset

---
For detailed technical documentation, see [AB_HYPOTHESIS_TESTING.md](AB_HYPOTHESIS_TESTING.md)

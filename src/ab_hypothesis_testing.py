"""
A/B Hypothesis Testing Module for Insurance Risk Analytics

This module implements comprehensive statistical hypothesis testing to compare
different groups in the insurance dataset. It includes:
- Chi-Square tests for categorical variables
- T-tests for numerical variables
- Statistical significance analysis
- Detailed reporting and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import os
import sys
from dataclasses import dataclass, asdict


@dataclass
class HypothesisTestResult:
    """Data class to store hypothesis test results."""
    test_name: str
    variable: str
    group_a: str
    group_b: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    effect_size: Optional[float] = None
    group_a_mean: Optional[float] = None
    group_b_mean: Optional[float] = None
    group_a_std: Optional[float] = None
    group_b_std: Optional[float] = None
    interpretation: str = ""


class ABHypothesisTester:
    """
    A/B Hypothesis Testing class for insurance risk analytics.
    
    This class performs various statistical tests to compare groups and
    determine if there are significant differences in insurance charges,
    risk factors, and other metrics.
    """
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize the A/B Hypothesis Tester.
        
        Args:
            data (pd.DataFrame): Insurance dataset
            alpha (float): Significance level (default: 0.05)
        """
        self.data = data.copy()
        self.alpha = alpha
        self.results: List[HypothesisTestResult] = []
        
    def chi_square_test(self, 
                       categorical_var1: str, 
                       categorical_var2: str) -> HypothesisTestResult:
        """
        Perform Chi-Square test of independence between two categorical variables.
        
        Null Hypothesis (H0): The two categorical variables are independent.
        Alternative Hypothesis (H1): The two categorical variables are dependent.
        
        Args:
            categorical_var1 (str): First categorical variable
            categorical_var2 (str): Second categorical variable
            
        Returns:
            HypothesisTestResult: Test results
        """
        # Create contingency table
        contingency_table = pd.crosstab(
            self.data[categorical_var1], 
            self.data[categorical_var2]
        )
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate Cramér's V as effect size
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
        
        is_significant = p_value < self.alpha
        
        interpretation = (
            f"Chi-Square test between {categorical_var1} and {categorical_var2}: "
            f"chi-square={chi2_stat:.4f}, p-value={p_value:.4f}. "
            f"{'REJECT' if is_significant else 'FAIL TO REJECT'} null hypothesis. "
            f"Variables are {'DEPENDENT' if is_significant else 'INDEPENDENT'} "
            f"at alpha={self.alpha}. Effect size (Cramer's V)={cramers_v:.4f}."
        )
        
        result = HypothesisTestResult(
            test_name="Chi-Square Test of Independence",
            variable=f"{categorical_var1} vs {categorical_var2}",
            group_a=categorical_var1,
            group_b=categorical_var2,
            statistic=chi2_stat,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=cramers_v,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def independent_t_test(self, 
                          numerical_var: str, 
                          grouping_var: str,
                          group_a_value: str,
                          group_b_value: str) -> HypothesisTestResult:
        """
        Perform independent samples t-test between two groups.
        
        Null Hypothesis (H0): Mean of group A = Mean of group B
        Alternative Hypothesis (H1): Mean of group A ≠ Mean of group B
        
        Args:
            numerical_var (str): Numerical variable to compare
            grouping_var (str): Variable that defines groups
            group_a_value (str): Value for group A
            group_b_value (str): Value for group B
            
        Returns:
            HypothesisTestResult: Test results
        """
        # Extract data for each group
        group_a_data = self.data[self.data[grouping_var] == group_a_value][numerical_var]
        group_b_data = self.data[self.data[grouping_var] == group_b_value][numerical_var]
        
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data)
        
        # Calculate Cohen's d as effect size
        mean_a = group_a_data.mean()
        mean_b = group_b_data.mean()
        std_a = group_a_data.std()
        std_b = group_b_data.std()
        
        pooled_std = np.sqrt(((len(group_a_data) - 1) * std_a**2 + 
                              (len(group_b_data) - 1) * std_b**2) / 
                             (len(group_a_data) + len(group_b_data) - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        is_significant = p_value < self.alpha
        
        interpretation = (
            f"Independent t-test for {numerical_var} between {group_a_value} and {group_b_value}: "
            f"t={t_stat:.4f}, p-value={p_value:.4f}. "
            f"Mean({group_a_value})={mean_a:.2f}+/-{std_a:.2f}, "
            f"Mean({group_b_value})={mean_b:.2f}+/-{std_b:.2f}. "
            f"{'REJECT' if is_significant else 'FAIL TO REJECT'} null hypothesis. "
            f"Means are {'SIGNIFICANTLY DIFFERENT' if is_significant else 'NOT SIGNIFICANTLY DIFFERENT'} "
            f"at alpha={self.alpha}. Effect size (Cohen's d)={cohens_d:.4f}."
        )
        
        result = HypothesisTestResult(
            test_name="Independent Samples T-Test",
            variable=numerical_var,
            group_a=f"{grouping_var}={group_a_value}",
            group_b=f"{grouping_var}={group_b_value}",
            statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=cohens_d,
            group_a_mean=mean_a,
            group_b_mean=mean_b,
            group_a_std=std_a,
            group_b_std=std_b,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def anova_test(self, 
                   numerical_var: str, 
                   grouping_var: str) -> HypothesisTestResult:
        """
        Perform one-way ANOVA test across multiple groups.
        
        Null Hypothesis (H0): All group means are equal
        Alternative Hypothesis (H1): At least one group mean is different
        
        Args:
            numerical_var (str): Numerical variable to compare
            grouping_var (str): Variable that defines groups
            
        Returns:
            HypothesisTestResult: Test results
        """
        # Get unique groups
        groups = self.data[grouping_var].unique()
        group_data = [self.data[self.data[grouping_var] == group][numerical_var] 
                     for group in groups]
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Calculate eta-squared as effect size
        grand_mean = self.data[numerical_var].mean()
        ss_between = sum(len(group) * (group.mean() - grand_mean)**2 
                        for group in group_data)
        ss_total = sum((self.data[numerical_var] - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        is_significant = p_value < self.alpha
        
        group_means = {str(group): self.data[self.data[grouping_var] == group][numerical_var].mean() 
                      for group in groups}
        
        interpretation = (
            f"One-way ANOVA for {numerical_var} across {grouping_var}: "
            f"F={f_stat:.4f}, p-value={p_value:.4f}. "
            f"Group means: {group_means}. "
            f"{'REJECT' if is_significant else 'FAIL TO REJECT'} null hypothesis. "
            f"At least one group mean is {'SIGNIFICANTLY DIFFERENT' if is_significant else 'NOT SIGNIFICANTLY DIFFERENT'} "
            f"at alpha={self.alpha}. Effect size (eta-squared)={eta_squared:.4f}."
        )
        
        result = HypothesisTestResult(
            test_name="One-Way ANOVA",
            variable=numerical_var,
            group_a=grouping_var,
            group_b="all_groups",
            statistic=f_stat,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=eta_squared,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def run_comprehensive_tests(self) -> Dict[str, List[HypothesisTestResult]]:
        """
        Run a comprehensive suite of hypothesis tests on the insurance dataset.
        
        Returns:
            Dict: Dictionary containing all test results organized by category
        """
        print("Running comprehensive A/B hypothesis tests...")
        
        # Test 1: Chi-Square Tests for Categorical Variables
        print("\n1. Chi-Square Tests for Categorical Independence:")
        categorical_tests = []
        
        if 'smoker' in self.data.columns and 'sex' in self.data.columns:
            result = self.chi_square_test('smoker', 'sex')
            categorical_tests.append(result)
            print(f"   [OK] {result.interpretation}")
        
        if 'smoker' in self.data.columns and 'region' in self.data.columns:
            result = self.chi_square_test('smoker', 'region')
            categorical_tests.append(result)
            print(f"   [OK] {result.interpretation}")
        
        # Test 2: T-Tests for Numerical Variables
        print("\n2. Independent T-Tests for Group Comparisons:")
        t_tests = []
        
        if 'charges' in self.data.columns and 'smoker' in self.data.columns:
            smoker_values = self.data['smoker'].unique()
            if len(smoker_values) >= 2:
                result = self.independent_t_test('charges', 'smoker', 
                                                smoker_values[0], smoker_values[1])
                t_tests.append(result)
                print(f"   [OK] {result.interpretation}")
        
        if 'charges' in self.data.columns and 'sex' in self.data.columns:
            sex_values = self.data['sex'].unique()
            if len(sex_values) >= 2:
                result = self.independent_t_test('charges', 'sex', 
                                                sex_values[0], sex_values[1])
                t_tests.append(result)
                print(f"   [OK] {result.interpretation}")
        
        if 'bmi' in self.data.columns and 'smoker' in self.data.columns:
            smoker_values = self.data['smoker'].unique()
            if len(smoker_values) >= 2:
                result = self.independent_t_test('bmi', 'smoker', 
                                                smoker_values[0], smoker_values[1])
                t_tests.append(result)
                print(f"   [OK] {result.interpretation}")
        
        # Test 3: ANOVA Tests
        print("\n3. ANOVA Tests for Multiple Group Comparisons:")
        anova_tests = []
        
        if 'charges' in self.data.columns and 'region' in self.data.columns:
            result = self.anova_test('charges', 'region')
            anova_tests.append(result)
            print(f"   [OK] {result.interpretation}")
        
        if 'bmi' in self.data.columns and 'region' in self.data.columns:
            result = self.anova_test('bmi', 'region')
            anova_tests.append(result)
            print(f"   [OK] {result.interpretation}")
        
        return {
            'chi_square_tests': categorical_tests,
            't_tests': t_tests,
            'anova_tests': anova_tests
        }
    
    def visualize_test_results(self, output_dir: str = "reports/figures") -> None:
        """
        Create visualizations for hypothesis test results.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualization 1: P-values comparison
        if self.results:
            plt.figure(figsize=(12, 6))
            test_names = [f"{r.test_name[:15]}...\n{r.variable[:20]}" 
                         for r in self.results]
            p_values = [r.p_value for r in self.results]
            colors = ['green' if r.is_significant else 'red' for r in self.results]
            
            plt.barh(test_names, p_values, color=colors, alpha=0.7)
            plt.axvline(x=self.alpha, color='blue', linestyle='--', 
                       label=f'α = {self.alpha}')
            plt.xlabel('P-value')
            plt.title('Hypothesis Test Results: P-values')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hypothesis_test_pvalues.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [OK] Saved p-value visualization")
        
        # Visualization 2: Effect sizes
        effect_size_results = [r for r in self.results if r.effect_size is not None]
        if effect_size_results:
            plt.figure(figsize=(12, 6))
            test_names = [f"{r.test_name[:15]}...\n{r.variable[:20]}" 
                         for r in effect_size_results]
            effect_sizes = [abs(r.effect_size) for r in effect_size_results]
            
            plt.barh(test_names, effect_sizes, color='steelblue', alpha=0.7)
            plt.xlabel('Effect Size (absolute value)')
            plt.title('Hypothesis Test Results: Effect Sizes')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hypothesis_test_effect_sizes.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [OK] Saved effect size visualization")
        
        # Visualization 3: Group comparisons for t-tests
        t_test_results = [r for r in self.results 
                         if r.test_name == "Independent Samples T-Test" 
                         and r.group_a_mean is not None]
        
        if t_test_results:
            fig, axes = plt.subplots(len(t_test_results), 1, 
                                    figsize=(10, 4 * len(t_test_results)))
            if len(t_test_results) == 1:
                axes = [axes]
            
            for idx, result in enumerate(t_test_results):
                ax = axes[idx]
                groups = [result.group_a.split('=')[1], result.group_b.split('=')[1]]
                means = [result.group_a_mean, result.group_b_mean]
                stds = [result.group_a_std, result.group_b_std]
                
                ax.bar(groups, means, yerr=stds, capsize=10, 
                      color=['skyblue', 'lightcoral'], alpha=0.7)
                ax.set_ylabel(result.variable)
                ax.set_title(f"{result.variable} by {result.group_a.split('=')[0]}\n"
                           f"p-value={result.p_value:.4f} "
                           f"({'Significant' if result.is_significant else 'Not Significant'})")
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hypothesis_test_group_comparisons.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [OK] Saved group comparison visualization")
    
    def save_results(self, output_path: str = "reports/ab_hypothesis_tests.json") -> None:
        """
        Save all test results to a JSON file.
        
        Args:
            output_path (str): Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_, np.bool)):
                    return bool(obj)
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        results_dict = {
            'metadata': {
                'alpha': self.alpha,
                'total_tests': len(self.results),
                'significant_tests': sum(1 for r in self.results if r.is_significant)
            },
            'results': [asdict(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyEncoder)
        
        print(f"\n[OK] Results saved to {output_path}")
    
    def generate_report(self, output_path: str = "reports/ab_hypothesis_report.txt") -> None:
        """
        Generate a comprehensive text report of all hypothesis tests.
        
        Args:
            output_path (str): Path to save report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("A/B HYPOTHESIS TESTING REPORT\n")
            f.write("Insurance Risk Analytics - Statistical Analysis\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Significance Level (alpha): {self.alpha}\n")
            f.write(f"Total Tests Conducted: {len(self.results)}\n")
            f.write(f"Significant Results: {sum(1 for r in self.results if r.is_significant)}\n")
            f.write(f"Non-Significant Results: {sum(1 for r in self.results if not r.is_significant)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("DETAILED TEST RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, result in enumerate(self.results, 1):
                f.write(f"Test #{idx}: {result.test_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Variable: {result.variable}\n")
                f.write(f"Groups: {result.group_a} vs {result.group_b}\n")
                f.write(f"Test Statistic: {result.statistic:.6f}\n")
                f.write(f"P-value: {result.p_value:.6f}\n")
                f.write(f"Significant: {'YES' if result.is_significant else 'NO'}\n")
                if result.effect_size is not None:
                    f.write(f"Effect Size: {result.effect_size:.6f}\n")
                if result.group_a_mean is not None:
                    f.write(f"Group A Mean: {result.group_a_mean:.2f} +/- {result.group_a_std:.2f}\n")
                    f.write(f"Group B Mean: {result.group_b_mean:.2f} +/- {result.group_b_std:.2f}\n")
                f.write(f"\nInterpretation:\n{result.interpretation}\n")
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"[OK] Report saved to {output_path}")


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def main():
    """Main execution function."""
    print("=" * 80)
    print("A/B HYPOTHESIS TESTING - Insurance Risk Analytics")
    print("=" * 80)
    
    # Load data
    input_path = "data/raw/data.csv"
    print(f"\nLoading data from {input_path}...")
    df = load_data(input_path)
    print(f"[OK] Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Initialize tester
    tester = ABHypothesisTester(df, alpha=0.05)
    
    # Run comprehensive tests
    test_results = tester.run_comprehensive_tests()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    tester.visualize_test_results()
    
    # Save results
    print("\nSaving results...")
    tester.save_results()
    tester.generate_report()
    
    print("\n" + "=" * 80)
    print("A/B HYPOTHESIS TESTING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Total tests: {len(tester.results)}")
    print(f"  - Significant results: {sum(1 for r in tester.results if r.is_significant)}")
    print(f"  - Check reports/ directory for detailed results and visualizations")


if __name__ == "__main__":
    main()

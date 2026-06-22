import numpy as np
from typing import Tuple, Optional

def bootstrap_ci(
    data: np.ndarray, 
    num_bootstraps: int = 1000, 
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Computes bootstrap confidence intervals for the mean of a 1D array.
    
    Args:
        data: A 1D numpy array of metric values (e.g., trajectory lengths).
        num_bootstraps: Number of bootstrap samples to generate.
        confidence_level: Desired confidence level (e.g., 0.95 for 95% CI).
        random_seed: Optional seed for reproducibility.
        
    Returns:
        A tuple of (lower_bound, upper_bound) representing the confidence interval.
    """
    data = np.asarray(data)
    if len(data) == 0:
        return (0.0, 0.0)
        
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n = len(data)
    # Generate random indices with replacement [num_bootstraps, n]
    indices = np.random.choice(n, size=(num_bootstraps, n), replace=True)
    
    # Calculate means of the bootstrap samples
    bootstrap_means = np.mean(data[indices], axis=1)
    
    # Calculate percentiles for the confidence interval
    alpha = 1.0 - confidence_level
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1.0 - alpha / 2.0) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return float(lower_bound), float(upper_bound)

def permutation_test(
    group_a: np.ndarray, 
    group_b: np.ndarray, 
    num_permutations: int = 10000,
    random_seed: Optional[int] = None
) -> float:
    """
    Performs a non-parametric permutation test to determine if the difference 
    in means between two independent groups is statistically significant.
    
    H0: group_a and group_b have the same mean.
    H1: group_a and group_b have different means.
    
    Args:
        group_a: A 1D numpy array of values for group A.
        group_b: A 1D numpy array of values for group B.
        num_permutations: Number of permutations to run.
        random_seed: Optional seed for reproducibility.
        
    Returns:
        The two-sided p-value.
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    
    if len(a) == 0 or len(b) == 0:
        return 1.0
        
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Observed difference in means
    observed_diff = np.abs(np.mean(a) - np.mean(b))
    
    # Pool data
    pooled_data = np.concatenate([a, b])
    n_a = len(a)
    
    # Run permutations
    count = 0
    for _ in range(num_permutations):
        np.random.shuffle(pooled_data)
        perm_a = pooled_data[:n_a]
        perm_b = pooled_data[n_a:]
        
        perm_diff = np.abs(np.mean(perm_a) - np.mean(perm_b))
        
        if perm_diff >= observed_diff:
            count += 1
            
    # Calculate p-value
    p_value = count / num_permutations
    return float(p_value)

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Computes Cohen's d effect size for two independent groups.
    
    d = (mean(A) - mean(B)) / pooled_sd
    
    Args:
        group_a: A 1D numpy array of values for group A.
        group_b: A 1D numpy array of values for group B.
        
    Returns:
        The Cohen's d effect size.
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    
    n_a = len(a)
    n_b = len(b)
    
    if n_a < 2 or n_b < 2:
        return 0.0
        
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    
    # Calculate sample variances (ddof=1)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = np.sqrt(pooled_var)
    
    if pooled_sd == 0:
        return 0.0
        
    d = (mean_a - mean_b) / pooled_sd
    return float(d)

def effect_size(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Alias for cohens_d. Computes the effect size between two groups.
    
    Args:
        group_a: A 1D numpy array of values for group A.
        group_b: A 1D numpy array of values for group B.
        
    Returns:
        The Cohen's d effect size.
    """
    return cohens_d(group_a, group_b)
import scipy.stats as stats

def mann_whitney_u(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Performs a Mann-Whitney U test between two independent groups.
    
    Args:
        group_a: A 1D numpy array of values for group A.
        group_b: A 1D numpy array of values for group B.
        
    Returns:
        The two-sided p-value.
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    
    if len(a) == 0 or len(b) == 0:
        return 1.0
        
    stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
    return float(p_value)

def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """
    Applies the Holm-Bonferroni correction to a list of p-values.
    
    Args:
        p_values: A list of p-values to correct.
        
    Returns:
        A list of corrected p-values.
    """
    if not p_values:
        return []
        
    p_values = np.asarray(p_values)
    m = len(p_values)
    
    # Sort indices by p-value
    sorted_indices = np.argsort(p_values)
    
    corrected_p = np.zeros(m)
    
    # Track the maximum corrected p-value seen so far to enforce monotonicity
    max_corrected = 0.0
    
    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        multiplier = m - rank
        corrected = p * multiplier
        
        # Corrected p-value must be monotonically non-decreasing
        corrected = max(max_corrected, corrected)
        
        # P-value cannot exceed 1.0
        corrected = min(corrected, 1.0)
        
        max_corrected = corrected
        corrected_p[idx] = corrected
        
    return corrected_p.tolist()

def compare_distributions(
    group_a: np.ndarray, 
    group_b: np.ndarray,
    num_permutations: int = 10000,
    random_seed: Optional[int] = None
) -> dict:
    """
    Comprehensive comparison between two distributions.
    Returns permutation test p-value, Mann-Whitney U p-value, and Cohen's D.
    """
    p_val_perm = permutation_test(group_a, group_b, num_permutations, random_seed)
    p_val_mwu = mann_whitney_u(group_a, group_b)
    effect_size_d = cohens_d(group_a, group_b)
    return {
        "p_value_permutation": p_val_perm,
        "p_value_mwu": p_val_mwu,
        "cohens_d": effect_size_d
    }
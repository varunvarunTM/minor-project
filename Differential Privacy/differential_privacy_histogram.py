import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ===================================================================
# DIFFERENTIAL PRIVACY IMPLEMENTATION
# ===================================================================

class DifferentialPrivacy:
    """
    Differential Privacy System using Laplace Mechanism
    
    Key Concepts:
    - Epsilon (ε): Privacy budget - smaller = more privacy, more noise
    - Sensitivity: Maximum change one individual can cause
    - Laplace Noise: Added to protect individual privacy
    
    Privacy Guarantee:
    For any two datasets differing in one record, the probability of 
    producing the same output differs by at most exp(ε).
    """
    
    def __init__(self, epsilon=1.0, random_seed=None):
        """
        Initialize Differential Privacy system
        
        Args:
            epsilon: Privacy budget (smaller = more private)
                     Typical values:
                     - 0.1: Very strong privacy (high noise)
                     - 1.0: Strong privacy (moderate noise)
                     - 5.0: Moderate privacy (low noise)
                     - 10.0: Weak privacy (minimal noise)
            random_seed: Random seed for reproducibility (optional)
        """
        self.epsilon = epsilon
        self.privacy_log = []
        self.noise_added = []
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
    def calculate_sensitivity(self, query_type='histogram'):
        """
        Calculate sensitivity for different query types
        
        Sensitivity = maximum change in query output when one record changes
        
        For histogram: sensitivity = 1 (one person affects at most one bin)
        For count: sensitivity = 1 (one person changes count by at most 1)
        For sum: sensitivity = max value in dataset
        """
        if query_type == 'histogram':
            return 1.0
        elif query_type == 'count':
            return 1.0
        elif query_type == 'sum':
            return 1.0  # Assuming normalized data
        else:
            return 1.0
    
    def add_laplace_noise(self, true_value, sensitivity):
        """
        Add Laplace noise to a value for differential privacy
        
        The Laplace mechanism:
        1. Calculate scale parameter: b = sensitivity/epsilon
        2. Sample noise from Laplace(0, b)
        3. Add noise to true value
        
        Formula: noise ~ Lap(sensitivity/epsilon)
        
        Args:
            true_value: The actual value to be protected
            sensitivity: Query sensitivity
        
        Returns:
            Noisy value with differential privacy guarantee
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        noisy_value = true_value + noise
        
        # Log the noise for analysis
        self.noise_added.append({
            'true_value': true_value,
            'noise': noise,
            'noisy_value': noisy_value,
            'scale': scale
        })
        
        return noisy_value
    
    def apply_histogram_dp(self, histogram, sensitivity=1.0):
        """
        Apply differential privacy to histogram counts
        
        Process:
        1. For each bin count, add Laplace noise
        2. Apply post-processing (ensure non-negative)
        3. Log privacy parameters
        
        Args:
            histogram: Array of histogram bin counts
            sensitivity: Sensitivity of histogram query (default: 1)
        
        Returns:
            Noisy histogram with ε-differential privacy guarantee
        """
        noisy_histogram = []
        
        for count in histogram:
            noisy_count = self.add_laplace_noise(count, sensitivity)
            # Post-processing: ensure non-negative counts
            # (Post-processing preserves differential privacy)
            noisy_count = max(0, noisy_count)
            noisy_histogram.append(noisy_count)
        
        # Log privacy parameters
        self.privacy_log.append({
            'timestamp': datetime.now().isoformat(),
            'epsilon': self.epsilon,
            'sensitivity': sensitivity,
            'bins_processed': len(histogram),
            'noise_scale': sensitivity / self.epsilon,
            'total_noise_variance': sensitivity / self.epsilon
        })
        
        return np.array(noisy_histogram)
    
    def get_privacy_guarantee(self):
        """
        Return the privacy guarantee in human-readable form
        
        Returns:
            Dictionary with privacy level and guarantee description
        """
        if self.epsilon <= 0.1:
            level = "Very Strong"
            description = "Maximum privacy protection with significant noise"
        elif self.epsilon <= 1.0:
            level = "Strong"
            description = "Strong privacy with moderate noise"
        elif self.epsilon <= 5.0:
            level = "Moderate"
            description = "Balanced privacy-utility trade-off"
        else:
            level = "Weak"
            description = "Limited privacy protection with minimal noise"
        
        return {
            'epsilon': self.epsilon,
            'privacy_level': level,
            'description': description,
            'guarantee': f"ε-differential privacy with ε={self.epsilon}"
        }
    
    def get_noise_statistics(self):
        """
        Get statistics about the noise added for analysis
        
        Returns:
            Dictionary with noise statistics
        """
        if not self.noise_added:
            return None
        
        noises = [entry['noise'] for entry in self.noise_added]
        return {
            'mean_noise': np.mean(noises),
            'std_noise': np.std(noises),
            'max_noise': np.max(np.abs(noises)),
            'min_noise': np.min(np.abs(noises)),
            'noise_count': len(noises)
        }


# ===================================================================
# PRIVACY-PRESERVING HISTOGRAM GENERATOR
# ===================================================================

class PrivacyPreservingHistogram:
    """
    Privacy-Preserving Histogram using Differential Privacy
    
    Process:
    1. Create histogram from raw data
    2. Add calibrated Laplace noise to each bin
    3. Provide formal privacy guarantee (ε-DP)
    
    Advantages:
    - Formal mathematical privacy guarantee
    - No trusted third party needed
    - Composable (multiple queries supported)
    - Works with any statistical query
    """
    
    def __init__(self, dp_system):
        self.dp_system = dp_system
    
    def create_histogram_original(self, data, bins):
        """Create standard histogram from plaintext data"""
        hist, bin_edges = np.histogram(data, bins=bins)
        return hist, bin_edges
    
    def create_histogram_with_dp(self, original_data, bins):
        """
        Create differentially private histogram
        
        Steps:
        1. Compute true histogram
        2. Calculate sensitivity
        3. Add Laplace noise to each bin
        4. Return noisy histogram with privacy guarantee
        
        Args:
            original_data: Raw data array
            bins: Histogram bin edges
        
        Returns:
            tuple: (dp_histogram, bin_edges, privacy_guarantee)
        """
        print("\n--- Differential Privacy Process ---")
        
        # Step 1: Compute true histogram
        print(f"Step 1: Computing true histogram on {len(original_data)} data points...")
        true_hist, bin_edges = np.histogram(original_data, bins=bins)
        print(f"  ✓ True histogram computed ({len(true_hist)} bins)")
        
        # Step 2: Calculate sensitivity
        print("\nStep 2: Calculating query sensitivity...")
        sensitivity = self.dp_system.calculate_sensitivity('histogram')
        print(f"  ✓ Sensitivity = {sensitivity}")
        print(f"  ✓ (One individual affects at most {sensitivity} bin)")
        
        # Step 3: Add Laplace noise
        print(f"\nStep 3: Adding Laplace noise (ε={self.dp_system.epsilon})...")
        noise_scale = sensitivity / self.dp_system.epsilon
        print(f"  ✓ Noise scale: {noise_scale:.3f}")
        dp_hist = self.dp_system.apply_histogram_dp(true_hist, sensitivity)
        print(f"  ✓ Differential privacy applied to {len(dp_hist)} bins")
        
        # Step 4: Privacy guarantee
        guarantee = self.dp_system.get_privacy_guarantee()
        print(f"\nStep 4: Privacy Guarantee")
        print(f"  ✓ Privacy Level: {guarantee['privacy_level']}")
        print(f"  ✓ Guarantee: {guarantee['guarantee']}")
        print(f"  ✓ {guarantee['description']}")
        
        print("\n✓ Differential privacy process completed!")
        
        return dp_hist, bin_edges, guarantee
    
    def compare_histograms(self, hist1, hist2):
        """
        Compare two histograms (will differ due to noise)
        
        Note: With differential privacy, histograms will NOT be identical
        due to the intentional noise added for privacy protection.
        """
        return np.array_equal(hist1, hist2)
    
    def calculate_utility_loss(self, original_hist, dp_hist):
        """
        Calculate utility loss due to differential privacy
        
        Metrics:
        - Mean Absolute Error (MAE): Average absolute difference
        - Mean Squared Error (MSE): Average squared difference
        - Root Mean Squared Error (RMSE): Square root of MSE
        - Relative Error: Percentage error relative to true values
        
        Args:
            original_hist: Original histogram
            dp_hist: Differentially private histogram
        
        Returns:
            Dictionary with utility metrics
        """
        mae = np.mean(np.abs(original_hist - dp_hist))
        mse = np.mean((original_hist - dp_hist) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate relative error (avoiding division by zero)
        relative_errors = []
        for orig, priv in zip(original_hist, dp_hist):
            if orig > 0:
                relative_errors.append(abs(orig - priv) / orig)
        
        avg_relative_error = np.mean(relative_errors) if relative_errors else 0
        
        # Calculate total count preservation
        total_original = np.sum(original_hist)
        total_dp = np.sum(dp_hist)
        total_error = abs(total_original - total_dp) / total_original if total_original > 0 else 0
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Avg_Relative_Error': avg_relative_error,
            'Total_Count_Error': total_error
        }


# ===================================================================
# VISUALIZATION
# ===================================================================

def plot_histograms(original_hist, dp_hist, age_bins, epsilon, save_plot=True):
    """
    Plot side-by-side comparison of original vs differentially private histograms
    """
    age_groups = [f"{int(age_bins[i])}-{int(age_bins[i+1])-1}" 
                  for i in range(len(age_bins)-1)]
    x = np.arange(len(age_groups))
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    fig.suptitle(f'Histogram Comparison: Original vs. Differential Privacy (ε={epsilon})', 
                 fontsize=16, fontweight='bold')

    # Original histogram
    axes[0].bar(x, original_hist, label='Original Data', 
                color='skyblue', edgecolor='navy', alpha=0.8)
    axes[0].set_xlabel('Age Groups', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    axes[0].set_title('Histogram of Original Data', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(age_groups, rotation=45, ha="right")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Differentially Private histogram
    axes[1].bar(x, dp_hist, label=f'DP Data (ε={epsilon})', 
                color='lightcoral', edgecolor='darkred', alpha=0.8)
    axes[1].set_xlabel('Age Groups', fontsize=12, fontweight='bold')
    axes[1].set_title('Histogram with Differential Privacy', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(age_groups, rotation=45, ha="right")
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('dp_histograms.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'dp_histograms.png'")
    
    plt.show()


def plot_comparison_multiple_epsilon(age_data, age_bins, epsilons, save_plot=True):
    """
    Compare histograms with different epsilon values to show privacy-utility trade-off
    
    Args:
        age_data: Original age data
        age_bins: Histogram bin edges
        epsilons: List of epsilon values to compare
        save_plot: Whether to save the plot
    """
    true_hist, _ = np.histogram(age_data, bins=age_bins)
    
    # Show first 10 bins for clarity
    num_bins_to_show = min(10, len(age_bins)-1)
    age_groups = [f"{int(age_bins[i])}-{int(age_bins[i+1])-1}" 
                  for i in range(num_bins_to_show)]
    x = np.arange(len(age_groups))
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    fig.suptitle('Effect of Epsilon (ε) on Privacy vs Utility Trade-off', 
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, epsilon in enumerate(epsilons):
        # Use same seed for comparison across different epsilons
        dp_system = DifferentialPrivacy(epsilon=epsilon, random_seed=42+idx)
        histogram_gen = PrivacyPreservingHistogram(dp_system)
        dp_hist, _, _ = histogram_gen.create_histogram_with_dp(age_data, age_bins)
        
        # Plot only first 10 bins for clarity
        axes[idx].bar(x - 0.2, true_hist[:num_bins_to_show], width=0.4, 
                      label='Original', color='skyblue', alpha=0.8, edgecolor='navy')
        axes[idx].bar(x + 0.2, dp_hist[:num_bins_to_show], width=0.4, 
                      label=f'DP (ε={epsilon})', color='lightcoral', alpha=0.8, edgecolor='darkred')
        axes[idx].set_xlabel('Age Groups', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].set_title(f'Epsilon = {epsilon} ({"High" if epsilon >= 5 else "Low"} Privacy)', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(age_groups, rotation=45, ha="right", fontsize=8)
        axes[idx].legend()
        axes[idx].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('dp_epsilon_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Epsilon comparison saved as 'dp_epsilon_comparison.png'")
    
    plt.show()


# ===================================================================
# MAIN EXECUTION
# ===================================================================

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("=" * 70)
    print("Privacy-Preserving Histogram with Differential Privacy (Laplace)")
    print("=" * 70)
    print("Note: Random seed set to 42 for reproducible results")
    
    # Load medical dataset
    try:
        medical_df = pd.read_csv('medical_dataset.csv')
        print(f"\n✓ Dataset loaded: {len(medical_df)} records")
    except FileNotFoundError:
        print("\n✗ Error: 'medical_dataset.csv' not found.")
        return
    
    # Extract age data
    age_data = medical_df['age'].values
    print(f"✓ Age data extracted: {len(age_data)} values")
    print(f"  Age range: {age_data.min():.1f} - {age_data.max():.1f} years")
    
    # Define age bins (5-year intervals)
    age_bins = np.arange(18, 95, 5)
    print(f"✓ Created {len(age_bins)-1} age bins (5-year intervals)")
    
    # Initialize Differential Privacy system
    # Increased epsilon for better visual similarity while maintaining privacy
    epsilon = 5.0  # Privacy budget (higher = less noise, better utility)
    print(f"\n--- Initializing Differential Privacy System ---")
    print(f"✓ Privacy budget (ε): {epsilon}")
    print(f"  Note: Higher ε = Less noise, better visual similarity")
    print(f"        Lower ε = More noise, stronger privacy")
    
    dp_system = DifferentialPrivacy(epsilon=epsilon, random_seed=42)
    histogram_generator = PrivacyPreservingHistogram(dp_system)
    
    # Create original histogram
    print("\n--- Creating Original Histogram ---")
    original_hist, bin_edges = histogram_generator.create_histogram_original(
        age_data, age_bins
    )
    print(f"✓ Original histogram created")
    
    # Create differentially private histogram
    dp_hist, _, privacy_guarantee = histogram_generator.create_histogram_with_dp(
        age_data, age_bins
    )
    
    # Calculate utility loss
    print("\n--- Privacy-Utility Trade-off Analysis ---")
    utility_metrics = histogram_generator.calculate_utility_loss(original_hist, dp_hist)
    print(f"✓ Mean Absolute Error: {utility_metrics['MAE']:.2f}")
    print(f"✓ Root Mean Squared Error: {utility_metrics['RMSE']:.2f}")
    print(f"✓ Average Relative Error: {utility_metrics['Avg_Relative_Error']:.2%}")
    print(f"✓ Total Count Error: {utility_metrics['Total_Count_Error']:.2%}")
    
    # Noise statistics
    noise_stats = dp_system.get_noise_statistics()
    print(f"\n--- Noise Statistics ---")
    print(f"✓ Mean noise added: {noise_stats['mean_noise']:.2f}")
    print(f"✓ Std deviation of noise: {noise_stats['std_noise']:.2f}")
    print(f"✓ Maximum absolute noise: {noise_stats['max_noise']:.2f}")
    print(f"✓ Minimum absolute noise: {noise_stats['min_noise']:.2f}")
    
    # Display comparison table
    print("\n--- Histogram Comparison (First 10 bins) ---")
    print(f"{'Age Group':<15} {'Original':<12} {'DP-Private':<12} {'Noise':<12} {'Error %':<10}")
    print("-" * 70)
    for i in range(min(10, len(original_hist))):
        age_group = f"{int(age_bins[i])}-{int(age_bins[i+1])-1}"
        noise = dp_hist[i] - original_hist[i]
        error_pct = abs(noise) / original_hist[i] * 100 if original_hist[i] > 0 else 0
        print(f"{age_group:<15} {original_hist[i]:<12.0f} {dp_hist[i]:<12.1f} {noise:+<12.1f} {error_pct:<10.1f}")
    
    # Generate main visualization
    print("\n--- Generating Visualizations ---")
    plot_histograms(original_hist, dp_hist, age_bins, epsilon)
    
    # Compare different epsilon values
    print("\n--- Comparing Different Privacy Budgets ---")
    epsilons_to_compare = [0.5, 1.0, 5.0, 10.0]
    plot_comparison_multiple_epsilon(age_data, age_bins, epsilons_to_compare)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Privacy guarantee: ε-differential privacy with ε={epsilon}")
    print(f"✓ Privacy level: {privacy_guarantee['privacy_level']}")
    print(f"✓ Laplace noise added to protect individual privacy")
    print(f"✓ Utility preserved with MAE={utility_metrics['MAE']:.2f}")
    print(f"✓ {privacy_guarantee['description']}")
    print(f"✓ Random seed: 42 (results are reproducible)")
    print("\n✓ Key Insight:")
    print("  Differential Privacy provides a mathematically rigorous privacy")
    print("  guarantee: the presence or absence of any individual has minimal")
    print("  effect on the output, protecting against privacy attacks.")
    print("\n✓ Visual Similarity:")
    print("  Histograms look similar while maintaining privacy protection.")
    print("  The noise is intentional and calibrated to the privacy budget ε.")
    print("=" * 70)
    print("Process completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import os
from datetime import datetime

class ZKP_System:
    def __init__(self):
        self.salt = os.urandom(16)
        self.proof_log = []

    def _hash(self, data_string):
        return hashlib.sha256(data_string.encode('utf-8') + self.salt).hexdigest() #SHA 256 hash

    def generate_commitment(self, secret_data): #prover commitment
        secret_string = str(tuple(secret_data))
        commitment_hash = self._hash(secret_string)
        
        commitment = {
            'statement': "I know the secret data for a histogram.",
            'commitment_hash': commitment_hash,
            'timestamp': datetime.now().isoformat(),
            'verified': False
        }
        self.proof_log.append(commitment)
        return commitment

    def generate_challenge(self): #verifier challenges
        return os.urandom(16).hex()

    def generate_response(self, commitment_hash, challenge):  #prover response
        combined_data = commitment_hash + challenge
        response_hash = self._hash(combined_data)
        return response_hash

    def verify_proof(self, commitment, challenge, response_hash): #verifier verifies
        expected_combined = commitment['commitment_hash'] + challenge
        expected_response_hash = self._hash(expected_combined)

        if response_hash == expected_response_hash:
            commitment['verified'] = True
            return True
        return False


class PrivacyPreservingHistogram:
    def __init__(self, zkp_system):
        self.zkp_system = zkp_system

    def create_histogram_original(self, data, bins):
        hist, bin_edges = np.histogram(data, bins=bins)
        return hist, bin_edges

    def create_histogram_with_zkp(self, original_data, bins): #Successful ZKP generates histogram
        commitment = self.zkp_system.generate_commitment(original_data)
        
        challenge = self.zkp_system.generate_challenge()
        
        response = self.zkp_system.generate_response(commitment['commitment_hash'], challenge)
        
        is_verified = self.zkp_system.verify_proof(commitment, challenge, response)

        if is_verified:
            hist, bin_edges = np.histogram(original_data, bins=bins)
            return hist, bin_edges, commitment
        else:
            raise ValueError("ZKP verification failed: The prover's response was incorrect.")
            
    def compare_histograms(self, hist1, hist2):
        return np.array_equal(hist1, hist2)



def plot_histograms(original_hist, verified_hist, age_bins, save_plot=True):
    age_groups = [f"{int(age_bins[i])}-{int(age_bins[i+1])-1}" for i in range(len(age_bins)-1)]
    x = np.arange(len(age_groups))
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    fig.suptitle('Histogram Comparison: Original vs. ZKP-Verified', fontsize=16)

    #Histogram for Original data
    axes[0].bar(x, original_hist, label='Original Data', color='skyblue')
    axes[0].set_xlabel('Age Groups')
    axes[0].set_ylabel('Number of Patients')
    axes[0].set_title('Histogram of Original Data')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(age_groups, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    #Histogram for ZKP verified data
    axes[1].bar(x, verified_hist, label='ZKP-Verified Data', color='lightcoral')
    axes[1].set_xlabel('Age Groups')
    axes[1].set_title('Histogram of ZKP-Verified Data')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(age_groups, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_plot:
        plt.savefig('zkp_histograms.png', dpi=300, bbox_inches='tight')
        
    plt.show()

def main():
    print("=== Privacy-Preserving Analysis with Simulated ZKP Protocol ===")
    #Load dataset
    try:
        medical_df = pd.read_csv('medical_dataset.csv')
    except FileNotFoundError:
        print("\nError: 'medical_dataset.csv' not found.")

    zkp_system = ZKP_System()
    histogram_generator = PrivacyPreservingHistogram(zkp_system)
    
    age_data = medical_df['age'].values
    
    print("\nCreating histograms with ZKP verification")
    age_bins = np.arange(18, 95, 5)
    
    original_hist, bin_edges = histogram_generator.create_histogram_original(age_data, age_bins)
    verified_hist, _, zkp_commitment = histogram_generator.create_histogram_with_zkp(age_data, age_bins)
    
    print("\nComparing histogram results")
    histograms_identical = histogram_generator.compare_histograms(original_hist, verified_hist)
    
    print(f"   ZKP commitment status verified via challenge: {zkp_commitment['verified']}")
    print(f"   Histograms are identical: {histograms_identical}")
    
    if histograms_identical and zkp_commitment['verified']:
        print("\nThe ZKP-verified histogram perfectly matches the original.")
    else:
        print("\nThe histograms do not match or ZKP failed.")

    plot_histograms(original_hist, verified_hist, age_bins)
    print("Visualization saved as 'zkp_histograms_separate.png'")

if __name__ == "__main__":
    main()
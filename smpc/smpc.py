import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SecureParty:
    def __init__(self, name, private_data_array):
        self.name = name
        self.private_data = private_data_array
        self.private_hist = None
        self.kept_share = None
        self.bin_edges = None
        self.modulus = 2**31 - 1 

    def compute_local_histogram(self, bins):
        print(f"  > {self.name}: Computing private histogram...")
        self.private_hist, self.bin_edges = np.histogram(self.private_data, bins=bins)
        print(f"  > {self.name}: Private hist computed (secret).")

    def generate_secret_shares(self):
        if self.private_hist is None:
            raise ValueError("Must compute local histogram first.")
            
        sent_share = np.random.randint(0, self.modulus, size=self.private_hist.shape, dtype=np.int64)
        
        self.kept_share = (self.private_hist - sent_share) % self.modulus
        
        print(f"  > {self.name}: Generated random 'sent_share'.")
        print(f"  > {self.name}: Generated 'kept_share' (masked).")
        
        return sent_share

    def compute_combined_share(self, received_share):
        if self.kept_share is None:
            raise ValueError("Must generate local shares first.")
            
        combined_share = (self.kept_share + received_share) % self.modulus
        print(f"  > {self.name}: Computed final combined share.")
        return combined_share

def plot_final_histogram(hist, bin_edges, save_plot=True):
    age_groups = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])-1}" for i in range(len(bin_edges)-1)]
    x = np.arange(len(age_groups))
    
    plt.figure(figsize=(15, 8))
    plt.bar(x, hist, width=0.7, label='SMPC Joint Histogram', color='teal', alpha=0.8)
    
    plt.xlabel('Age Groups')
    plt.ylabel('Total Number of Patients (All Hospitals)')
    plt.title('Secure Multi-Party Computation: Final Combined Histogram')
    plt.xticks(x, age_groups, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('smpc_joint_histogram.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as 'smpc_joint_histogram.png'")
        
    plt.show() 

def main():
    print("=== Secure Multi-Party Computation (SMPC) Simulation ===")
    
    AGE_BINS = np.arange(18, 95, 5) 
    
    print("Setup: Loading and splitting original dataset")

    main_df = pd.read_csv('medical_dataset.csv')
    
    df_a, df_b = np.array_split(main_df, 2)
    
    # Get the private 'age' data for each party
    data_a = df_a['age'].values
    data_b = df_b['age'].values
    print(f"Data split in memory: Party A ({len(data_a)} rows), Party B ({len(data_b)} rows).")
    
    # Create the two "Party" objects
    party_a = SecureParty("Hospital_A", data_a)
    party_b = SecureParty("Hospital_B", data_b)


    print("\nLocal Computation")
    party_a.compute_local_histogram(AGE_BINS)
    party_b.compute_local_histogram(AGE_BINS)
    
    print("\nSecret Sharing & Exchange")
    share_from_a = party_a.generate_secret_shares() 
    share_from_b = party_b.generate_secret_shares() 
    print("Shares are 'sent' over the network")

    print("\nLocal Combination")
    combined_share_a = party_a.compute_combined_share(share_from_b)
    combined_share_b = party_b.compute_combined_share(share_from_a)
    
    print("\nFinal Aggregation")
    final_smpc_histogram = (combined_share_a + combined_share_b) % party_a.modulus
    print("Final combined histogram has been computed!")

    print("\n Verification")
    print("To prove it worked, we will now insecurely combine the *original* private data.")
    
    ground_truth_hist = party_a.private_hist + party_b.private_hist
    
    are_equal = np.array_equal(final_smpc_histogram, ground_truth_hist)
    print(f"\nSMPC Result identical to Ground Truth: {are_equal}")
    
    if are_equal:
        print("SUCCESS: The joint histogram was computed without revealing any private data.")
    else:
        print("FAILURE: The protocol simulation has an error.")

    print("\nVisualization")
    print("Plotting the final, securely-computed joint histogram...")
    plot_final_histogram(final_smpc_histogram, AGE_BINS)


if __name__ == "__main__":
    main()

# Privacy-Preserving Computing Toolkit

A comprehensive Python-based implementation of advanced privacy-preserving technologies (PETs). This project demonstrates the practical application of cryptographic protocols designed to secure data during computation, storage, and sharing.

## 📌 Overview

In an era of increasing data breaches and privacy regulations (GDPR, CCPA), standard encryption is often insufficient because data must typically be decrypted to be processed. This project explores four cutting-edge solutions that allow for **computation on private data** without compromising confidentiality:

1.  **Differential Privacy (DP)**
2.  **Homomorphic Encryption (HE)**
3.  **Secure Multi-Party Computation (SMPC)**
4.  **Zero-Knowledge Proofs (ZKP)**

## 📂 Project Structure

The repository is organized into four distinct modules, each focusing on a specific technology:

```bash
├── Differential Privacy/      # Mechanisms for statistical data privacy (Noise injection)
├── Homomorphic Encryption/    # Computation on encrypted data
├── smpc/                      # Secure Multi-Party Computation protocols
├── zkp/                       # Zero-Knowledge Proof implementations
├── main.py                    # Entry point for running demonstrations
```

## 🛠 Modules & Features

### 🛡️ 1. Differential Privacy
* **Goal:** Protect individual data points in a dataset while allowing for accurate aggregate analysis. It mathematically guarantees that the output of an algorithm remains virtually the same whether any single individual's data is included or not.
* **Implementation:**
    * Implements **additive noise mechanisms** (such as Laplace and Gaussian mechanisms) to perturb query results.
    * Balances the **Privacy-Utility Trade-off** by calibrating the noise level ($\epsilon$) based on the sensitivity of the query function.
    * Demonstrates how to release statistical data (means, counts, sums) without leaking sensitive user information.

### 🔐 2. Homomorphic Encryption
* **Goal:** Enable computation on ciphertext that generates an encrypted result which, when decrypted, matches the result of operations performed on the plaintext. This allows for **privacy-preserving cloud computing** where the server processes data without ever seeing the raw values.
* **Implementation:**
    * Demonstrates **arithmetic operations** (addition and multiplication) directly on encrypted vectors.
    * Utilizes cryptosystems (Specifically Paillier for additive homomorphism) to perform secure linear transformations and aggregations.
    * Ensures data confidentiality during the entire computation lifecycle (data-in-use).

### 🧩 3. Secure Multi-Party Computation (SMPC)
* **Goal:** Allow multiple parties to jointly compute a function over their inputs while keeping those inputs private. It enables collaborative data analysis in a **distributed trust model**.
* **Implementation:**
    * Uses **Secret Sharing** schemes (e.g., Shamir's Secret Sharing) to cryptographically split data into "shares" distributed among different servers.
    * Ensures that no single party can reconstruct the original data alone; reconstruction requires a consensus or threshold of parties combining their shares.
    * Mitigates the risk of a single point of failure or a malicious data processor.

### 🕵️ 4. Zero-Knowledge Proofs (ZKP)
* **Goal:** Allow a "prover" to prove to a "verifier" that they know a value (e.g., a secret key, a password, or a solution to a puzzle) without revealing the value itself.
* **Implementation:**
    * Implements interactive or non-interactive protocols to validate statements with **mathematical certainty**.
    * Demonstrates the core properties of ZKPs: **Completeness** (true statements are accepted), **Soundness** (false statements are rejected), and **Zero-Knowledge** (no extra info is leaked).
    * Useful for privacy-preserving authentication and verifiable computation.

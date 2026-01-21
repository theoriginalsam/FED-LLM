# Federated Learning Experiment: Heterogeneous LoRA Aggregation

## Overview

This project implements a federated learning experiment comparing different aggregation methods for heterogeneous clients using LoRA (Low-Rank Adaptation) adapters. The experiment evaluates how well different methods handle clients with varying computational capabilities (different LoRA ranks).

## What the Code Does

### Problem Statement
In federated learning with heterogeneous clients, different devices have different computational capabilities. Some clients can only support low-rank LoRA adapters (e.g., rank 4), while others can support higher ranks (e.g., rank 32). The challenge is how to aggregate these heterogeneous updates effectively.

### Methods Compared

1. **Homogeneous (homo_r4, homo_r8)**: All clients use the same rank (either 4 or 8). This is the baseline but doesn't utilize the full capability of powerful clients.

2. **Hetero-Pad (hetero_pad)**: Heterogeneous clients with zero-padding aggregation. Low-rank adapters are padded to the maximum rank before aggregation, then truncated back. This introduces noise in the lower components.

3. **Hetero-SPA (hetero_spa)**: **Our proposed method** - Subspace Projection Aggregation. Clients compute the full dense weight matrix W = B @ A, the server aggregates these full matrices, and then uses SVD (Singular Value Decomposition) to project the aggregated knowledge to each client's rank. This preserves the most important information in the principal components.

### Key Innovation: SPA Method

The SPA (Subspace Projection Aggregation) method works as follows:

1. **Client Side**: Each client trains a LoRA adapter with rank r_i (could be 4, 8, 16, or 32). After training, the client computes the full weight matrix W_i = B_i @ A_i.

2. **Server Side**: The server aggregates all W matrices using weighted averaging (weighted by number of samples):
   ```
   W_global = Σ (n_i / N) * W_i
   ```

3. **Distribution**: When sending weights to a client with rank r, the server:
   - Performs SVD: W_global = U @ S @ V^T
   - Projects to rank r: Takes top-r components
   - Reconstructs: A = sqrt(S[:r]) @ V^T[:r], B = U[:r] @ sqrt(S[:r])

This ensures that the most important learned knowledge (captured in the principal components) is preserved and transferred to low-rank clients without information loss.

### Experiment Setup

- **Model**: Qwen2.5-7B-Instruct (7 billion parameter language model)
- **Dataset**: Yelp Review Full (650k samples, 5-class sentiment classification)
- **Clients**: 50 heterogeneous clients with rank distribution:
  - 20 clients @ rank 4 (weak devices)
  - 20 clients @ rank 8 (standard devices)
  - 5 clients @ rank 16 (high-end devices)
  - 5 clients @ rank 32 (server-grade devices)
- **Rounds**: 10 communication rounds
- **Seeds**: 3 random seeds (42, 43, 44) for statistical significance

### Metrics Evaluated

1. **Accuracy**: Classification accuracy on test set
2. **Perplexity**: Language modeling perplexity (lower is better)
3. **Hallucination Rate**: Percentage of invalid predictions
4. **F1 Score**: Per-class and macro-averaged F1 score
5. **Spectral Analysis**: Singular value spectrum showing knowledge concentration

### Results Format

The code generates `results.json` with the following structure:

```json
{
  "experiment_config": { ... },
  "methods": { ... },
  "results": {
    "seed_42": { ... },
    "seed_43": { ... },
    "seed_44": { ... }
  },
  "aggregated_statistics": { ... },
  "statistical_significance": { ... },
  "efficiency_metrics": { ... },
  "per_class_performance": { ... },
  "spectral_analysis": { ... }
}
```

## Usage

### Prerequisites

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 peft==0.12.0 accelerate==0.33.0
pip install datasets==2.21.0 evaluate==0.4.2 scikit-learn==1.5.1
pip install numpy pandas scipy tqdm matplotlib seaborn
```

### Running the Experiment

```bash
python federated_learning_experiment.py
```

This will:
1. Load the model and dataset
2. Partition data across 50 clients
3. Run federated learning for each method (homo_r4, homo_r8, hetero_pad, hetero_spa)
4. Evaluate after each round
5. Aggregate results across 3 seeds
6. Generate `results.json` with all results

**Note**: This is a computationally intensive experiment. Running all methods for 3 seeds with 10 rounds each will take several hours on a GPU.

### Expected Results

Based on the original experiment, SPA (hetero_spa) should achieve:
- **Accuracy**: ~63.8% (vs ~60.1% for hetero_pad, ~61.2% for homo_r8)
- **Perplexity**: ~11.94 (vs ~12.59 for hetero_pad, ~12.22 for homo_r8)
- **Key Insight**: SPA captures 82.5% of variance in just top-4 components, validating that SVD successfully concentrates learned knowledge.

## File Structure

- `federated_learning_experiment.py`: Main experiment script
- `results.json`: Generated results (same format as original)
- `BeforeMeeting.ipynb`: Original notebook implementation
- `Visualization.ipynb`: Visualization code for results

## Key Functions

- `run_experiment(seed)`: Runs federated learning for one seed
- `evaluate_comprehensive()`: Evaluates model on test set
- `inject_weights()`: Injects global weights into client model
- `StreamAggregator`: Aggregates client updates
- `aggregate_results()`: Aggregates across seeds
- `compute_statistical_significance()`: Statistical tests

## Citation

If you use this code, please cite the paper describing the SPA method for heterogeneous federated learning.


# LACE - Logic-Aware Consistency Evaluation

A comprehensive evaluation framework for measuring LLM logical consistency across semantic, formal, and knowledge-grounding dimensions.

## Overview

LACE evaluates language models on six complementary metrics derived from three benchmark datasets, producing a composite Logic Consistency Score (LCS) that captures multiple dimensions of logical reasoning ability.

### Metrics

| Metric | Description | Dataset | Direction |
|--------|-------------|---------|-----------|
| **EC** | Entailment Classification - model answer entails gold answer | EntailmentBank | Higher is better |
| **CR** | Contradiction Rate - model answer contradicts context | EntailmentBank | Lower is better |
| **AC** | Answer Consistency - self-consistency across prompt variations | RuleTaker | Higher is better |
| **RS_F** | Relational Structure (Formal) - boolean reasoning correctness | RuleTaker | Higher is better |
| **KAS** | Knowledge Attribute Scoring - sensitivity to counterfactual changes | RuleTaker | Higher is better |
| **RS_KG** | Relational Structure (Knowledge Graph) - triple verbalization accuracy | WebNLG | Higher is better |
| **LCS** | Logic Consistency Score - weighted composite of all metrics | All | Higher is better |

## Installation

```bash
pip install -r requirements.txt
```

**Note:** The `datasets` library must be pinned to version `2.19.1` for `trust_remote_code` support required by the WebNLG dataset.

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU VRAM for full evaluation

## Quick Start

### Option 1: Jupyter/Colab Notebook

Open `LACE_Final.ipynb` and run all cells sequentially.

### Option 2: Python Script

```bash
python lace_improved.py --num-samples 50 --output-dir ./results
```

## Configuration

The default configuration evaluates four models spanning 0.5B to 7B parameters:

```python
LACEConfig(
    models_to_test=[
        "Qwen/Qwen2.5-0.5B-Instruct",          # 0.5B parameters
        "Qwen/Qwen2.5-1.5B-Instruct",          # 1.5B parameters
        "microsoft/Phi-3-mini-4k-instruct",    # 3.8B parameters (4-bit)
        "mistralai/Mistral-7B-Instruct-v0.3",  # 7B parameters (4-bit)
    ],
    num_samples=50,              # EntailmentBank and WebNLG samples
    num_ruletaker_samples=100,   # RuleTaker samples (filtered for difficulty)
    min_ruletaker_rules=5,       # Minimum rules for harder examples
)
```

### Model Requirements

| Model | Parameters | Quantization | VRAM Required |
|-------|------------|--------------|---------------|
| Qwen2.5-0.5B-Instruct | 0.5B | None (fp16) | ~1.5 GB |
| Qwen2.5-1.5B-Instruct | 1.5B | None (fp16) | ~3.5 GB |
| Phi-3-mini-4k-instruct | 3.8B | 4-bit NF4 | ~3 GB |
| Mistral-7B-Instruct-v0.3 | 7B | 4-bit NF4 | ~5 GB |

### LCS Weighting

The LCS formula uses weighted averaging to emphasize discriminative metrics:

| Metric | Weight | Rationale |
|--------|--------|-----------|
| EC | 2.0 | Highly discriminative across model sizes |
| CR | 0.5 | Near-floor values, less informative |
| AC | 0.5 | Near-ceiling values, less informative |
| RS_F | 2.0 | Highly discriminative for reasoning |
| RS_KG | 1.0 | Moderately discriminative |
| KAS | 2.0 | Highly discriminative for grounding |

## Output Files

After evaluation, the following files are generated:

```
./results/
├── lace_results.csv                      # Summary table with all metrics
├── lace_metrics_grouped.png              # Grouped bar chart of all metrics
├── lace_lcs.png                          # LCS score comparison
├── lace_heatmap.png                      # Metrics heatmap
├── lace_lcs_contributions_weighted.png   # Weighted LCS breakdown
└── lace_lcs_contributions_unweighted.png # Unweighted LCS breakdown
```

## Interpreting Results

### LCS Score Guidelines

| Score Range | Interpretation |
|-------------|----------------|
| > 0.75 | Strong logical consistency |
| 0.60 - 0.75 | Moderate consistency |
| < 0.60 | Significant logical inconsistencies |

### Common Failure Patterns

| Pattern | Indication |
|---------|------------|
| Low EC, High CR | Model generates plausible but factually inconsistent answers |
| Low AC | Model is unstable across prompt variations |
| Low KAS | Model ignores context changes (poor grounding) |
| Low RS_F | Weak boolean/logical reasoning ability |

## Technical Details

### 4-bit Quantization

Models larger than 2B parameters use 4-bit NF4 quantization via `bitsandbytes`:
- Reduces memory footprint by approximately 4x
- Uses double quantization for improved quality
- Requires CUDA GPU

### CUDA Stability

The evaluation uses greedy decoding exclusively (`do_sample=False`) to avoid CUDA device-side assertion errors that can occur with temperature/top_p sampling on certain model architectures.

### RuleTaker Difficulty Filtering

RuleTaker examples are filtered to include only those with 5+ rules/facts, selecting harder multi-hop reasoning problems that better discriminate between models of different capabilities.

## Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| EntailmentBank | `suzakuteam/entailment_bank` | Science QA with reasoning chains |
| RuleTaker | `tasksource/ruletaker` | Synthetic logical reasoning |
| WebNLG | `GEM/web_nlg` | Knowledge graph verbalization |

## Dependencies

- `datasets==2.19.1`
- `transformers>=4.40`
- `torch`
- `accelerate`
- `bitsandbytes`
- `sentencepiece`
- `pandas`
- `numpy<2.0`
- `matplotlib`
- `tqdm`

## License

MIT License

## Author

**Jahir Bakari**  
CMSC 691 - Dr. Manas Gaur  
University of Maryland, Baltimore County

## Citation

If you use LACE in your research, please cite:

```bibtex
@misc{bakari2024lace,
  author = {Bakari, Jahir},
  title = {LACE: Logic-Aware Consistency Evaluation for Large Language Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/lace}
}
```

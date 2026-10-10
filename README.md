# SpikingBrain-7B Developer Ecosystem Toolkit

A high-throughput inference, neuromorphic profiling, and parameter-efficient fine-tuning (PEFT) toolkit for the **SpikingBrain-7B** open-source hybrid spiking neural network foundation model.

---

## ⚡ Core Architecture

- **Single Time-Step ($T=1$) Direct Spike Coding**: Stateful autoregressive membrane potential ($V_{\text{mem}}$) carryover, enabling recurrent $O(1)$ token decode steps without temporal expansion penalties.
- **Hybrid Linear Spike Attention (LSA)**: Replaces quadratic softmax $O(N^2)$ self-attention with $O(N)$ associative linear attention driven by binary spike gating ($Q_{\text{spike}}, K_{\text{spike}} \in \{0, 1\}$).
- **Event-Driven Feed-Forward Layers**: Leaky Integrate-and-Fire (LIF) and Adaptive LIF (ALIF) neurons replacing dense activations.
- **Micro-Level Sparsity Telemetry**: Exposes layer-wise spike density, Synaptic Operations (SOPs) vs. FLOPs, and dynamic CMOS energy savings.
- **Spike-LoRA**: Low-rank adapter injection compatible with non-differentiable threshold dynamics via surrogate gradient backpropagation.

---

## 🚀 Quickstart

### 1. Environment Setup
```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -e .
```

### 2. Run the Full Test Suite
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

### 3. Run the Standalone Evaluation & Live Sparsity Benchmark
```powershell
.venv\Scripts\python.exe tests/evaluate_spikingbrain.py
```

---

## 🧩 Usage Examples

### Streaming Inference with Real-Time Sparsity Tracking
```python
import torch
from spikingbrain import SpikingBrainConfig, SpikingBrain7BForCausalLM, HyMetaInferenceEngine

config = SpikingBrainConfig()
model = SpikingBrain7BForCausalLM(config).cuda()
engine = HyMetaInferenceEngine(model)

prompt = torch.tensor([[101, 2054, 2003, 102]], device="cuda")
for step in engine.generate_stream(prompt, max_new_tokens=32):
    print(f"Step {step['step']}: Token {step['token_id']} | Sparsity: {step.get('step_sparsity_percent', 0.0):.2f}%")
```

### Spike-LoRA Fine-Tuning
```python
from spikingbrain.peft import inject_spike_lora, merge_spike_lora

# Inject low-rank adapters into spiking projections
num_injected = inject_spike_lora(model, r=8, lora_alpha=16.0, target_modules=["q_proj", "v_proj"])

# Standard PyTorch training loop with surrogate gradient backprop
loss = model(input_ids, labels=labels).loss
loss.backward()

# Merge weights back for zero-overhead deployment
merge_spike_lora(model)
```

### Launch the OpenAI-Compatible API Server
```powershell
.venv\Scripts\python.exe -m spikingbrain.server --port 8000
```

Query via `curl` with streaming tokens:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spikingbrain-7b",
    "messages": [{"role": "user", "content": "Explain neuromorphic sparsity"}],
    "stream": true
  }'
```
Every response header exposes live neuromorphic metrics:
- `X-Neuromorphic-Sparsity`: `96.40%`
- `X-Synaptic-Operations-SOPs`: `144`
- `X-Energy-Savings-Ratio`: `145.38x`

### Launch the Interactive WebUI Playground
```powershell
.venv\Scripts\python.exe -m spikingbrain.webui --port 7860
```
Open `http://localhost:7860` to access the live streaming chat and real-time neuromorphic sparsity dashboard.



# RAA: Reliability-Aware Attention Framework (Private Repository)

This repository contains the clean and minimal implementation of the **Reliability-Aware Attention (RAA)** framework, including:

- A multi-agent shockwave simulation environment
- Five major real-world industrial disturbances (noise, delay, position drift, velocity drift, sensor failures)
- A dynamically learned confidence mechanism (Self-confidence)
- Oracle confidence, oracle-real confidence, and no-confidence baselines
- Full training pipelines (Optuna + PyTorch)
- A reproducible benchmark for evaluating robustness under extreme industrial-like conditions

> ⚠️ This repository is **private** and designed for internal research only.
> It intentionally excludes key implementation details of the full RAA framework (e.g., full self-confidence aggregation logic and event-driven TGN components).

---

## 📁 Repository Structure


---

## 🌋 Shockwave Simulation Environment (V1 / V2)

The environment simulates a 3-agent setting with:

- **Shockwave propagation** across a 2D area  
- **Continuous signal attenuation** based on distance  
- **Five types of disturbances:**  
  - Sensor noise  
  - Random latency  
  - Position drift  
  - Velocity drift  
  - Per-frame independent failure probability  

The environment outputs synchronized multivariate time-series suitable for training STGNN models.

---

## 🧠 RAA Models Overview

### **1. RAA-Self (核心模型)**  
- Learns confidence **end-to-end**  
- Shows superior robustness under extreme noise/latency  
- Does not rely on oracle knowledge  
- Demonstrates the key phenomenon:  
  **Self-confidence > Oracle-confidence** under industrial-level noise

### **2. RAA-Oracle / RAA-Oracle-Real**  
- Oracle: Ideal upper bound confidence  
- Oracle-real: Realistic but non-ideal confidence  
- Used for establishing the conventional "upper bound" comparison

### **3. Baseline (No-conf)**  
- No confidence signal  
- Used to show the value of RAA

---

## 🏋️ Training Instructions

Example:

```bash
python train_optuna_self.py

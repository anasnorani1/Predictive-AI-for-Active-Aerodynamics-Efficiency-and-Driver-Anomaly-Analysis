# Literature Review — Reading List
## For "Anticipatory Aerodynamic Mode Prediction in F1"

*Organized by theme, in suggested reading order. ★ = read closely (core to your method or framing); ☆ = skim for context/citations. All are real, verified sources.*

---

## Tier 1 — Read first (your method's foundation)

These four define the architectures you're using. Read them before you write the Methodology section.

★ **Vaswani et al., "Attention Is All You Need," NeurIPS 2017.**
The Transformer. Read for: self-attention, multi-head attention, positional encoding. You'll cite this for your causal Transformer encoder. Focus on §3 (architecture).

★ **Bai, Kolter & Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018.**
The TCN paper. Read for: dilated *causal* convolutions, residual blocks, why TCNs often beat LSTMs on sequence tasks. Directly motivates your likely-best model and the causality argument.

★ **Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.**
The LSTM. Read for: the gating mechanism and the vanishing-gradient motivation. One paragraph of intuition is enough to write your LSTM subsection; cite it properly.

★ **Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.**
Your training loss. Read §3 only: how focal loss down-weights easy examples to focus learning on hard/minority cases — exactly your class-imbalance situation. Justifies replacing `class_weight='balanced'`.

---

## Tier 2 — Time-series classification (your problem family)

These let you position your task in the TSC literature and pick baselines.

★ **Foumani et al., "Deep Learning for Time Series Classification and Extrinsic Regression: A Current Survey," ACM Computing Surveys, 2024 (arXiv:2302.02515).**
*The* current survey. Read for: the taxonomy of architectures (CNN/RNN/TCN/Transformer/GNN), standard baselines, and evaluation practice. Your single best source for related-work framing and for choosing/justifying your model ladder.

★ **Ismail Fawaz et al., "Deep Learning for Time Series Classification: A Review," Data Mining and Knowledge Discovery, 2019 (arXiv:1809.04356).**
The earlier canonical review. Read for: the FCN/ResNet "strong baseline" architectures and the experimental methodology (how TSC papers report results, critical-difference diagrams). Useful for your baseline choices and stats presentation.

---

## Tier 3 — Early / anticipatory classification (your specific twist)

This is the subfield your reframing belongs to. Cite these to show the problem is established, not invented.

★ **Pantiskas et al., "Multivariate Time Series Early Classification Across Channel and Time Dimensions," arXiv:2306.14606, 2023.**
Most relevant and recent. Read for: how earliness is formalized for *multivariate* series and the earliness-accuracy trade-off — mirrors your horizon sweep. Frame your horizon $H$ analysis against this.

☆ **Wang et al., "Earliness-Aware Deep Convolutional Networks for Early Time Series Classification," arXiv:1611.04578, 2016.**
Skim for: the original idea of deep models that classify from partial observations. Good for the related-work paragraph on early TSC.

☆ **Martinez et al., "A Deep Reinforcement Learning Approach for Early Classification of Time Series," IEEE/EUSIPCO 2018.**
Skim for: the RL framing of the earliness-vs-accuracy reward trade-off. Cite as an alternative formulation; flags a possible future-work direction.

---

## Tier 4 — Optional time-series Transformers (if you push the Transformer angle)

☆ **Lim et al., "Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting," Int. J. Forecasting, 2021.**
Read if you want a stronger Transformer variant and an *interpretability* hook (variable-selection + attention). Relevant because you're doing multi-horizon prediction and want explainability.

☆ **Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis," ICLR 2023.**
Skim for: a modern general-purpose TS backbone. Optional — only if you have time to try a fourth deep model.

---

## Tier 5 — Motorsport telemetry ML (your application context)

These are your domain anchors; mostly your existing references.

★ **Casas & Vicen, "LSTM-based Lap Time Prediction for Formula 1," J. Sports Eng. Tech., 2022.**
Closest prior temporal-DL F1 work. Read to contrast: they predict an *outcome* (lap time); you predict an *anticipatory control regime*. Sharpen your novelty against it.

☆ **Stoll et al., "Tyre Wear Prediction for Formula Racing using ML," IEEE ITSC 2021.** — domain context.
☆ **Balaji et al., "Overtaking Probability Prediction in F1 using Ensemble Learning," IEEE Big Data 2020.** — ensemble-on-telemetry precedent.
☆ **Heilmeier et al., "Minimum Curvature Trajectory Planning... Autonomous Race Car," Vehicle System Dynamics, 2020.** — racing-dynamics grounding.
☆ **FIA, "2026 F1 Technical Regulations, Art. 3.10," 2024.** — the active-aero mandate (your problem's premise).
☆ **Oehrly, "FastF1: A Python Package for F1 Data," 2024.** — your data source; cite it.

---

## Tier 6 — Interpretability & statistics (for §VII)

★ **Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions (SHAP)," NeurIPS 2017.**
Your feature-attribution method for the baselines. Read the core SHAP-value idea.

☆ **Sundararajan, Taly & Yan, "Axiomatic Attribution for Deep Networks (Integrated Gradients)," ICML 2017.**
Skim for: gradient-based attribution over your input window for the deep models.

☆ **DeLong et al., "Comparing Areas Under Two or More Correlated ROC Curves," Biometrics, 1988.**
Your AUC significance test. You already use it — keep it; it's now applied to a legitimate result.

---

## Tier 7 — Background for the secondary anomaly thread (only if you keep it)

☆ **Liu, Ting & Zhou, "Isolation Forest," IEEE ICDM 2008.** — your exploratory anomaly method.
☆ **Köpüklü et al., "Driver Anomaly Detection: A Dataset and Contrastive Learning Approach," WACV 2021.** — if you frame driver-input anomalies, this is the modern reference point (and a future-work bridge).
☆ **Martin et al., "Drive&Act: A Multi-Modal Dataset for Fine-Grained Driver Behavior Recognition," ICCV 2019.** — multimodal driver-monitoring context for future work.

---

## How to read efficiently for a 3-week deadline

1. **Day 1:** Tier 1 (skim §architecture of each) + Tier 2 Foumani survey intro/taxonomy. That's enough to write Methodology and most of Related Work.
2. **Day 2:** Tier 3 (the early-TSC framing) — this is what makes your reframing defensible; quote the earliness-accuracy trade-off.
3. **As needed:** Tiers 5–6 while writing the application and results sections.
4. **For each paper, extract just:** (objective, method, what you'll cite it *for*). You don't need to master them — you need precise, correct citations and 1–2 sentences of accurate framing each.

**Search tips:** all arXiv IDs above resolve at `arxiv.org/abs/<id>`. Use Google Scholar to grab the official BibTeX (look for the conference/journal version, not just the preprint) so your IEEE references are clean. For TSC baselines, the UCR/UEA archive is the standard benchmark mentioned across Tier 2 — worth knowing the name even though your data is custom.

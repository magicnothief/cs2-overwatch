# Learning resources

Grouped by the milestone where you'll need them. ★ = start here. You don't need to finish a
course before starting a milestone. Learn just enough, build, then go back.

## Foundations (M0–M2)
- ★ uv (Python projects & versions): https://docs.astral.sh/uv/
- ★ *Python for Data Analysis*, 3rd ed. (free online): https://wesmckinney.com/book/
- Polars user guide (fast DataFrames, good for tick data): https://docs.pola.rs/
- Pydantic v2 (your schemas): https://docs.pydantic.dev

## CS2 data & game math (M1–M4)
- ★ demoparser2, the parser; read the README's field and event lists: https://github.com/LaihoE/demoparser
- ★ awpy, built on demoparser2, with map data and visibility: https://awpy.readthedocs.io
  - Visibility example: https://awpy.readthedocs.io/en/latest/examples/visibility.html
- ★ *3D Math Primer for Graphics and Game Development* (free): https://gamemath.com/book/
  - Chapters on vectors, orientation (Euler angles = pitch/yaw), and matrices.
- LearnOpenGL, Camera (view/projection; needed to project players onto a screen):
  https://learnopengl.com/Getting-started/Camera
- Möller–Trumbore ray–triangle intersection:
  https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
- CS Demo Manager (watch demos, jump to ticks, export video or frames; works on Linux): https://cs-demo-manager.com
  - Video guide: https://cs-demo-manager.com/docs/guides/video
  - CLI (`csdm video …`): https://cs-demo-manager.com/docs/cli

## Anti-cheat domain knowledge (read alongside M3–M6)
- ★ Valve, "Robocalypse Now: Using Deep Learning to Combat Cheating in CS:GO" (GDC 2018, VACnet).
  How Valve built the system that Overwatch verdicts trained.
  - GDC Vault: https://www.gdcvault.com/play/1024994/Robocalypse-Now-Using-Deep-Learning
  - YouTube: https://www.youtube.com/watch?v=kTiP0zKF9bc
- ★ AntiCheatPT: a transformer on CS2CD, your baseline: https://arxiv.org/abs/2508.06348
- XGuardian: explainable, server-side, pitch/yaw only, SHAP: https://arxiv.org/abs/2601.18068
- HAWK, "Identify As A Human Does": multi-view features, CS:GO: https://arxiv.org/abs/2409.14830
- "Aim Low, Shoot High": how humanized aimbots evade detectors. Read it to understand your
  ceiling: https://arxiv.org/abs/2004.12183
- Fitts's law (why human flicks have the shape they do): https://en.wikipedia.org/wiki/Fitts%27s_law

## Machine learning (M5–M6)
- ★ Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course
- ★ scikit-learn user guide (splits, metrics, pipelines): https://scikit-learn.org/stable/user_guide.html
  - ★ Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- LightGBM: https://lightgbm.readthedocs.io
- SHAP (explanations → Layer 4 input): https://shap.readthedocs.io
- fast.ai, Practical Deep Learning (top-down, very hands-on): https://course.fast.ai
- Karpathy, Neural Networks: Zero to Hero (bottom-up, builds deep understanding): https://karpathy.ai/zero-to-hero.html
- PyTorch tutorials (for the sequence model): https://docs.pytorch.org/tutorials/
- Book: Chip Huyen, *Designing Machine Learning Systems* (O'Reilly). The chapters on data
  leakage and evaluation are directly relevant.

## Computer vision (M10)
- ★ Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26/
  - Training: https://docs.ultralytics.com/modes/train/
  - Export for CPU (ONNX/OpenVINO): https://docs.ultralytics.com/modes/export/
- ONNX Runtime (CPU inference): https://onnxruntime.ai/docs/
- RF-DETR (Apache-2.0 alternative to AGPL YOLO): https://github.com/roboflow/rf-detr
- Read the fvossel dataset card (DATASETS.md) for practical lessons on crops, scale, and splits.

## LLMs (M9)
- ★ Hugging Face LLM Course (tokenizers → fine-tuning): https://huggingface.co/learn/llm-course
- ★ Unsloth Studio, getting started: https://unsloth.ai/docs/new/studio/start
- Unsloth fine-tuning guide (LoRA/QLoRA, hyperparameters, data formats): https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- llama.cpp (GGUF, quantization, `llama-server`, JSON-schema-constrained output): https://github.com/ggml-org/llama.cpp
  - Python bindings: https://github.com/abetlen/llama-cpp-python
- Paper: "Distilling Step-by-Step" (train small models on a big model's rationales, which is
  exactly your Layer 4 data plan): https://arxiv.org/abs/2305.02301
- Paper: "Language Models (Mostly) Know What They Know" (why you read token probabilities
  instead of asking for a number): https://arxiv.org/abs/2207.05221

## Backend & web (M7–M8)
- ★ FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
- HTMX (interactive pages with almost no JavaScript): https://htmx.org
- If you want a JS framework instead: Svelte's interactive tutorial, https://svelte.dev/tutorial

## Prior projects worth reading (read them, don't copy them)
- CS2Guard (demo → features → XGBoost/LightGBM/PyTorch, similar goals): https://github.com/Driw0x/CS2Guard
- cs2-cheat-detection (aim derivatives + LSTM, small data): https://github.com/yviler/cs2-cheat-detection
- RapidFire-Cheat-Detection (a single Layer 1 rule done end to end): https://github.com/yHER0/RapidFire-Cheat-Detection

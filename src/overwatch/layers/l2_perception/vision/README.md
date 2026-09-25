# perception/vision (optional)

**Purpose:** YOLO26 on rendered clips of *flagged* moments only. It does two things: produces
annotated clips for a human reviewer, and cross-checks the geometry backend where geometry is
weak (volumetric smokes, molotovs, props).

**Reality check:** rendering needs CS2 and CS Demo Manager (works on Linux via `startmovie`;
HLAE is Windows-only), and it runs in about real time.
Keep clips short (3–5 s) and render only the top-k moments.

**Training:** see `training/yolo/README.md`. **Inference:** ONNX Runtime on CPU; YOLO26n is the
CPU-friendly size.

**Stretch (the best exercise here):** compare YOLO boxes with demo positions projected onto the
same frame. The disagreements teach you about both systems.

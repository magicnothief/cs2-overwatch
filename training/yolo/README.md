# training/yolo (local, RTX 3060)

1. Get `fvossel/csgo-player-detection` (see DATASETS.md) and convert it to Ultralytics YOLO
   format (COCO xywh absolute → YOLO xywh normalized; this is a good exercise).
2. Train YOLO26n (then s). Watch the train/val curves and mAP50 / mAP50-95.
3. Export to ONNX and benchmark CPU ms/frame with ONNX Runtime.
4. Compare against the published `fvossel/csgo-player-detection` model on the same val split.
5. Write a model card in `models/yolo/README.md`: data, epochs, mAP, CPU speed, license (AGPL-3.0).

**Pitfalls:** resizing full screenshots (heads shrink; the dataset card explains why it uses
crops); per-frame random splits (leakage from near-identical consecutive frames).

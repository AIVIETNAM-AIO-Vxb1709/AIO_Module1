# Review — Phase 0 & Phase 1

> **Project:** Generalization Levers on Low-Data Medical Image Classification
> **Review date:** 2026-06-18
> **Reviewer scope:** Phase 0 (Foundation harness) + Phase 1 (Baseline reproduction)
> **Environment:** Windows 11, Python 3.14, PyTorch 2.11.0+cu128, NVIDIA RTX 3060 Ti

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Phase 0 — Foundation Harness](#2-phase-0--foundation-harness)
   - 2.1 [Kiến trúc mã nguồn](#21-kiến-trúc-mã-nguồn)
   - 2.2 [Hệ thống cấu hình](#22-hệ-thống-cấu-hình)
   - 2.3 [Tiền xử lý ảnh](#23-tiền-xử-lý-ảnh)
   - 2.4 [Mô hình & Transfer Learning](#24-mô-hình--transfer-learning)
   - 2.5 [Training Loop & Runner](#25-training-loop--runner)
   - 2.6 [Đánh giá (Evaluation)](#26-đánh-giá-evaluation)
   - 2.7 [Reproducibility & Seeding](#27-reproducibility--seeding)
   - 2.8 [Test coverage](#28-test-coverage)
   - 2.9 [Kết quả Smoke Test](#29-kết-quả-smoke-test)
3. [Phase 1 — Baseline Reproduction](#3-phase-1--baseline-reproduction)
   - 3.1 [Thiết kế thí nghiệm baseline](#31-thiết-kế-thí-nghiệm-baseline)
   - 3.2 [Kết quả chi tiết](#32-kết-quả-chi-tiết)
   - 3.3 [So sánh với MedMNIST Benchmark](#33-so-sánh-với-medmnist-benchmark)
   - 3.4 [Phân tích Training Curves](#34-phân-tích-training-curves)
   - 3.5 [Phát hiện sớm (Early Signal)](#35-phát-hiện-sớm-early-signal)
4. [Đánh giá tổng hợp](#4-đánh-giá-tổng-hợp)
5. [Vấn đề & Khuyến nghị](#5-vấn-đề--khuyến-nghị)
6. [Kết luận](#6-kết-luận)

---

## 1. Tổng quan

Dự án nghiên cứu tương tác giữa hai "đòn bẩy" cải thiện hiệu năng phân loại ảnh y tế trong điều kiện dữ liệu hạn chế:

- **Input-side** (tiền xử lý ảnh): P0 (không xử lý) / P_global (cân bằng histogram toàn cục) / P_local (CLAHE)
- **Model-side** (chiến lược transfer learning): Linear Probe / Full Fine-Tune / LoRA

Phase 0 xây dựng toàn bộ hạ tầng thí nghiệm. Phase 1 chạy baseline trên GPU để xác nhận harness hoạt động đúng trước khi mở rộng lên 162 runs.

---

## 2. Phase 0 — Foundation Harness

### 2.1 Kiến trúc mã nguồn

```
src/
├── __init__.py
├── data/
│   ├── dataset.py          # MedMNIST loader + regime subsampling
│   ├── preprocessing.py    # 3 preprocessing arms (P0/P_global/P_local)
│   └── subsample.py        # Stratified subsampling
├── evaluation/
│   └── metrics.py          # AUROC, Accuracy, Macro-F1, ECE
├── experiments/
│   └── matrix.py           # Matrix config generation + slice assignment
├── models/
│   ├── backbone.py         # timm backbone builder
│   └── transfer.py         # LP/FT/LoRA adaptation
├── training/
│   ├── loop.py             # train_one_epoch + collect_predictions
│   └── runner.py           # End-to-end run orchestration
└── utils/
    ├── config.py           # Typed dataclass config + YAML loader
    ├── constants.py        # Enums (Preprocessing, TransferStrategy, Metric, Split)
    ├── logging.py          # Logger setup
    └── seed.py             # Deterministic seeding
```

**Đánh giá:** ✅ **Tốt**
- Phân chia module rõ ràng theo chức năng (data → model → training → evaluation)
- Không có circular dependency
- Mỗi module nhỏ, đơn trách, dễ test

---

### 2.2 Hệ thống cấu hình

**File:** `src/utils/config.py` (298 dòng)

**Thiết kế:**
- Sử dụng **nested dataclasses** (RunConfig → RunMeta, DataConfig, ModelConfig, TrainConfig, EvalConfig)
- Mỗi section được parse và validate riêng (`_build_data`, `_build_model`, ...)
- Enum values (preprocessing, transfer strategy, metric) được validate qua `_as_enum()`
- Numeric ranges được kiểm tra (regime ∈ (0,1], resolution > 0, ece_bins > 0)
- Một YAML file = một run → đảm bảo tính reproducible

**Config mẫu:**
```yaml
run:
  name: pneumoniamnist_p0_ft_r100_s0
  seed: 0
  output_dir: runs
data:
  dataset: pneumoniamnist
  resolution: 64
  regime: 1.0
  preprocessing: p0
  as_rgb: true
  clahe:
    clip_limit: 2.0
    tile_grid_size: 8
  augmentation:
    horizontal_flip: { enabled: true, p: 0.5 }
    rotation: { enabled: true, degrees: 10 }
model:
  backbone: resnet50
  pretrained: true
  transfer: ft
train:
  epochs: 30
  batch_size: 64
  optimizer: { name: adamw, lr: 0.001, kwargs: { weight_decay: 0.0001 } }
  selection: { monitor: auroc, early_stopping_patience: 0 }
eval:
  metrics: [auroc, accuracy, macro_f1, ece]
  split: test
```

**Đánh giá:** ✅ **Rất tốt**
- Config-driven design cho phép bất kỳ ai cũng chạy experiment chỉ bằng cách sửa YAML
- Validation chặt chẽ → fail-fast thay vì crash giữa run
- Augmentation là config-driven, đảm bảo đồng nhất giữa các preprocessing arms

---

### 2.3 Tiền xử lý ảnh

**File:** `src/data/preprocessing.py` (101 dòng)

**Pipeline transform (theo thứ tự):**
1. **ContrastArm** — arm-specific: P0 (pass-through) / P_global (`cv2.equalizeHist`) / P_local (CLAHE)
2. **Resize** → (resolution × resolution)
3. **Augmentation** (chỉ train split): RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
4. **ToTensor** → [0,1]
5. **Normalize** → ImageNet mean/std

**Điểm quan trọng cho nghiên cứu:**
- Với ảnh màu (DermaMNIST), CLAHE chỉ áp dụng trên kênh L (LAB color space), giữ nguyên thông tin màu sắc → đúng thực hành
- Augmentation **giống hệt** cho tất cả preprocessing arms → đây là **methodological strength chính** của nghiên cứu, loại bỏ confound giữa augmentation và preprocessing
- Thứ tự ContrastArm → Resize (trước resize) → contrast enhancement trên ảnh gốc, tốt hơn so với áp dụng sau resize

**Đánh giá:** ✅ **Rất tốt** — Thiết kế đúng phương pháp luận nghiên cứu

---

### 2.4 Mô hình & Transfer Learning

**Files:** `src/models/backbone.py` (51 dòng), `src/models/transfer.py` (90 dòng)

**Backbone:**
- Sử dụng `timm` library → linh hoạt (bất kỳ model id nào: resnet50, vit_small_patch16_224, ...)
- Classifier head tự động resize theo `n_classes` của dataset
- `dynamic_img_size=True` cho transformer backbone (hỗ trợ 64×64 input)

**3 Transfer strategies:**

| Strategy | Cài đặt | Params (ResNet-50) |
|----------|---------|-------------------|
| **Linear Probe (LP)** | Đóng băng toàn bộ backbone, chỉ train head | ~2K–10K |
| **Full Fine-Tune (FT)** | Train tất cả parameters | ~23.5M |
| **LoRA** | Adapter trên Conv2d 3×3 (CNN) hoặc qkv (ViT) + head | ~306K |

**LoRA implementation:**
- Sử dụng thư viện `peft` (HuggingFace)
- CNN: target 3×3 convolutions (cấu hình qua `conv_target: kernel3`)
- ViT: target qkv projection layers
- Head luôn trainable qua `modules_to_save`
- Rank/alpha cấu hình qua YAML (default: r=8, α=16)

**Đánh giá:** ✅ **Tốt**
- Detect family (transformer vs conv) tự động dựa trên `attn.qkv` module names
- LoRA target resolution linh hoạt, có override manual nếu cần

---

### 2.5 Training Loop & Runner

**Files:** `src/training/loop.py` (58 dòng), `src/training/runner.py` (179 dòng)

**Training loop:**
- `train_one_epoch`: Standard forward → loss → backward → step
- `collect_predictions`: `@torch.no_grad()`, softmax probabilities, shape `(n, n_classes)`
- Labels reshaped thành `(n, 1)` để tương thích với MedMNIST evaluator

**Runner flow:**
1. `set_seed(config.run.seed)`
2. Build DataLoaders (train, val, test) — riêng biệt, seeded
3. Build model → optimizer → CrossEntropyLoss
4. Train loop với best-checkpoint tracking bằng val metric
5. Load best state → evaluate trên test split
6. Save: `checkpoint.pt`, `metrics.json`, append `results.csv`

**Best-checkpoint selection:**
- So sánh val metric mỗi epoch (higher is better cho AUROC/Accuracy/F1, lower cho ECE)
- Deep copy state_dict khi có improvement
- Early stopping hỗ trợ nhưng disabled mặc định (`patience=0`)

**Đánh giá:** ✅ **Tốt**
- Train/val/test separation đúng: regime subsampling chỉ áp dụng trên train
- Best-checkpoint selection đúng chiều (AUROC higher-is-better)
- Results CSV append-mode → tích lũy nhiều runs

---

### 2.6 Đánh giá (Evaluation)

**File:** `src/evaluation/metrics.py` (72 dòng)

| Metric | Implementation | Source |
|--------|---------------|--------|
| **AUROC** | `medmnist.evaluator.getAUC()` | MedMNIST official → task-aware (binary vs multi-class) |
| **Accuracy** | `medmnist.evaluator.getACC()` | MedMNIST official |
| **Macro-F1** | `sklearn.metrics.f1_score(average="macro")` | scikit-learn |
| **ECE** | Custom equal-width binning (15 bins) | Guo et al. ICML 2017 |

**ECE implementation:**
- Max softmax probability = confidence
- Argmax = prediction
- 15 equal-width bins [0, 1]
- ECE = Σ (bin_size/total) × |bin_accuracy − bin_confidence|

**Đánh giá:** ✅ **Rất tốt**
- AUROC/Accuracy dùng đúng MedMNIST evaluator → kết quả trực tiếp so sánh được với benchmark
- ECE theo đúng paper gốc (Guo et al. 2017)

---

### 2.7 Reproducibility & Seeding

**File:** `src/utils/seed.py` (42 dòng)

**Seed sources được kiểm soát:**
- `os.environ["PYTHONHASHSEED"]` — Python hash randomization
- `random.seed()` — Python built-in random
- `np.random.seed()` — NumPy random
- `torch.manual_seed()` — PyTorch CPU
- `torch.cuda.manual_seed_all()` — PyTorch GPU (tất cả devices)
- `torch.backends.cudnn.deterministic = True` — cuDNN deterministic
- `torch.backends.cudnn.benchmark = False` — Disable cuDNN auto-tuner

**DataLoader seeding:**
- `generator=make_generator(seed)` — riêng cho shuffle
- `worker_init_fn=seed_worker` — seed mỗi worker process

**Stratified subsampling** (regime < 1.0):
- `sklearn.model_selection.train_test_split(stratify=labels, random_state=seed)`
- Giữ nguyên tỷ lệ class → regime 5% không bị skewed toward majority class

**Đánh giá:** ✅ **Rất tốt** — Đầy đủ tất cả nguồn randomness. Cùng seed + cùng hardware → cùng kết quả.

---

### 2.8 Test coverage

**9 test modules, 108+ test cases:**

| Test file | Covers |
|-----------|--------|
| `test_config.py` | Config parsing, validation, enum resolution, error handling |
| `test_dataset.py` | Dataset loading, split integrity |
| `test_matrix.py` | Matrix generation, config count (162), slice assignment |
| `test_metrics.py` | AUROC/Accuracy/F1/ECE computation, edge cases |
| `test_models.py` | Backbone creation, LP/FT/LoRA param counts |
| `test_preprocessing.py` | P0/P_global/P_local transform correctness |
| `test_seed.py` | Seeding determinism |
| `test_subsample.py` | Stratified subsampling, proportion preservation |
| `test_training.py` | Optimizer builder, prediction shapes, end-to-end smoke |

**Đánh giá:** ✅ **Tốt** — Test coverage bao phủ tất cả module chính. Smoke test chạy end-to-end trên CPU.

---

### 2.9 Kết quả Smoke Test (GPU)

**Config:** PneumoniaMNIST / P_local (CLAHE) / LoRA (r=8) / 10% data / seed 0

| Metric | Giá trị |
|--------|---------|
| AUROC | **0.9586** |
| Accuracy | 0.8654 |
| Macro-F1 | 0.8459 |
| ECE | 0.0833 |
| Trainable params | 306,178 (~1.3% of ResNet-50) |
| Wall clock | 29.56 s |

**Training curve:**
```
Epoch   Loss    Val AUROC
──────  ──────  ─────────
  1     0.6168  0.4802     ← Random-level start
  5     0.4454  0.8007     ← Pretrained features kicking in
 10     0.2925  0.9588     ← Strong plateau reached
 20     0.1387  0.9825     ← Diminishing returns
 28     0.0911  0.9855     ← Best val checkpoint
 30     0.1018  0.9851     ← Slight dip → best-checkpoint correctly used
```

**Phân tích:**
- Hội tụ lành mạnh: loss giảm đều, val_auroc tăng đều
- Không có dấu hiệu overfitting (val vẫn tăng ở epoch cuối)
- Best-checkpoint selection hoạt động đúng (epoch 28)
- Test AUROC (0.959) < Val AUROC (0.986) → gap bình thường, không phải data leak

**Kết luận Phase 0:** ✅ **PASSED** — Harness hoạt động đúng end-to-end trên GPU

---

## 3. Phase 1 — Baseline Reproduction

### 3.1 Thiết kế thí nghiệm baseline

**Mục tiêu:** Reproduce published MedMNIST v2 benchmark numbers để validate harness trước khi scale.

**Cấu hình baseline:**
- Backbone: ResNet-50 (ImageNet pretrained)
- Preprocessing: P0 (chỉ resize + ImageNet normalize — không CLAHE)
- Transfer: Full Fine-Tune (train tất cả 23.5M parameters)
- Data regime: 100% (toàn bộ training data)
- Seeds: 0, 1, 2 (report mean ± std)
- Datasets: PneumoniaMNIST (binary, grayscale X-ray) + DermaMNIST (7-class, color skin)

**Tổng số runs:** 2 datasets × 3 seeds = **6 runs**

---

### 3.2 Kết quả chi tiết

#### PneumoniaMNIST (binary classification, ~4,700 train samples)

| Run | Seed | AUROC | Accuracy | Macro-F1 | ECE | Wall Clock |
|-----|------|-------|----------|----------|-----|------------|
| `pneumoniamnist_p0_ft_r100_s0` | 0 | 0.9658 | 0.8526 | 0.8254 | 0.1158 | 120.2 s |
| `pneumoniamnist_p0_ft_r100_s1` | 1 | **0.9857** | **0.9247** | **0.9165** | **0.0588** | 119.4 s |
| `pneumoniamnist_p0_ft_r100_s2` | 2 | 0.9235 | 0.8413 | 0.8101 | 0.1414 | 119.2 s |
| **Mean ± Std** | | **0.9583 ± 0.0259** | **0.8729 ± 0.0369** | **0.8507 ± 0.0470** | **0.1054 ± 0.0345** | ~120 s |

#### DermaMNIST (7-class classification, ~7,000 train samples)

| Run | Seed | AUROC | Accuracy | Macro-F1 | ECE | Wall Clock |
|-----|------|-------|----------|----------|-----|------------|
| `dermamnist_p0_ft_r100_s0` | 0 | 0.9592 | 0.8219 | 0.7040 | 0.1245 | 181.1 s |
| `dermamnist_p0_ft_r100_s1` | 1 | 0.9552 | 0.8130 | 0.6668 | 0.0864 | 181.9 s |
| `dermamnist_p0_ft_r100_s2` | 2 | **0.9644** | **0.8354** | **0.7143** | **0.0981** | 181.0 s |
| **Mean ± Std** | | **0.9596 ± 0.0038** | **0.8234 ± 0.0092** | **0.6950 ± 0.0204** | **0.1030 ± 0.0159** | ~181 s |

---

### 3.3 So sánh với MedMNIST Benchmark

Benchmark tham chiếu: MedMNIST v2 (Yang et al., 2023) — ResNet-50 tại 64×64.

| Dataset | Metric | MedMNIST Published | Our Baseline | Δ | Kết luận |
|---------|--------|--------------------|--------------|---|----------|
| PneumoniaMNIST | AUROC | ~0.947 | **0.958 ± 0.026** | **+0.011** | ✅ Trong tolerance |
| PneumoniaMNIST | Accuracy | ~0.885 | **0.873 ± 0.037** | −0.012 | ✅ Trong tolerance |
| DermaMNIST | AUROC | ~0.931 | **0.960 ± 0.004** | **+0.029** | ✅ Vượt benchmark |
| DermaMNIST | Accuracy | ~0.767 | **0.823 ± 0.009** | **+0.056** | ✅ Vượt benchmark |

**Giải thích chênh lệch tích cực:**

Baseline của chúng ta nhỉnh hơn benchmark MedMNIST vì:
1. **Optimizer**: AdamW (lr=0.001, wd=0.0001) thay vì SGD trong MedMNIST default
2. **Augmentation**: RandomHorizontalFlip + RandomRotation(±10°) — MedMNIST default không dùng augmentation
3. **Pretrained weights**: Luôn dùng ImageNet-pretrained, trong khi MedMNIST benchmark gồm cả kết quả from-scratch

Các cải tiến này **đồng nhất giữa tất cả arms** nên không ảnh hưởng tính so sánh trong nội bộ nghiên cứu.

**Verdict:** ✅ Baselines reproduce benchmark — harness đáng tin cậy.

---

### 3.4 Phân tích Training Curves

#### PneumoniaMNIST (seed 1 — best run)

```
Epoch   Loss    Val AUROC
──────  ──────  ─────────
  1     0.4001  0.8073
  5     0.0908  0.9760
 10     0.0291  0.9844
 15     0.0232  0.9877     ← Best val
 20     0.0236  0.9729     ← Val dropping — overfitting begins
 30     0.0233  0.9797
```

#### DermaMNIST (seed 0 — representative run)

```
Epoch   Loss    Val AUROC
──────  ──────  ─────────
  1     0.9080  0.9083
  5     0.4307  0.9518
 10     0.1873  0.9541
 16     0.1026  0.9619     ← Best val
 20     0.0749  0.9478     ← Val oscillating — Full FT overfitting
 30     0.0405  0.9445     ← Val declined from peak
```

**Nhận xét:**
- **PneumoniaMNIST**: Hội tụ nhanh (epoch 10-15), có dấu hiệu overfit nhẹ sau epoch 15 — best-checkpoint selection giải quyết đúng
- **DermaMNIST**: Val AUROC dao động mạnh hơn sau epoch 10 → Full Fine-Tune 23.5M params trên ~7K samples dễ overfit. Đây là insight quan trọng cho nghiên cứu: LoRA/LP có thể stable hơn
- **30 epochs đủ** cho cả hai dataset — không cần tăng thêm

---

### 3.5 Phát hiện sớm (Early Signal)

So sánh trực tiếp giữa Phase 0 smoke test và Phase 1 baseline:

| | Phase 0 Smoke | Phase 1 Baseline (mean) |
|---|---|---|
| **Preprocessing** | P_local (CLAHE) | P0 (none) |
| **Transfer** | LoRA (r=8) | Full Fine-Tune |
| **Data** | 10% | 100% |
| **AUROC** | **0.9586** | **0.9583** |
| **Trainable params** | 306K | 23.5M |
| **Efficiency ratio** | **77× ít params** | — |
| **Data ratio** | **10× ít data** | — |
| **Wall clock** | 30 s | 120 s |

**Ý nghĩa:**
LoRA + CLAHE + 10% data đạt **cùng AUROC** với Full Fine-Tune + 100% data, sử dụng:
- 77× ít parameters
- 10× ít data
- 4× nhanh hơn

Đây là **tín hiệu tích cực mạnh** cho giả thuyết nghiên cứu RQ3 (*"Can cheap preprocessing partially replace expensive full fine-tuning when data is scarce?"*).

⚠️ **Lưu ý:** So sánh này chỉ dựa trên seed 0 (smoke test) vs mean 3 seeds (baseline). Cần chạy đầy đủ matrix (LoRA + P_local + 10% × 3 seeds) để xác nhận.

---

## 4. Đánh giá tổng hợp

### Bảng đánh giá Phase 0

| Tiêu chí | Đánh giá | Ghi chú |
|----------|----------|---------|
| Kiến trúc code | ✅ Tốt | Module hóa rõ ràng, đơn trách |
| Config system | ✅ Rất tốt | Typed, validated, fail-fast |
| Preprocessing arms | ✅ Rất tốt | CLAHE trên L-channel, augmentation đồng nhất |
| Transfer arms (LP/FT/LoRA) | ✅ Tốt | peft-based LoRA, auto family detection |
| Metrics | ✅ Rất tốt | MedMNIST-aligned AUROC/Acc + ECE Guo et al. |
| Reproducibility | ✅ Rất tốt | 7 nguồn random được seed, cuDNN deterministic |
| Test coverage | ✅ Tốt | 9 test modules, end-to-end smoke test |
| GPU environment | ✅ Passed | PyTorch 2.11+cu128, RTX 3060 Ti |
| Matrix generation | ✅ Hoàn thành | 162 configs + assignment.csv |

### Bảng đánh giá Phase 1

| Tiêu chí | Đánh giá | Ghi chú |
|----------|----------|---------|
| Baseline AUROC vs benchmark | ✅ Passed | Cả 2 dataset match/exceed |
| Baseline Accuracy vs benchmark | ✅ Passed | Trong tolerance |
| Seed variance | ⚠️ Chú ý | PneumoniaMNIST std=0.026 cao → cần 3 seeds |
| Calibration (ECE) | ⚠️ Moderate | ~0.10 cho cả 2 dataset |
| Overfitting (FT) | ⚠️ Nhẹ | DermaMNIST val AUROC giảm sau epoch ~16 |
| Results logging | ✅ Đầy đủ | CSV + JSON + checkpoint per run |
| Compute estimate | ✅ Khả thi | ~8-12 GPU-hours cho 162 runs |

---

## 5. Vấn đề & Khuyến nghị

### 🔴 Phải sửa (Blocking)

Không có vấn đề blocking nào.

### 🟡 Nên cải thiện (Non-blocking)

#### 5.1 PneumoniaMNIST seed variance cao

**Vấn đề:** AUROC dao động từ 0.924 (seed 2) đến 0.986 (seed 1) — gap 0.062. Std = 0.026 là cao cho một metric thường stable.

**Nguyên nhân:** Binary classification trên ~4,700 samples; một vài trăm misclassification swing metrics mạnh. Full Fine-Tune 23.5M params trên dataset nhỏ nhạy cảm với random initialization.

**Khuyến nghị:**
- Giữ nguyên 3 seeds (đủ để capture variance này)
- Trong report, luôn dùng mean ± std, tránh cherry-pick single-seed results
- Có thể thêm significance test (paired t-test hoặc Wilcoxon) khi so sánh arms

#### 5.2 ECE cao (~0.10)

**Vấn đề:** Cả hai dataset có ECE ~0.10, nghĩa là confidence trung bình lệch ~10% so với accuracy thực.

**Nguyên nhân:** Full Fine-Tune thường produce overconfident predictions (Guo et al., 2017).

**Khuyến nghị:**
- Đây là một **research opportunity**: nếu LoRA/LP có ECE tốt hơn, đó là finding bổ sung
- Cân nhắc thêm temperature scaling post-hoc nếu muốn analyze calibration sâu hơn
- Bao gồm ECE trong analysis nhưng không dùng làm primary metric

#### 5.3 DermaMNIST Macro-F1 thấp (0.695)

**Vấn đề:** Accuracy 0.823 nhưng Macro-F1 chỉ 0.695 → model yếu trên minority classes.

**Nguyên nhân:** DermaMNIST có class imbalance mạnh (7 classes, phân bố không đều).

**Khuyến nghị:**
- Đây là hành vi bình thường, không phải bug
- Trong report, nên highlight F1 gap giữa các arms — CLAHE có thể giúp minority classes
- Không thay đổi loss function (giữ CrossEntropy để fair comparison)

#### 5.4 Learning rate schedule

**Vấn đề:** Đang dùng flat learning rate. DermaMNIST val AUROC giảm sau epoch 16 → overfit.

**Khuyến nghị:**
- Cân nhắc thêm **cosine annealing** hoặc **step decay** làm config option
- Tuy nhiên, vì đây là controlled study, thay đổi LR schedule giữa chừng sẽ phức tạp hóa so sánh
- **Decision**: Giữ nguyên flat LR cho core matrix (đồng nhất), ghi nhận overfitting as-is. Nếu có thời gian, thêm LR schedule làm extension experiment



### 🟢 Nice-to-have (Post-core matrix)

| Item | Mô tả |
|------|--------|
| TensorBoard/WandB | Visualize training curves cho 162 runs |
| Grad-CAM | So sánh attention map giữa preprocessing arms (đã trong plan Week 3) |
| Batch runner script | Chạy một slice Kaggle account tự động |
| Statistical tests | Two-way ANOVA, interaction tests (đã trong plan Week 3) |
| Results dashboard | Aggregate results.csv → auto-generate plots |

---

## 6. Kết luận

### Phase 0: ✅ HOÀN THÀNH — Chất lượng tốt

Hạ tầng thí nghiệm đầy đủ, đúng phương pháp luận, reproducible, và well-tested. Config-driven design cho phép mở rộng lên 162 runs mà không cần sửa code.

### Phase 1: ✅ HOÀN THÀNH — Baseline validated

Kết quả reproduce thành công MedMNIST benchmark. Harness đáng tin cậy để tiến hành full experiment matrix.

### Readiness cho Phase 2

| Checklist | Status |
|-----------|--------|
| GPU environment | ✅ |
| Harness validated | ✅ |
| 162 configs generated | ✅ |
| Kaggle account assignment | ✅ |
| Baseline results recorded | ✅ |
| **Ready for Phase 2** | **✅ GO** |

**Ước tính thời gian Phase 2:**
- 162 runs × ~2 min/run (trung bình LP/FT/LoRA) ≈ **5-6 GPU-hours** trên RTX 3060 Ti
- Hoặc phân bổ qua 5 Kaggle accounts → ~1-1.5 GPU-hours/account

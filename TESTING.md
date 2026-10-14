# GROGU — Test Plan & Results (Review 3)

Scope: the deep-learning models (MobileNetV2 + Tabular-MLP fusion), the inference layer
(`predict.py`, Grad-CAM), the Flask API/UI (`app.py`, `index.html`) and the dataset generator.

## 1. How to run

```bash
pip install -r requirements-dev.txt
./run_tests.sh            # everything + coverage + junit/html in test_reports/
./run_tests.sh smoke      # seconds - run after every change
./run_tests.sh sanjay     # model/inference tests
./run_tests.sh jai        # API/UI/dataset tests
```
Tests run offline: the ImageNet download is stubbed because the trained checkpoint overwrites all weights.

## 2. Who owns what (graded individually)

| | Sanjay — model & inference | Jai — API, UI & data |
|---|---|---|
| **White-box** | `test_sanjay_model_whitebox.py` (architecture shapes, fusion dim 1312, freeze/unfreeze, Grad-CAM hooks, feature extraction, quantisation), `test_sanjay_dataset_loader.py` | `test_jai_generator_whitebox.py` (identity/transaction maths, 5 template branches, tamper functions, split leakage) |
| **Black-box** | `test_sanjay_metamorphic.py` (metamorphic relations, sensitivity, boundary values on the metadata vector) | `test_jai_blackbox_api.py` (equivalence classes, boundary, error-guessing, schema, HTTP-level) |
| **Shared** | `test_smoke.py` | `test_smoke.py` |

Counts: smoke 8 · Sanjay 51 · Jai 86 (parametrised) = 145 test cases.

## 3. Test-design techniques used

| Technique | Where | Example |
|---|---|---|
| Smoke / sanity | `test_smoke.py` | Model loads, `/health` up, all 4 demo samples return a verdict |
| Unit (white-box) | model, loader, generator | forward output is `(N,4)`; wrong metadata width raises |
| Statement + branch coverage | `.coveragerc` | 93 % overall (see §5) |
| Metamorphic testing | `test_sanjay_metamorphic.py` | MR1 identity re-save; MR2 determinism; MR3 JPEG noise; MR4 cosmetic-field change; MR5 text change leaves metadata features unchanged |
| Differential / sensitivity | MR6, MR7 | dropping `/CreationDate` flips verdict; dropping `/Producer` lowers confidence |
| Boundary-value analysis | metadata Δ-year ∈ {-3,0,1,7,50}; 1-page vs 80-page PDF | |
| Equivalence partitioning | upload types: valid / wrong extension / empty / corrupt / multi-page | |
| Error guessing / negative | junk bytes with `.pdf` name, path traversal, wrong HTTP verb | |
| Contract (schema) testing | response keys & types of `/predict`, `/predict_sample` | |
| Regression oracle | `xfail(strict=True)` tests | a defect that gets fixed turns the test red → forces removing the marker |

## 4. Defect register (found by these tests)

Each is encoded as a strict `xfail`, so the suite stays green today and **fails the day someone fixes it** — then delete the marker.

| ID | Defect | Found by | Suggested fix |
|---|---|---|---|
| D1 | Feature #13 (temporal anachronism) is hard-coded to 0 at inference (`predict.py`) although training data has non-zero values | white-box | compute it from the PDF with `compute_temporal_anachronism` |
| D2 | Verdict flips Font→Metadata Tampered when a PDF is merely re-saved (pypdf adds `/Producer`) — model leans on metadata presence | metamorphic MR1 | normalise metadata artefacts across classes in the generator, retrain |
| D3 | UI claims "100 % offline" but loads Google Fonts | black-box | self-host fonts |
| D4 | "Both Tampered" demo sample is classified Metadata Tampered (~54 %) | black-box | retrain / replace sample |
| D5 | Uploaded PDFs are never deleted (PII retained) | black-box | delete after inference |
| D6 | Dataset generation not reproducible (Faker unseeded) | white-box | `Faker.seed(42)` |
| D7 | Genuine vs font-tampered separable from metadata alone (image-only files have no Creator/Producer/Author) | white-box | give both classes the same metadata |
| D8 | **Security:** `Flask(static_folder='.')` serves the whole project folder — source, model weights, `uploads/` — to anyone; also path-traversal via `%2e%2e%2f` | black-box | `static_folder` → a dedicated `static/` dir; serve `index.html` explicitly |

## 5. Results (this repo, commit 1e9fa98)

`131 passed, 14 xfailed (documented defects), 0 failed` in ~40 s.

| Module | Branch-inclusive coverage |
|---|---|
| `src/model/architecture.py` | 100 % |
| `dataset/dataset_generator.py` | 94 % |
| `app.py` | 93 % |
| `src/model/dataset.py` | 89 % |
| `src/model/predict.py` | 87 % |
| **Total** | **93 %** |

## 6. Limits (be upfront about these)

- `train.py`, `precache_images.py`, `verify_dataset.py` are not unit-tested (they need the full 10 K dataset / GPU hours).
- Only the 4 shipped demo PDFs plus synthetic PDFs are used; there is no real-world / scanned test set.
- Model *accuracy* is not asserted here (that is the evaluation script's job). Measured test accuracy from `training_results.json` is 78.65 %, with ~57 % F1 on the two metadata classes.

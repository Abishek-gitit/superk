# AI-Based Building Damage Assessment and Disaster Recovery Recommendation System

A mini end-to-end prototype that lets a user upload a post-disaster building
image, predicts damage severity using a transfer-learning deep learning
model (EfficientNetB0), and generates rule-based recovery recommendations
plus a downloadable PDF report — all through a Streamlit dashboard.

---

## 1. Project Structure

```
building_damage_ai/
│
├── app.py                  # Streamlit dashboard (main entry point)
├── train_model.py          # Transfer-learning training script
├── predict.py               # Model loading + inference
├── recommendation.py        # Rule-based recovery recommendation engine
├── report_generator.py      # PDF report generation (ReportLab)
├── requirements.txt
├── models/                  # Saved model (.h5) goes here
├── dataset/                 # Training images, organized by class
│   ├── No_Damage/
│   ├── Minor_Damage/
│   ├── Major_Damage/
│   └── Destroyed/
├── assets/                  # Generated plots (e.g. training curves)
└── README.md
```

---

## 2. Dataset

This prototype expects the **xView2 (xBD)** dataset, pre-processed and
organized into per-class image folders (building "chips" cropped from the
full satellite tiles), e.g.:

```
dataset/
├── No_Damage/       *.jpg / *.png
├── Minor_Damage/
├── Major_Damage/
└── Destroyed/
```

> xBD ships with polygon annotations over full satellite images. To use it
> here, first crop each labeled building polygon into its own image chip
> and sort the chips into the four folders above (a one-time preprocessing
> step, not included in this prototype).

If you don't have the dataset yet, you can still run the Streamlit app —
it will simply show a friendly error asking you to train a model first.

---

## 3. Setup

### Step 1 — Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Train the model
Once `dataset/` is populated with your four class folders:
```bash
python train_model.py --data_dir dataset --epochs 15 --batch_size 32
```
This will:
- Load EfficientNetB0 pretrained on ImageNet (frozen convolutional base)
- Train a new classification head on your 4 damage classes
- Save the best model to `models/building_damage_model.h5`
- Save accuracy/loss curves to `assets/training_history.png`

> If you don't have a GPU, training on a small dataset (a few hundred to a
> few thousand images) is still feasible on CPU, just slower.

### Step 4 — Run the Streamlit app
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 4. How to Use the App

1. **Upload Image** — upload a JPG/PNG photo of a building.
2. **Preview** — the uploaded image is displayed.
3. Click **Predict Damage**.
4. View results:
   - Predicted damage class
   - Confidence score
   - Safety status (color-coded)
   - Full probability breakdown chart
   - Recovery recommendations
5. Click **Download PDF Report** to get a formatted report containing the
   image, prediction, safety status, recommendations, and timestamp.

---

## 5. Damage Classes & Recommendations (Rule-Based Engine)

| Damage Class   | Safety Status                  | Example Recommendations                                              |
|----------------|---------------------------------|------------------------------------------------------------------------|
| No Damage      | Safe                            | Safe for occupancy; continue periodic inspection                       |
| Minor Damage   | Safe with Monitoring            | Repair surface cracks; inspect walls/roof; monitor for progression     |
| Major Damage   | Restricted / Unsafe for Normal Use | Structural inspection required; restrict occupancy; repair members  |
| Destroyed      | Unsafe - Do Not Enter            | Evacuate immediately; recommend demolition or reconstruction          |

See `recommendation.py` for the full lookup table — it's intentionally
simple so it can later be replaced with an LLM-generated engineering
narrative without changing the rest of the app.

---

## 6. Notes on Model Choice

- **EfficientNetB0** is used as the backbone because it offers a strong
  accuracy/efficiency trade-off and works well with a frozen base + small
  trainable head on limited data (common in disaster-response datasets).
- Only the classification head is trained (base frozen) to keep training
  fast and reduce overfitting risk on a small dataset. For larger datasets,
  consider unfreezing the top few blocks of EfficientNetB0 for fine-tuning.
- Swap-in alternative: `ResNet50` — `train_model.py` can be adapted by
  replacing the `EfficientNetB0` import/call with
  `tensorflow.keras.applications.ResNet50` and its matching
  `preprocess_input` function (also update `predict.py` accordingly).

---

## 7. Error Handling

- Invalid/corrupt image uploads are caught and shown as a friendly error
  instead of crashing the app (`safe_load_image` in `app.py`).
- Missing model file (`models/building_damage_model.h5`) shows a clear
  message instructing the user to run `train_model.py` first.
- Prediction failures are caught and surfaced without crashing the session.

---

## 8. Future Enhancements

This prototype is intentionally minimal. Planned/possible extensions:

- **Building Localization (YOLO)** — detect and crop individual buildings
  from a full satellite/aerial image before classification, instead of
  assuming a single pre-cropped building per upload.
- **Segmentation (U-Net)** — pixel-level damage segmentation to highlight
  *which parts* of a building are damaged, not just an overall class.
- **Pre- vs Post-Disaster Comparison** — accept a "before" image alongside
  the "after" image and compute a change-detection score for more reliable
  damage estimation.
- **GIS Map Integration** — plot assessed buildings on an interactive map
  (e.g., Folium/Leaflet) using geotagged coordinates, useful for city-wide
  disaster response dashboards.
- **LLM-Generated Engineering Reports** — replace/augment the rule-based
  recommendation engine with an LLM that drafts a narrative structural
  engineering report referencing the specific damage patterns observed.
- **Multi-Disaster Support** — extend beyond generic "damage severity" to
  disaster-specific models/labels for earthquakes, floods, wildfires, and
  hurricanes, each with tailored recommendation rules.

---

## 9. Disclaimer

This is a prototype for demonstration purposes only. Predictions and
recommendations are **not** a substitute for a certified structural
engineer's on-site inspection.

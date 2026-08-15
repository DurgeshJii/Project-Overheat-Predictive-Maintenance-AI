# Project Overheat: Predictive Maintenance AI

## Repository Description

A time-series machine learning project that uses IoT sensor telemetry and a class-balanced Random Forest to predict catastrophic machine failure **24 hours in advance**, with explainable feature importance for maintenance decision-making.

---

## Project Overview

**Project Overheat: Predictive Maintenance AI** is a Data Science capstone focused on preventing unexpected motor failures in automated warehouse sorting robots.

The project explores how IoT telemetry can be transformed into actionable predictive-maintenance signals rather than waiting for a machine to fail.

The core objective is:

> **Predict whether a catastrophic machine failure will occur within the next 24 hours.**

The project uses 10,000 hourly observations containing machine temperature, vibration, voltage, and catastrophic-failure indicators.

## Business Problem

In the project scenario, AeroLogistics operates high-speed automated sorting robots whose motor failures can cause major warehouse disruption. The objective is to transition from:

**Reactive Maintenance → Predictive Maintenance**

Instead of discovering a problem after a motor fails, the system attempts to identify high-risk machines early enough for maintenance teams to investigate them.

## Dataset

The dataset contains:

* **10,000 hourly observations**
* Timestamp
* Sensor temperature
* Sensor vibration
* Sensor voltage
* Catastrophic failure indicator

The original dataset contains only **15 catastrophic failures**, representing approximately **0.15%** of observations. This severe class imbalance makes failure prediction significantly more challenging than ordinary classification.

## Machine Learning Pipeline

### 1. Data Preparation

* Loaded IoT telemetry
* Checked missing values
* Converted timestamps to datetime
* Sorted observations chronologically

### 2. Time-Series Feature Engineering

Two rolling features were created:

* `Temp_Rolling_Mean_12h`
* `Vibration_Rolling_Std_12h`

The purpose was to capture longer-term machine behavior rather than relying only on individual sensor readings.

### 3. 24-Hour Target Engineering

The original catastrophic-failure signal was shifted by 24 hours to create:

`Failure_In_24_Hours`

This changes the prediction problem from:

**"Is the machine failing now?"**

to:

**"Will the machine fail 24 hours from now?"**

The original failure column was removed from the modeling dataset to reduce target leakage.

### 4. Time-Series Train/Test Split

Instead of randomly shuffling observations, the project uses a chronological split:

* **80%:** Training
* **20%:** Testing

This prevents future observations from leaking into the training process.

### 5. Model

A **Random Forest Classifier** was trained with class balancing to address the extremely rare failure events.

Key configuration included:

* 300 trees
* `class_weight="balanced"`
* `random_state=42`
* `min_samples_leaf=8`
* `max_features="sqrt"`

## Results

The chronological test set contained:

* **1,993 observations**
* **3 actual failure events**

Model performance:

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 99.75% |
| Failure Precision | 33.33% |
| Failure Recall    | 66.67% |
| Failure F1        | 44.44% |
| ROC-AUC           | 99.63% |

### Confusion Matrix

* True Negatives: **1,986**
* False Positives: **4**
* False Negatives: **1**
* True Positives: **2**

Because the failure class is extremely rare, accuracy should not be considered the primary success metric. Failure-class recall and precision provide a more meaningful view of the model's usefulness.

## Explainable AI

Feature importance from the Random Forest showed:

| Feature                     | Importance |
| --------------------------- | ---------: |
| `Sensor_Vibration_mm`       |   0.388978 |
| `Temp_Rolling_Mean_12h`     |   0.265527 |
| `Sensor_Temp_C`             |   0.259787 |
| `Vibration_Rolling_Std_12h` |   0.055106 |
| `Sensor_Voltage_V`          |   0.030603 |

### Key Finding

**Vibration was the strongest predictive feature in this experiment**, followed by the 12-hour temperature trend and raw temperature.

This gives the maintenance team an interpretable signal instead of treating the model as a black box.

## Business Recommendation

Machines receiving a high-risk prediction should be inspected before the predicted 24-hour failure window.

The recommended monitoring priority is:

1. **Vibration**
2. **12-hour temperature trend**
3. **Temperature level**

The model should also be monitored using failure-class **recall and precision**, rather than relying on accuracy alone.

## Technology Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Random Forest
* Time-Series Feature Engineering
* Explainable AI

## Project Structure

```text
Project-Overheat-Predictive-Maintenance-AI/
│
├── data/
│   └── iot_sensor_telemetry.csv
│
├── notebooks/
│   └── predictive_maintenance.ipynb
│
├── outputs/
│   ├── vibration_rolling_std.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── README.md
└── requirements.txt
```

## What I Learned

This project reinforced several important Data Science principles:

* Time-series data requires time-aware validation.
* Feature engineering can be more important than model complexity.
* Random train/test splitting can create serious temporal leakage.
* Highly imbalanced datasets make accuracy misleading.
* Failure prediction should prioritize recall and precision.
* Explainable AI helps translate model output into business decisions.

## Final Takeaway

The goal of predictive maintenance is not simply to build a model with a high accuracy score.

The real objective is to convert sensor data into an **early warning system** that helps maintenance teams act before a costly failure occurs.

**Project Overheat demonstrates that workflow:**

IoT Telemetry → Feature Engineering → 24-Hour Target → Leakage-Safe Validation → Balanced Random Forest → Explainable Predictions → Maintenance Action

# By- Durgesh Yadav

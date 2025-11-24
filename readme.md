# Oil Spill Classification

**Author:** Ipek Koçışarli  
**Task:** Binary classification to detect oil spill presence from tabular data  

---

## Project Overview
This project aims to **detect oil spills** using a neural network on a small, imbalanced dataset.  
The workflow covers a full ML pipeline:

1. Data loading and inspection  
2. Data preprocessing (scaling, removing constant features)  
3. Stratified train/validation/test split  
4. Handling class imbalance using **class weights**  
5. Neural network model building and training  
6. Threshold tuning to maximize **F1 score** for the minority class  
7. Evaluation using confusion matrix, classification report, and ROC/AUC  

---

## Dataset
- **Samples:** 937  
- **Features:** 50 (after dropping constant columns)  
- **Target:** `0` = no oil spill, `1` = oil spill  
- Highly imbalanced, with very few positive (spill) cases  

---

## Model Architecture
- Input layer: 50 features  
- Dense layers: 64 → 32 → 16 neurons with ReLU activation  
- BatchNormalization + Dropout (0.3) for regularization  
- Output layer: 1 neuron, sigmoid activation  
- Optimizer: Adam, learning rate 1e-3  
- Loss: Binary crossentropy  
- Class weights applied to address imbalance  

---

## Key Results

- **Test Accuracy:** 95.7% (high due to the majority class)  

- **Oil Spill Class (minority, 1):**
  - **Recall:** 0.83 → model correctly detects most spills (5 out of 6)  
  - **Precision:** 0.50 → some false alarms occur  
  - **F1-score:** 0.625 → reasonable balance between precision and recall  

- **Non-Spill Class (majority, 0):**
  - **Precision & Recall:** >0.96 → almost all non-spill cases are correctly predicted  

- **Optimal Threshold:** 0.44 (instead of default 0.5) to improve minority class recall  

**Observations:**  
- The model **detects most oil spills**, which is crucial for safety-critical applications.  
- Some false positives are expected due to the small and imbalanced dataset.  
- Overall, the performance is **realistic and acceptable** given the data limitations.  

---

## Visualizations
The notebook includes:  
- Confusion matrix  
- ROC curve  
- Training/validation loss and accuracy curves  

These visualizations help evaluate the model performance for both majority and minority classes.

---

## How to Run
1. Clone the repository:  
```bash
git clone <repo-url>
```
2. Install dependencies

```bash
pip install -r requirements.txt
```
Open the Jupyter Notebook notebooks/OilSpill_Classification.ipynb

Update the dataset path if needed

Run all cells sequentially to reproduce results

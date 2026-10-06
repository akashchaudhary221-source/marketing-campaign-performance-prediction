[README (1).md](https://github.com/user-attachments/files/33103003/README.1.md)

# Marketing Campaign Performance Prediction System

Built for **aeoflo AB** (Stockholm) as part of my MSc dissertation in Data Science for Business at the University of Stirling, through the Work-Based Learning programme.

The goal of this project was to build a machine learning pipeline that predicts whether a marketing campaign will succeed or fail, based on its setup and targeting — before it goes live. It covers everything from exploratory data analysis and feature engineering to model comparison and a live interactive demo presented to the company's co-founder and CEO.

> **Note on datasets:** The files in this repo are sample extracts only. Full data has been withheld under a confidentiality agreement with aeoflo AB.

## The Business Problem

aeoflo AB runs marketing campaigns across Facebook, Instagram, Pinterest, and Twitter. The problem was simple: how do you know if a campaign is going to work before you spend money on it? The original brief asked for ROI prediction, but none of the available datasets had revenue data. So the project was reframed — instead of predicting returns, the system predicts success or failure using a composite score built from metrics like click-through rate, conversion rate, and cost efficiency. This was agreed with the client.

## The Data

Two datasets were used across three modelling iterations.

The first was a Facebook Ads dataset from Kaggle with 1,143 rows. After cleaning out corrupted rows and zero-spend campaigns, 558 records were usable. It was good for getting started but limited — only ages 30 to 49, anonymous interest codes, and Facebook only.

The second was a Social Media Advertising dataset from Kaggle with over 300,000 rows covering four platforms. A 2,000-row stratified sample was used for the final model (random_state=42), giving much better coverage across channels.

### About the sample files

The `data/` folder has two CSV files so you can run the notebooks locally. `facebook_sample.csv` is a cleaned extract from the Facebook dataset with the same structure used in Iteration 1. `social_media_sample.csv` is the exact 2,000-row sample used in Iterations 2 and 3 to train the final model, covering all four channels.

The full Facebook dataset is publicly available on Kaggle. The full Social Media Advertising dataset is also on Kaggle, but the complete version has been kept out of this repo per the client agreement.

## What I Did

The project ran through three iterations.

**Iteration 1** used the Facebook dataset to build the composite scoring framework and run a first classification pipeline. This is where I found a data leakage problem — some features were post-campaign outcomes, not pre-campaign inputs. The model looked accurate but was cheating.

**Iteration 2** switched to the multi-channel dataset and used only pre-campaign features. Accuracy dropped, which was expected — the model was now making real predictions from information you actually have before launching.

**Iteration 3** was the final model. All four channels, composite threshold tuned to 0.4626, and pre-campaign features only (acquisition cost, impressions, campaign type, etc.). Logistic Regression came out on top after comparing all three classifiers.

A Gradio demo was also built so non-technical stakeholders could interact with the model. You put in seven campaign parameters and get a YES/NO prediction with a confidence score.

## Results

Test set performance (400 rows, 200 per class):

| Model | Accuracy | Precision | Recall | F1 Score | Train/Test Gap |
|---|---|---|---|---|---|
| **Logistic Regression** ✓ | **79.5%** | 78% | **85%** | 81% | 0.31% |
| Decision Tree | 78.5% | 77% | 78% | 77% | ~2% |
| Random Forest | 74.5% | 76% | 81% | 78% | ~4% |

Logistic Regression was chosen as the final model. The 85% recall mattered most — it's better to flag a campaign that might fail than to miss one that could have worked. The 0.31% train/test gap shows it's not overfitting.

## Key Takeaways

No revenue data means no ROI prediction — that was the hard constraint. Reframing it as success/failure classification still gives the client something actionable.

Removing data leakage hurt accuracy short-term but was the right call. The final model's numbers are honest.

A single platform wasn't enough. Adding Instagram, Pinterest, and Twitter made the model more generalisable, even if it cost some accuracy on Facebook-specific patterns.

## Tools Used

`Python` · `pandas` · `scikit-learn` · `NumPy` · `matplotlib` · `Gradio` · `Jupyter Notebook` · `pickle`

## Repository Structure

```
marketing-campaign-prediction/
├── data/
│   ├── facebook_sample.csv         # Sample only — full data withheld
│   └── social_media_sample.csv     # 2000-row stratified sample
├── notebooks/
│   ├── 01_facebook_eda.ipynb
│   ├── 02_iteration1_facebook.ipynb
│   ├── 03_iteration2_multichannel.ipynb
│   └── 04_final_model.ipynb
├── demo/
│   ├── gradio_app.py
│   ├── campaign_model.pkl
│   └── scaler.pkl
├── requirements.txt
└── README.md
```

## Running the Demo

```bash
git clone https://github.com/akash-chaudhary/marketing-campaign-prediction.git
cd marketing-campaign-prediction
pip install -r requirements.txt
python demo/gradio_app.py
```

Enter budget, duration, impressions, acquisition cost, target audience, campaign goal, and platform — the model returns YES or NO with a confidence score.

---

*MSc Data Science for Business, University of Stirling (2025-26). Work-Based Learning project with aeoflo AB. Presented to Zinan Lin, Co-founder and CEO, September 2026.*

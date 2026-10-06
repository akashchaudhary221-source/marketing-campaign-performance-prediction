Marketing Campaign Performance Prediction System
📍 Built for aeoflo AB (Stockholm) 🎓 University of Stirling · Work-Based Learning Programme

This project is the practical output of my MSc dissertation in Data Science for Business at the University of Stirling, completed in collaboration with aeoflo AB — a Stockholm-based marketing technology startup. The goal was to build a machine learning pipeline that could predict, before a campaign goes live, whether it is likely to succeed or fail based on its configuration and targeting setup.

The system goes beyond simple model training. It includes exploratory analysis, iterative feature engineering across multiple datasets, a composite scoring framework to handle the absence of direct revenue data, and a live Gradio demo built for stakeholder presentation to the company's co-founder and CEO.

Dataset notice: The datasets included in this repository are sample extracts. Full data has been withheld in accordance with the client confidentiality agreement signed with aeoflo AB under the University of Stirling's Work-Based Learning programme.
The Business Problem
aeoflo AB needed a way to evaluate the likely performance of marketing campaigns without waiting for post-campaign results. For a startup working across multiple social media channels — Facebook, Instagram, Pinterest, and Twitter — spending budget on poorly configured campaigns is a real operational risk.

The challenge was that the original brief asked for ROI prediction, but none of the publicly available datasets included revenue figures. This meant the project had to be reframed: instead of predicting a monetary return, the system predicts whether a campaign will succeed or fail, using a composite performance score built from measurable metrics like click-through rate, conversion rate, and cost efficiency. This reframing was discussed and agreed with the client and is documented fully in the dissertation.

The Data
The project worked through two main datasets across three modelling iterations:

The first was a Facebook Ads dataset sourced from Kaggle, containing 1,143 rows of campaign-level data across 15 columns. After removing structurally corrupted rows and campaigns with zero spend, 558 usable records remained. This dataset was useful for initial exploration but had significant limitations: it only covered ages 30–49, used anonymous interest codes, and was Facebook-only.

The second was a Social Media Advertising dataset from Kaggle with over 300,000 rows covering four platforms. A stratified 2,000-row sample was taken, filtered down to the most relevant records, and used for the final model. This dataset had much better channel diversity and pre-campaign feature coverage, though it lacked direct revenue data.

Both datasets required substantial cleaning — handling column shifts, zero-spend rows, and missing conversions — before any modelling could begin.

About the Sample Files
The data/ folder contains two sample CSV files included so you can run the notebooks locally and follow along with the analysis. facebook_sample.csv is a cleaned extract from the original Facebook Ads dataset — it has the same column structure used in Iteration 1, with the corrupted rows and zero-spend entries already removed. social_media_sample.csv is the 2,000-row stratified sample drawn from the full Social Media Advertising dataset (random_state=42), covering all four channels: Facebook, Instagram, Pinterest, and Twitter. This is the exact data used in Iterations 2 and 3 and for training the final model.

The full datasets are not included here. The Facebook dataset is publicly available on Kaggle. The Social Media Advertising dataset is also available on Kaggle, but the full 300,000-row version has been withheld from this repository per the confidentiality agreement with aeoflo AB.

What I Did
The project was structured as three modelling iterations, each building on lessons from the previous one.

In the first iteration, I built the composite scoring framework on the Facebook dataset and ran an initial classification pipeline. This revealed a data leakage issue: several of the engineered features (like approved conversion rate) were post-campaign outcomes, not pre-campaign inputs. The model was accurate, but for the wrong reason.

The second iteration moved to the multi-channel dataset and switched entirely to pre-campaign features. This removed the leakage but also dropped accuracy, which was the expected trade-off — the model was now making genuine predictions from upfront information only.

The third and final iteration combined all four channels, tuned the composite threshold, and focused on pre-campaign inputs like acquisition cost, impressions, and campaign type. The threshold was set at 0.4626, producing a balanced split between successful and unsuccessful campaigns in the training set. Logistic Regression was selected as the final model after comparing all three classifiers on both accuracy and recall.

A Gradio demo was built at the end to let non-technical stakeholders interact with the model directly. Users input seven campaign parameters (budget, duration, target audience, channel, campaign goal, and acquisition cost), and the model returns a YES/NO prediction with a confidence score.

Results
Final model performance on the held-out test set (400 rows, 200 per class):

Model	Accuracy	Precision	Recall	F1 Score	Train/Test Gap
Logistic Regression ✓	79.5%	78%	85%	81%	0.31%
Decision Tree	78.5%	77%	78%	77%	~2%
Random Forest	74.5%	76%	81%	78%	~4%
Logistic Regression was chosen as the final model. The 85% recall was the deciding factor — in a marketing context, missing a campaign that would have succeeded (false negative) is more costly than flagging one that fails. The near-zero train/test gap also confirms that the model generalises well and is not overfitting to the training data.

Key Insights
A few things stood out during the project that were worth documenting beyond the model numbers.

Data availability is the real bottleneck in marketing ML. The original brief asked for ROI prediction, but without revenue data that simply was not possible. The reframing to binary success/failure still delivers real business value — it is arguably more actionable — but it is a constraint the client should be aware of for future work.

Removing data leakage hurt accuracy, and that is fine. The first model hit 78.57% accuracy across all three classifiers, which looked impressive. But those results were driven by post-campaign features. Once the pipeline was restricted to genuine pre-campaign inputs, accuracy dropped, stabilised, and then recovered — and the final model's predictions are actually meaningful.

Channel diversity matters. The single-platform Facebook dataset was too narrow to build a generalised prediction system. Adding Instagram, Pinterest, and Twitter data significantly improved the model's real-world applicability, even at the cost of some accuracy on the Facebook-specific patterns.

Composite scoring works when revenue data does not exist. Building a six-metric composite score and deriving a binary label from a data-driven threshold was a practical workaround that kept the project grounded in business reality rather than forcing a target variable that did not exist.

Tools and Technologies
Python
pandas
scikit-learn
NumPy
matplotlib
Gradio
Jupyter Notebook
pickle
Repository Structure
marketing-campaign-prediction/
├── data/
│ ├── facebook_sample.csv   # Sample — full data withheld (confidentiality)
│ └── social_media_sample.csv   # 2000-row stratified sample
├── notebooks/
│ ├── 01_facebook_eda.ipynb
│ ├── 02_iteration1_facebook.ipynb
│ ├── 03_iteration2_multichannel.ipynb
│ └── 04_final_model.ipynb
├── demo/
│ ├── gradio_app.py
│ ├── campaign_model.pkl
│ └── scaler.pkl
├── requirements.txt
└── README.md
Running the Demo
# 1. Clone the repo
git clone https://github.com/akash-chaudhary/marketing-campaign-prediction.git
cd marketing-campaign-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Gradio demo
python demo/gradio_app.py
The demo opens in your browser and takes seven inputs: budget, duration, impressions, acquisition cost, target audience, campaign goal, and platform. It returns a YES / NO prediction with a confidence percentage.

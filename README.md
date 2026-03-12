Customer Churn Prediction using Machine Learning

A machine learning project built for the UCS321 AI for Engineers 
course at Thapar Institute of Engineering and Technology under 
the guidance of Dr. Vikram. The goal was to predict whether a 
customer of a subscription based digital service is likely to 
churn or not using behavioral and billing data.



The Problem

Customer churn is when a customer stops using a subscription 
service. For businesses like Netflix, Spotify or any digital 
platform this is a serious issue because losing customers 
directly impacts revenue and growth. The earlier a company 
can identify a customer who is about to leave the better 
chance they have to retain them.



What Makes This Project Different

Most churn prediction projects use the standard Telco Customer 
Churn dataset from Kaggle. We tried it but found it too small 
and not representative enough for what we wanted to build.

So we generated our own synthetic dataset using Claude AI. 
We designed 16 features ourselves based on real world customer 
behavior patterns including usage data, billing history and 
support interactions. This gave us 1,00,000 customer records 
that were clean, balanced and realistic.

This was one of the more interesting parts of the project 
because we had to actually think about what drives churn 
in real businesses before we could describe the data properly.



Dataset

- Generated synthetically using Claude AI
- 1,00,000 customer records
- 16 features covering customer profile, usage and billing
- Target variable: churn (0 = No Churn, 1 = Churn)

| Feature | Description |
|---------|-------------|
| customer_id | Unique customer identifier |
| gender | Customer gender |
| region | Geographic region |
| age | Customer age |
| tenure_months | How long the customer has been subscribed |
| plan_type | Basic, Standard or Premium |
| autopay_enabled | Whether autopay is on |
| daily_usage_hours | Average daily usage |
| monthly_logins | Number of logins per month |
| features_used | Number of features actively used |
| monthly_charges | Amount billed per month |
| payment_delays | Number of late payments |
| support_tickets | Support requests raised |
| complaint_count | Number of complaints filed |
| satisfaction_score | Customer satisfaction rating |
| churn | Target variable (0 or 1) |



Project Flow

1. Data Collection
2. Data Preprocessing  
3. Model Development
4. Model Performance
5. Model Accuracy
6. Confusion Matrix
7. Our Findings
8. Business Insights
9. Success and Innovation




Data Preprocessing

We followed a full cleaning and preprocessing pipeline:

- Removed unnecessary column (customer_id)
- Separated features (X) and target variable (y)
- Applied One-Hot Encoding to categorical variables 
  (gender, region, plan_type, autopay_enabled)
- Applied StandardScaler to numerical features 
  (age, tenure, usage, billing, support data)
- Handled missing values using median imputation for 
  numerical columns and Unknown fill for categorical columns
- Removed duplicate records and rows with missing churn labels
- Used Stratified Train-Test Split (80% training, 20% testing) 
  to maintain the same churn distribution in both sets
- Integrated all preprocessing and model steps into a 
  single Pipeline to prevent data leakage


Model Development

We chose XGBoost Classifier for this project.

Why XGBoost?
- Handles complex non-linear relationships in data
- Reduces overfitting using regularisation
- High performance on structured and tabular data
- Efficient and scalable on large datasets

Training Process:
- Built a classification model to predict churn (1) or 
  no churn (0)
- Applied RandomizedSearchCV for hyperparameter tuning
- Optimized specifically for ROC-AUC score
- Used 5-fold cross-validation to validate performance
- Selected best model based on highest ROC-AUC



Results

| Metric | Score |
|--------|-------|
| Accuracy | 95.61% |
| ROC-AUC Score | 0.852 |

Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Churn) | 0.97 | 0.98 | 0.98 | 18,007 |
| 1 (Churn) | 0.83 | 0.70 | 0.76 | 1,993 |
| Weighted Avg | 0.95 | 0.96 | 0.95 | 20,000 |

Confusion Matrix:
- True Negatives (correctly predicted no churn): 17,720
- True Positives (correctly predicted churn): 1,402
- False Positives: 287
- False Negatives: 591

The ROC-AUC score of 0.852 means the model has an 85.2% 
chance of correctly ranking a churned customer higher than 
a non-churned one. The curve sitting well above the diagonal 
line confirms the model performs significantly better than 
random guessing.



Key Findings

After analyzing feature importances the top factors driving 
churn were:

1. **Plan Type (Basic)** — highest importance score of 0.32
2. **Satisfaction Score** — second most important at 0.23
3. **Autopay Status** — customers without autopay churn more
4. **Payment Delays** — strong indicator of churn risk
5. **Complaint Count** — more complaints means higher risk

Behavioral features like usage patterns, complaints and 
billing had a significantly greater impact on churn than 
demographic features like age or gender.



Business Recommendations

Based on our findings we suggested four strategies:

1. Focus on Engagement
Keep users engaged through tailored content suggestions 
to reduce daily inactivity which is an early churn signal.

2. Target Basic Plan Users
Customers on basic plans are the highest churn risk. 
Encourage upgrades by showing extra benefits and 
offering special deals.

3. Monitor Inactivity
Set up automated warnings when users start logging in 
less frequently or reducing feature usage.

4. Retention Campaigns
Send targeted offers to customers the model flags as 
likely to churn before they actually leave.


Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| XGBoost | Classification model |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Pipeline, encoding, scaling, evaluation |
| Claude AI | Synthetic dataset generation |
| Matplotlib | ROC curve and confusion matrix plots |



How To Run

1. Clone the repository
2. Install dependencies
3. Run the notebook or script


What We Learned

The biggest learning from this project was that data quality 
matters more than model complexity. Once we had a clean and 
well designed dataset the model performed significantly better.

We also found that behavioral data like how often someone 
complains, whether they pay on time and how much they actually 
use the service tells you far more about churn risk than basic 
demographics like age or gender.

Generating our own dataset using Claude AI was an unexpected 
but genuinely useful approach. It forced us to think deeply 
about the problem before writing a single line of code.

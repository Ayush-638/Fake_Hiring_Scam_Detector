
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Create 'model' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# 1. Sample mock dataset
data = {
    'text': [
        "We are looking for a software engineer with 2+ years experience in Python and Django.",
        "Congratulations! You've won a work-from-home job. Pay $300 upfront for training.",
        "Join our growing team at TechCorp. Great benefits, and exciting projects await.",
        "Get a job instantly! No experience needed. Just send us your Aadhar and pay ₹500."
    ],
    'label': [0, 1, 0, 1]  # 0 = Real, 1 = Fake
}

df = pd.DataFrame(data)

# 2. Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# 3. Train the model
model.fit(df['text'], df['label'])

# 4. Save the model
joblib.dump(model, 'model/fake_job_classifier.pkl')

print("✅ Model trained and saved in 'model/fake_job_classifier.pkl'")

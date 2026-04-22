import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys

# 1. Find file path automatically (Definitive solution to avoid errors)
# Finds the directory where baseline.py is located and appends the csv file name to it
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'cleaned_reviews.csv')

# 2. Load and verify the data
try:
    df = pd.read_csv(csv_path).dropna()
    print(f"Data loaded successfully. Number of rows: {len(df)}")
except FileNotFoundError:
    print(f"ERROR: File not found! Please ensure the file is located at:\n{csv_path}")
    sys.exit() # Prevents the code from continuing and throwing errors if the file is missing

# 3. Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['is_spoiler'], test_size=0.2, random_state=42
)

# 4. Convert texts to vectors (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Train the SVM Model (Baseline required by the instructor)
model = LinearSVC(class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# 6. Get the results
y_pred = model.predict(X_test_tfidf)

print("\n--- BASELINE SVM CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix Visualization (Mandatory per Guideline Section 3.6)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Spoiler Detection - Confusion Matrix')
plt.show()
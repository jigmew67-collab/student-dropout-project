import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

DATA_FILE = 'data.csv'
df = pd.read_csv(DATA_FILE, sep=None, engine='python')
df.columns = [c.replace('\ufeff','').strip() for c in df.columns]
X = df.drop(columns='Target')
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    'Baseline (Most Frequent)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=3000))
    ]),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    'Gradient Boosting': HistGradientBoostingClassifier(random_state=42)
}

print('DATASET')
print('Rows:', len(df))
print('Input features:', X.shape[1])
print('Missing values:', int(df.isna().sum().sum()))
print('\nClass distribution:')
print(y.value_counts())

print('\n20% HOLDOUT RESULTS')
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average='weighted', zero_division=0
    )
    print(f'{name}: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}')

print('\n5-FOLD STRATIFIED CROSS-VALIDATION')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv,
                            scoring={'accuracy':'accuracy','f1':'f1_weighted'}, n_jobs=-1)
    print(f"{name}: accuracy={scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}, "
          f"weighted_f1={scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}")

best = models['Gradient Boosting']
best.fit(X_train, y_train)
pred = best.predict(X_test)
print('\nGRADIENT BOOSTING CONFUSION MATRIX (rows=true, columns=predicted)')
print('Labels: Dropout, Enrolled, Graduate')
print(confusion_matrix(y_test, pred, labels=['Dropout','Enrolled','Graduate']))

rf = models['Random Forest']
rf.fit(X_train, y_train)
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print('\nTOP RANDOM FOREST FEATURE IMPORTANCES')
print(importance)

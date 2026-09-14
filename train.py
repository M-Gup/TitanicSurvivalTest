import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score



df = pd.read_csv("data/Titanic-Dataset.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())



features = ["Pclass","Sex","Age","SibSp","Parch","Fare"]

target = "Survived"

X = df[features]
y = df[target]



#Train/test split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

numerical_features = ["Pclass","Age","SibSp","Parch","Fare"]

categorical_features = ["Sex"]

# Age has missing values
numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

# Handle any missing categorical values
categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])


preprocessor = ColumnTransformer(transformers=[("numerical", numerical_pipeline, numerical_features),("categorical", categorical_pipeline, categorical_features)])

#Model creation
model = Pipeline(steps=[("preprocessor", preprocessor),("classifier", LogisticRegression(max_iter=1000))])


#Model training
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")


#Test predictions
y_pred = model.predict(X_test)

# Probability of survival
y_probability = model.predict_proba(X_test)[:, 1]


#Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#Saving pipeline
joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")
import pandas as pd
import joblib#Allows model saving and loading

from sklearn.model_selection import train_test_split#80% of csv for training, 20% for testing
from sklearn.compose import ColumnTransformer#Helps convert categorical features to numerical features, column scaling
from sklearn.pipeline import Pipeline#This was unfamiliar but apparently multiple steps can be combined into a single pipeline, which is useful for preprocessing and model training
from sklearn.impute import SimpleImputer#Deals with stuff like missing age values from the dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler#Categorical into Numerical Conversion
from sklearn.linear_model import LogisticRegression#We're using Logistic Regression for this model as it's a binary classification problem (survived or not survived)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score#Model Performance Metrics



df = pd.read_csv("data/Titanic-Dataset.csv")

features = ["Pclass","Sex","Age","SibSp","Parch","Fare"]

target = "Survived"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)#42 means set seed, stratify means even splitting

numerical_features = ["Pclass","Age","SibSp","Parch","Fare"]#Things like Fare are likely irrelevant yet still included for sake of completeness, as they may have some correlation with survival

categorical_features = ["Sex"]

# Age has missing values
numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

# Handle any missing categorical values
categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])


preprocessor = ColumnTransformer(transformers=[("numerical", numerical_pipeline, numerical_features),("categorical", categorical_pipeline, categorical_features)])

#Model creation
model = Pipeline(steps=[("preprocessor", preprocessor),("classifier", LogisticRegression(max_iter=1000))])


#Model training
model.fit(X_train, y_train)

#Test predictions
y_pred = model.predict(X_test)

# Probability of survival
y_probability = model.predict_proba(X_test)[:, 1]


#Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)#Performance metric for classification problems, gives percentage of correct predictions, you can make print commands if you want to see the accuracy, confusion matrix, and classification report in the console
roc_auc = roc_auc_score(y_test, y_probability)

#Saving pipeline
joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")
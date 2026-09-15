import streamlit as st#Streamlit helps make a web ui
import pandas as pd
import joblib


#Loading the model using joblib
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(page_title="Titanic Survival Predictor",page_icon="🚢",layout="centered")

st.title("Titanic Survival Predictor")
st.write("Enter the passenger's details below to predict ""their probability of survival.")

# User inputs
pclass = st.selectbox("Passenger Class",options=[1, 2, 3],format_func=lambda x: f"{x}{'st' if x == 1 else 'nd' if x == 2 else 'rd'} Class",)

sex = st.selectbox("Gender",options=["male", "female"])

age = st.number_input("Age",min_value=0.0,max_value=100.0,value=30.0,step=1.0,)

sibsp = st.number_input("Number of Siblings/Spouses Aboard",min_value=0,max_value=10,value=0,step=1)

parch = st.number_input("Number of Parents/Children Aboard",min_value=0,max_value=10,value=0,step=1)

fare = st.number_input("Fare",min_value=0.0,value=30.0,step=1.0,)

# Prediction
if st.button("Predict Survival", type="primary"):

    passenger = pd.DataFrame({"Pclass": [pclass],"Sex": [sex],"Age": [age],"SibSp": [sibsp],"Parch": [parch],"Fare": [fare]})

    prediction = model.predict(passenger)[0]#Using the model to predict whether the passenger is likely to survive (1) or not (0)

    probability = model.predict_proba(passenger)[0][1]#Getting the probability of survival for the passenger, which is the second element in the array returned by predict_proba

    st.divider()

    if prediction == 1:
        st.success("Prediction: Likely to Survive")
    else:
        st.error("Prediction: Likely Not to Survive")

    st.metric(
        "Probability of Survival",#Shows the actual probability alongside Yes or No answer written above.
        f"{probability:.1%}",
    )
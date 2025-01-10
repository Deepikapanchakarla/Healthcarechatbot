from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained models
with open('health.pkl', 'rb') as f:
    clf_decision_tree, clf_svc = pickle.load(f)

# Load other necessary data
training = pd.read_csv('Training.csv')
description_list = pd.read_csv('symptom_Description.csv')
precautionDictionary = pd.read_csv('symptom_precaution.csv')

@app.route('/')
def home():
    return render_template('health.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    symptoms = request.form.get('symptoms')
    num_days = int(request.form.get('num_days'))

    # Predict disease using Decision Tree classifier
    symptoms_list = symptoms.split(',')
    input_vector = [1 if symptom in symptoms_list else 0 for symptom in training.columns[:-1]]
    predicted_disease_dt = clf_decision_tree.predict([input_vector])[0]

    # Predict disease using SVC classifier
    predicted_disease_svc = clf_svc.predict([input_vector])[0]

    # You can choose to use one of the predictions or combine them as needed
    predicted_disease = predicted_disease_dt

    # Get description and precautions for the predicted disease
    disease_description = description_list[description_list['Drug Reaction'] == predicted_disease]['Drug Reaction'].values[0]
    disease_precautions = precautionDictionary[precautionDictionary['Drug Reaction'] == predicted_disease].values[0][1:]

    return render_template('result.html', disease=predicted_disease, description=disease_description, precautions=disease_precautions)

if __name__ == '_main_':
        app.run(debug=True)
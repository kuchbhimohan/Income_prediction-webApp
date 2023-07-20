from flask import Flask, render_template, request, url_for, redirect
import pandas as pd



import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        age = int(request.form['age'])
        workclass = request.form['workclass']
        fnlwgt = int(request.form['fnlwgt'])
        education = request.form['education']
        marital_status = request.form['marital-status']
        occupation = request.form['occupation']
        relationship = request.form['relationship']
        sex = request.form['sex']
        hours_per_week = int(request.form['hours-per-week'])

        # Convert input data to DataFrame
        data = {
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [fnlwgt],
            'education': [education],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'sex': [sex],
            'hours-per-week': [hours_per_week]
        }
        df = pd.DataFrame(data)

        # Load the pre-trained model pipeline
        with open('final_model_3.pkl', 'rb') as f:
            model_pipeline = joblib.load(f)
            

        print(type(model_pipeline))

        # Make the salary prediction
        prediction = (model_pipeline.predict(df))[0]
        salary_range = "greater than 50k" if prediction == 1 else "less than 50k"

        return redirect(url_for('result', salary_range=salary_range))

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
@app.route('/result')
def result():
    salary_range = request.args.get('salary_range', '')
    return render_template('result.html', salary_range=salary_range)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models
model_new = joblib.load('model_actual_new_RandomForestRegressor.pkl')
model_review = joblib.load('model_actual_review_GradientBoostingRegressor.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result_new = None
    prediction_result_review = None

    if request.method == 'POST':
        # Get form data
        date = request.form['date']
        type_of_day = request.form['type_of_day']

        # Parse date and extract month, day, and year
        date_obj = pd.to_datetime(date)
        month_number = date_obj.month
        day = date_obj.day
        year = date_obj.year

        # Convert type_of_day to numerical (for the models)
        type_of_day_numeric = {
            'Normal': 1,
            'National_Holiday': 2,
            'Auspicious': 3,
            'Local Festival': 4
        }.get(type_of_day, 0)

        # Prepare data for the models
        input_data = {
            'Day': day,
            'Day_Of_Week': date_obj.day_of_week,  # Assuming day_of_week is required
            'Month_Number': month_number,
            'Year': year,
            'Type_of_Day': type_of_day_numeric
        }
        input_df = pd.DataFrame([input_data])

        # Make predictions
        prediction_new = model_new.predict(input_df)
        prediction_review = model_review.predict(input_df)

        prediction_result_new = f"The number of new persons : {int(float(prediction_new[0]))}"
        prediction_result_review = f"The number of review person: {int(float(prediction_review[0]))}"

        return render_template('index.html', 
                              input_data=input_data, 
                              prediction_result_new=prediction_result_new,
                              prediction_result_review=prediction_result_review)

    return render_template('index.html', 
                          input_data=None, 
                          prediction_result_new=None,
                          prediction_result_review=None)

if __name__ == '__main__':
    app.run(debug=True)

import streamlit as st
import pandas as pd
from datetime import datetime
from joblib import load

# Load your trained ARIMA model
model = load(r'D:\2024\data science portfolio\article for website\Accident_announcement_prediction_app\model_strmlt.joblib')

def main():
    st.title('Announcement Prediction App')

    # User inputs the date
    user_date = st.date_input("Select a Date")

    # User inputs the time
    user_time = st.time_input("Select a Time")

    # Combine date and time into a single datetime object
    user_datetime = datetime.combine(user_date, user_time)

    # Extract characteristics from the date
    hour = user_datetime.hour
    quarter = (user_datetime.month - 1) // 3 + 1
    month = user_datetime.month
    week_day = user_datetime.weekday()
    is_weekend = week_day >= 5

    # Prepare the features as expected by the model
    features = pd.DataFrame([[hour, quarter, month, week_day, is_weekend]],
                            columns=['HOUR', 'QUARTER', 'MONTH', 'WEEK_DAY', 'IS_WEEKEND'])

    if st.button('Predict Announcement'):
        prediction = model.predict(features)

        if prediction == 1:
            st.success('An announcement is predicted.')
        else:
            st.error('No announcement is predicted.')

if __name__ == "__main__":
    main()

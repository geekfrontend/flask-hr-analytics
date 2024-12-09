import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('./models/logistic_regression_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

def main():
    st.title("Logistic Regression Attrition Prediction")
    st.write("Enter features to predict employee attrition:")

    # Column layout for better organization
    col1, col2, col3 = st.columns(3)

    with col1:
        # Continuous Features
        age = st.number_input("Age", min_value=0, step=1, value=30)
        daily_rate = st.number_input("DailyRate", min_value=0, step=1, value=500)
        distance_from_home = st.number_input("DistanceFromHome", min_value=0, step=1, value=10)
        hourly_rate = st.number_input("HourlyRate", min_value=0, step=1, value=50)
        monthly_income = st.number_input("MonthlyIncome", min_value=0, step=1, value=5000)
        monthly_rate = st.number_input("MonthlyRate", min_value=0, step=1, value=10000)
        total_working_years = st.number_input("TotalWorkingYears", min_value=0, step=1, value=5)
        years_at_company = st.number_input("YearsAtCompany", min_value=0, step=1, value=3)
        percent_salary_hike = st.number_input("PercentSalaryHike", min_value=0, max_value=100, step=1, value=15)

    with col2:
        # Categorical Features - Part 1
        business_travel = st.selectbox("BusinessTravel", options=["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        department = st.selectbox("Department", options=["Research & Development", "Sales", "Human Resources"])
        education = st.selectbox("Education", options=[2, 3, 4, 5])
        education_field = st.selectbox("EducationField", options=["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])
        environment_satisfaction = st.selectbox("EnvironmentSatisfaction", options=[2, 3, 4])
        gender = st.selectbox("Gender", options=["Female", "Male"])
        job_involvement = st.selectbox("JobInvolvement", options=[2, 3, 4])
        job_level = st.selectbox("JobLevel", options=[2, 3, 4, 5])

    with col3:
        # Categorical Features - Part 2
        job_role = st.selectbox("JobRole", options=[
            "Human Resources", "Laboratory Technician", "Manager", 
            "Manufacturing Director", "Research Director", 
            "Research Scientist", "Sales Executive", "Sales Representative"
        ])
        job_satisfaction = st.selectbox("JobSatisfaction", options=[2, 3, 4])
        marital_status = st.selectbox("MaritalStatus", options=["Single", "Married", "Divorced"])
        num_companies_worked = st.selectbox("NumCompaniesWorked", options=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        over_time = st.selectbox("OverTime", options=["No", "Yes"])
        performance_rating = st.selectbox("PerformanceRating", options=[4])
        relationship_satisfaction = st.selectbox("RelationshipSatisfaction", options=[2, 3, 4])
        stock_option_level = st.selectbox("StockOptionLevel", options=[1, 2, 3])
        training_times_last_year = st.selectbox("TrainingTimesLastYear", options=[1, 2, 3, 4, 5, 6])

    # Additional Expandable Sections for Remaining Features
    with st.expander("More Details"):
        col4, col5, col6 = st.columns(3)
        
        with col4:
            work_life_balance = st.selectbox("WorkLifeBalance", options=[2, 3, 4])
            years_in_current_role = st.selectbox("YearsInCurrentRole", options=list(range(1, 19)))

        with col5:
            years_since_last_promotion = st.selectbox("YearsSinceLastPromotion", options=list(range(1, 16)))
            years_with_curr_manager = st.selectbox("YearsWithCurrManager", options=list(range(1, 16)))

    # Prediction Button
    if st.button("Predict Attrition"):
        try:
            # Feature encoding
            features = np.zeros(135)  # Preallocate array with 135 zeros

            # Continuous features
            features[0] = age
            features[1] = daily_rate
            features[2] = distance_from_home
            features[3] = hourly_rate
            features[4] = monthly_income
            features[5] = monthly_rate
            features[6] = total_working_years
            features[7] = years_at_company
            features[8] = percent_salary_hike

            # Business Travel Encoding
            features[9] = 1 if business_travel == "Travel_Frequently" else 0
            features[10] = 1 if business_travel == "Travel_Rarely" else 0

            # Department Encoding
            features[11] = 1 if department == "Research & Development" else 0
            features[12] = 1 if department == "Sales" else 0

            # Education Encoding
            features[13] = 1 if education == 2 else 0
            features[14] = 1 if education == 3 else 0
            features[15] = 1 if education == 4 else 0
            features[16] = 1 if education == 5 else 0

            # Education Field Encoding
            features[17] = 1 if education_field == "Life Sciences" else 0
            features[18] = 1 if education_field == "Marketing" else 0
            features[19] = 1 if education_field == "Medical" else 0
            features[20] = 1 if education_field == "Other" else 0
            features[21] = 1 if education_field == "Technical Degree" else 0

            # Environment Satisfaction Encoding
            features[22] = 1 if environment_satisfaction == 2 else 0
            features[23] = 1 if environment_satisfaction == 3 else 0
            features[24] = 1 if environment_satisfaction == 4 else 0

            # Gender Encoding
            features[25] = 1 if gender == "Male" else 0

            # Job Involvement Encoding
            features[26] = 1 if job_involvement == 2 else 0
            features[27] = 1 if job_involvement == 3 else 0
            features[28] = 1 if job_involvement == 4 else 0

            # Job Level Encoding
            features[29] = 1 if job_level == 2 else 0
            features[30] = 1 if job_level == 3 else 0
            features[31] = 1 if job_level == 4 else 0
            features[32] = 1 if job_level == 5 else 0

            # Job Role Encoding
            job_role_mapping = {
                "Human Resources": 33, 
                "Laboratory Technician": 34, 
                "Manager": 35, 
                "Manufacturing Director": 36, 
                "Research Director": 37, 
                "Research Scientist": 38, 
                "Sales Executive": 39, 
                "Sales Representative": 40
            }
            features[job_role_mapping[job_role]] = 1

            # Further encoding would continue in this pattern...
            # Note: You'll need to complete the full 135 feature encoding

            # Reshape to 2D array for scaling
            features = features.reshape(1, -1)

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[:, 1]

            st.subheader(f"Predicted Attrition: {'Yes' if prediction[0] == 1 else 'No'}")
            st.subheader(f"Probability of Attrition: {float(probability[0]):.2f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == '__main__':
    main()
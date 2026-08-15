
import os
import streamlit as st
import pandas as pd
import joblib



# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")

st.write("Enter the customer details below to predict "
    "whether the customer is likely to purchase "
    "the tourism package."
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    value=10
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=0,
    value=3
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    value=2
)

passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=0
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=25000.0
)

typeofcontact = st.selectbox(
    "Type of Contact",
    [
        "Self Enquiry",
        "Company Invited"
    ]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

productpitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Deluxe",
        "Standard",
        "Super Deluxe",
        "King"
    ]
)

maritalstatus = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

designation = st.selectbox(
    "Designation",
    [
        "AVP",
        "VP",
        "Senior Manager",
        "Manager",
        "Executive"
    ]
)

input_data = pd.DataFrame({

    "Age": [age],

    "CityTier": [city_tier],

    "DurationOfPitch": [
        duration_of_pitch
    ],

    "NumberOfPersonVisiting": [
        number_of_person_visiting
    ],

    "NumberOfFollowups": [
        number_of_followups
    ],

    "PreferredPropertyStar": [
        preferred_property_star
    ],

    "NumberOfTrips": [
        number_of_trips
    ],

    "Passport": [
        passport
    ],

    "PitchSatisfactionScore": [
        pitch_satisfaction_score
    ],

    "OwnCar": [
        own_car
    ],

    "NumberOfChildrenVisiting": [
        number_of_children_visiting
    ],

    "MonthlyIncome": [
        monthly_income
    ],

    "TypeofContact": [
        typeofcontact
    ],

    "Occupation": [
        occupation
    ],

    "Gender": [
        gender
    ],

    "ProductPitched": [
        productpitched
    ],

    "MaritalStatus": [
        maritalstatus
    ],

    "Designation": [
        designation
    ]
})

if st.button(
    "Predict Package Purchase"
):

    prediction_probability = (
        model.predict_proba(
            input_data
        )[:, 1][0]
    )

    prediction = int(
        prediction_probability >= 0.45
    )


    st.subheader(
        "Prediction Result"
    )

    if prediction == 1:

        st.success(
            "The customer is likely to purchase the package."
        )

    else:

        st.info(
            "The customer is unlikely to purchase the package."
        )


    st.write(
        f"Purchase probability: "
        f"{prediction_probability:.2%}"
    )


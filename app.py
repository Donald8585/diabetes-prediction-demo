import gradio as gr
import numpy as np
import pickle
import json
import traceback

# Load model
try:
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
    print(f"   Model expects {model.n_features_in_} features")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Load sample cases
with open('sample_patients.json', 'r') as f:
    sample_patients = json.load(f)

# Get default patient (Healthy 30-year-old)
default_case = sample_patients["Healthy 30-year-old"]

def predict_diabetes(
    age, alcohol, physical_activity, diet_score, sleep_hours,
    screen_time, bmi, waist_hip_ratio, systolic_bp, diastolic_bp,
    heart_rate, cholesterol_total, hdl, ldl, triglycerides,
    gender, ethnicity, education, income, smoking, employment,
    family_history, hypertension, cardiovascular
):
    """Predict diabetes risk based on patient data"""

    try:
        if model is None:
            return "❌ Model not loaded. Please train the model first.", 0.0, {}, "⚠️ Error"

        # Build feature vector matching training encoding
        # Order must match exactly what pandas.get_dummies produced during training

        # Numerical features (15)
        numerical_features = [
            age, alcohol, physical_activity, diet_score, sleep_hours,
            screen_time, bmi, waist_hip_ratio, systolic_bp, diastolic_bp,
            heart_rate, cholesterol_total, hdl, ldl, triglycerides
        ]

        # Categorical features encoded (drop_first=True means we drop first category)
        # Gender: Female (baseline), Male, Other
        gender_male = 1 if gender == "Male" else 0
        gender_other = 1 if gender == "Other" else 0

        # Ethnicity: Asian (baseline), Black, Hispanic, Other, White
        ethnicity_black = 1 if ethnicity == "Black" else 0
        ethnicity_hispanic = 1 if ethnicity == "Hispanic" else 0
        ethnicity_other = 1 if ethnicity == "Other" else 0
        ethnicity_white = 1 if ethnicity == "White" else 0

        # Education: No formal (baseline), Highschool, Undergraduate, Postgraduate
        education_highschool = 1 if education == "Highschool" else 0
        education_no_formal = 1 if education == "No formal" else 0
        education_postgraduate = 1 if education == "Postgraduate" else 0

        # Income: Low (baseline), Lower-Middle, Middle, Upper-Middle, High
        income_low = 1 if income == "Low" else 0
        income_lower_middle = 1 if income == "Lower-Middle" else 0
        income_middle = 1 if income == "Middle" else 0
        income_upper_middle = 1 if income == "Upper-Middle" else 0

        # Smoking: Never (baseline), Former, Current
        smoking_former = 1 if smoking == "Former" else 0
        smoking_never = 1 if smoking == "Never" else 0

        # Employment: Employed (baseline), Retired, Student, Unemployed
        employment_retired = 1 if employment == "Retired" else 0
        employment_student = 1 if employment == "Student" else 0
        employment_unemployed = 1 if employment == "Unemployed" else 0

        # Binary features (3)
        binary_features = [family_history, hypertension, cardiovascular]

        # Combine all features
        features = numerical_features + [
            gender_male, gender_other,
            ethnicity_black, ethnicity_hispanic, ethnicity_other, ethnicity_white,
            education_highschool, education_no_formal, education_postgraduate,
            income_low, income_lower_middle, income_middle, income_upper_middle,
            smoking_former, smoking_never,
            employment_retired, employment_student, employment_unemployed
        ] + binary_features

        features_array = np.array([features])

        # Debug: Print feature count
        print(f"\n🔍 DEBUG INFO:")
        print(f"   Features provided: {len(features)}")
        print(f"   Model expects: {model.n_features_in_}")

        if len(features) != model.n_features_in_:
            error_msg = f"Feature mismatch! Provided {len(features)}, expected {model.n_features_in_}"
            print(f"   ❌ {error_msg}")
            return f"❌ {error_msg}", 0.0, {}, "⚠️ Error"

        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]

        # Determine risk level
        diabetes_prob = probability[1] * 100

        print(f"   ✅ Prediction: {prediction}")
        print(f"   Probability: {diabetes_prob:.2f}%")

        if diabetes_prob < 30:
            risk_level = "🟢 Low Risk"
        elif diabetes_prob < 60:
            risk_level = "🟡 Moderate Risk"
        else:
            risk_level = "🔴 High Risk"

        result = "✅ Diabetes Detected" if prediction == 1 else "✅ No Diabetes"

        # Key factors summary
        key_factors = {
            "Family History": "Yes ⚠️" if family_history else "No ✓",
            "Age": f"{age} years",
            "BMI": f"{bmi:.1f} {'(Overweight)' if bmi > 25 else '(Normal)'}",
            "Physical Activity": f"{physical_activity} min/week {'(Low)' if physical_activity < 60 else '(Good)'}"
        }

        return result, diabetes_prob, key_factors, risk_level

    except Exception as e:
        error_msg = f"❌ Prediction Error: {str(e)}"
        print(f"\n{error_msg}")
        print(traceback.format_exc())
        return error_msg, 0.0, {"Error": str(e)}, "⚠️ Error"

def load_sample_case(case_name):
    """Load a pre-defined sample patient case"""
    case = sample_patients[case_name]
    return [case[key] for key in [
        "age", "alcohol", "physical_activity", "diet_score", "sleep_hours",
        "screen_time", "bmi", "waist_hip_ratio", "systolic_bp", "diastolic_bp",
        "heart_rate", "cholesterol_total", "hdl", "ldl", "triglycerides",
        "gender", "ethnicity", "education", "income", "smoking", "employment",
        "family_history", "hypertension", "cardiovascular"
    ]]

# Build Gradio interface
with gr.Blocks(title="Diabetes Prediction System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Diabetes Risk Prediction System
    ### Machine Learning Model trained on 700,000 patients

    **Model Performance:** 68% accuracy, 72% ROC-AUC  
    **Key Insight:** Family history is the strongest predictor (82% feature importance)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Patient Information")

            # Sample case selector
            sample_dropdown = gr.Dropdown(
                choices=list(sample_patients.keys()),
                label="💡 Load Sample Patient",
                value="Healthy 30-year-old"
            )

            with gr.Tab("Demographics"):
                age = gr.Slider(19, 89, value=default_case["age"], step=1, label="Age")
                gender = gr.Radio(["Female", "Male", "Other"], value=default_case["gender"], label="Gender")
                ethnicity = gr.Radio(["Asian", "Black", "Hispanic", "Other", "White"], value=default_case["ethnicity"], label="Ethnicity")
                education = gr.Radio(["No formal", "Highschool", "Undergraduate", "Postgraduate"], value=default_case["education"], label="Education Level")
                income = gr.Radio(["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"], value=default_case["income"], label="Income Level")
                employment = gr.Radio(["Employed", "Retired", "Student", "Unemployed"], value=default_case["employment"], label="Employment Status")

            with gr.Tab("Lifestyle"):
                alcohol = gr.Slider(1, 9, value=default_case["alcohol"], step=1, label="Alcohol Consumption (drinks/week)")
                physical_activity = gr.Slider(1, 750, value=default_case["physical_activity"], step=5, label="Physical Activity (minutes/week)")
                diet_score = gr.Slider(0.1, 9.9, value=default_case["diet_score"], step=0.1, label="Diet Quality Score (0-10)")
                sleep_hours = gr.Slider(3.1, 9.9, value=default_case["sleep_hours"], step=0.1, label="Sleep (hours/day)")
                screen_time = gr.Slider(0.6, 16.5, value=default_case["screen_time"], step=0.1, label="Screen Time (hours/day)")
                smoking = gr.Radio(["Never", "Former", "Current"], value=default_case["smoking"], label="Smoking Status")

            with gr.Tab("Medical Measurements"):
                bmi = gr.Slider(15.1, 38.4, value=default_case["bmi"], step=0.1, label="BMI")
                waist_hip_ratio = gr.Slider(0.68, 1.05, value=default_case["waist_hip_ratio"], step=0.01, label="Waist-to-Hip Ratio")
                systolic_bp = gr.Slider(91, 163, value=default_case["systolic_bp"], step=1, label="Systolic Blood Pressure")
                diastolic_bp = gr.Slider(51, 104, value=default_case["diastolic_bp"], step=1, label="Diastolic Blood Pressure")
                heart_rate = gr.Slider(42, 101, value=default_case["heart_rate"], step=1, label="Heart Rate (bpm)")
                cholesterol_total = gr.Slider(117, 289, value=default_case["cholesterol_total"], step=1, label="Total Cholesterol")
                hdl = gr.Slider(21, 90, value=default_case["hdl"], step=1, label="HDL Cholesterol (Good)")
                ldl = gr.Slider(51, 205, value=default_case["ldl"], step=1, label="LDL Cholesterol (Bad)")
                triglycerides = gr.Slider(31, 290, value=default_case["triglycerides"], step=1, label="Triglycerides")

            with gr.Tab("Medical History"):
                family_history = gr.Checkbox(label="Family History of Diabetes", value=default_case["family_history"])
                hypertension = gr.Checkbox(label="Hypertension History", value=default_case["hypertension"])
                cardiovascular = gr.Checkbox(label="Cardiovascular History", value=default_case["cardiovascular"])

            predict_btn = gr.Button("🔬 Analyze Risk", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Results")
            result_text = gr.Textbox(label="Diagnosis", interactive=False)
            probability = gr.Number(label="Diabetes Probability (%)", precision=2)
            risk_level = gr.Textbox(label="Risk Level", interactive=False)
            key_factors = gr.JSON(label="Key Health Factors")

            gr.Markdown("""
            ### ℹ️ Model Information

            **Dataset:** 700,000 patients, 26 features  
            **Algorithm:** XGBoost Classifier  
            **Performance:**
            - Accuracy: 68%
            - ROC-AUC: 72%

            **Top Predictors:**
            1. Family History (82% importance)
            2. Age (3.2%)
            3. Physical Activity (3.0%)

            **Note:** This is a demonstration model for educational purposes only.  
            Always consult healthcare professionals for medical advice.
            """)

    # Connect sample selector
    inputs = [
        age, alcohol, physical_activity, diet_score, sleep_hours,
        screen_time, bmi, waist_hip_ratio, systolic_bp, diastolic_bp,
        heart_rate, cholesterol_total, hdl, ldl, triglycerides,
        gender, ethnicity, education, income, smoking, employment,
        family_history, hypertension, cardiovascular
    ]

    sample_dropdown.change(
        fn=load_sample_case,
        inputs=[sample_dropdown],
        outputs=inputs
    )

    predict_btn.click(
        fn=predict_diabetes,
        inputs=inputs,
        outputs=[result_text, probability, key_factors, risk_level]
    )

if __name__ == "__main__":
    demo.launch()

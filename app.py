import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from catboost import CatBoostClassifier

st.set_page_config(page_title="ReadmitAI | Hospital Analytics",page_icon="🏥",layout="wide",initial_sidebar_state="expanded")

BASE_DIR=Path(__file__).resolve().parent
MODEL_DIR=BASE_DIR/"models"
FIGURE_DIR=BASE_DIR/"figures"
METRICS_PATH=FIGURE_DIR/"model_performance_comparison.csv"

st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"]{background:#0B1220!important;color:#E7EEF8!important}
[data-testid="stHeader"]{background:#0B1220!important}
.block-container{max-width:1300px;padding:2rem 2rem 3rem}
h1,h2,h3,h4{color:#E7EEF8!important}
p,label{color:#C5D0E0!important}
[data-testid="stSidebar"]{background:#111C30!important;border-right:1px solid #25344B}
[data-testid="stSidebar"] *{color:#E7EEF8!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#9FB0C8!important}
[data-testid="stSidebar"] hr{border-color:#293B55!important}
[data-testid="stMetric"]{background:#17243A!important;border:1px solid #2A3B54!important;border-radius:14px;padding:18px;box-shadow:0 4px 15px rgba(0,0,0,.15)}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p{color:#9FB0C8!important}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] div,[data-testid="stMetricValue"] span{color:#E7EEF8!important;-webkit-text-fill-color:#E7EEF8!important;font-weight:700!important}
.hero{background:linear-gradient(115deg,#142A48,#164E70);border:1px solid #27577C;border-radius:18px;padding:30px 34px;margin-bottom:22px}
.hero .eyebrow{color:#60C8ED;font-size:12px;letter-spacing:2px;font-weight:700}
.hero h1{color:#FFFFFF!important;font-size:38px;margin:8px 0}
.hero p{color:#D2E5F4!important;margin:0;font-size:16px}
.section-card{background:#17243A;border:1px solid #293D57;border-radius:13px;padding:20px;margin:8px 0}
.section-card h3{color:#E7EEF8!important}
.small-muted{color:#9FB0C8!important;font-size:13px}
[data-testid="stForm"]{background:#121E31!important;border:1px solid #293D57!important;border-radius:14px;padding:20px}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="base-input"]{background:#19283E!important;border-color:#354A65!important;color:#E7EEF8!important}
[data-baseweb="select"] *{color:#E7EEF8!important}
input{color:#E7EEF8!important}
[data-baseweb="popover"]{background:#19283E!important}
[data-baseweb="menu"]{background:#19283E!important}
[data-testid="stDataFrame"]{border:1px solid #293D57;border-radius:12px}
div.stButton>button,[data-testid="stFormSubmitButton"] button{background:#1687B5!important;color:#FFFFFF!important;border:0!important;border-radius:9px!important;font-weight:650!important;padding:9px 18px}
div.stButton>button:hover,[data-testid="stFormSubmitButton"] button:hover{background:#20A4D8!important;color:#FFFFFF!important}
[data-testid="stAlert"]{border-radius:11px}
hr{border-color:#293D57!important}
.stProgress>div>div>div>div{background:#21A6D5}
a{color:#56C4EE!important}
@media(max-width:768px){.block-container{padding:1rem!important}.hero{padding:22px}.hero h1{font-size:29px}}
</style>
""",unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models={
        "Logistic Regression":joblib.load(MODEL_DIR/"logistic_regression.joblib"),
        "Random Forest":joblib.load(MODEL_DIR/"random_forest.joblib"),
        "XGBoost":joblib.load(MODEL_DIR/"xgboost.joblib")
    }
    cb=CatBoostClassifier()
    cb.load_model(str(MODEL_DIR/"catboost_model.cbm"))
    models["CatBoost"]=cb
    config=joblib.load(MODEL_DIR/"feature_config.joblib")
    return models,config

try:
    models,config=load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

numeric_features=config["numeric_features"]
categorical_features=config["categorical_features"]
feature_cols=config["feature_cols"]

metrics=pd.read_csv(METRICS_PATH,index_col=0) if METRICS_PATH.exists() else pd.DataFrame()

ADMISSION_TYPES={
    "Emergency (ID 1)":"1",
    "Urgent (ID 2)":"2",
    "Elective / Planned (ID 3)":"3",
    "Newborn (ID 4)":"4",
    "Not Available (ID 5)":"5",
    "Null / Missing (ID 6)":"6",
    "Trauma Center (ID 7)":"7",
    "Not Mapped (ID 8)":"8"
}

DISCHARGE_TYPES={
    "Discharged to Home (ID 1)":"1",
    "Transferred to Short-Term Hospital (ID 2)":"2",
    "Transferred to Skilled Nursing Facility (ID 3)":"3",
    "Transferred to Intermediate Care Facility (ID 4)":"4",
    "Other Inpatient Care Institution (ID 5)":"5",
    "Home with Home Health Service (ID 6)":"6",
    "Left Against Medical Advice (ID 7)":"7",
    "Home with Home IV Provider (ID 8)":"8",
    "Admitted as Inpatient to Same Hospital (ID 9)":"9",
    "Neonatal Aftercare Hospital (ID 10)":"10",
    "Still Patient / Outpatient Return Expected (ID 12)":"12",
    "Medicare Swing Bed (ID 15)":"15",
    "Outpatient Services at Same Institution (ID 16)":"16",
    "Outpatient Services (ID 17)":"17",
    "Null / Missing (ID 18)":"18",
    "Rehabilitation Facility (ID 22)":"22",
    "Long-Term Care Hospital (ID 23)":"23",
    "Medicaid Nursing Facility (ID 24)":"24",
    "Not Mapped (ID 25)":"25",
    "Unknown / Invalid (ID 26)":"26",
    "Home with Home IV Care (ID 27)":"27",
    "Federal Healthcare Facility (ID 28)":"28",
    "Medicare Approved Swing Bed (ID 29)":"29",
    "Other Healthcare Institution (ID 30)":"30"
}

ADMISSION_SOURCES={
    "Physician Referral (ID 1)":"1",
    "Clinic Referral (ID 2)":"2",
    "HMO Referral (ID 3)":"3",
    "Transfer from Hospital (ID 4)":"4",
    "Transfer from Skilled Nursing Facility (ID 5)":"5",
    "Transfer from Healthcare Facility (ID 6)":"6",
    "Emergency Room (ID 7)":"7",
    "Court / Law Enforcement (ID 8)":"8",
    "Not Available (ID 9)":"9",
    "Critical Access Hospital (ID 10)":"10",
    "Normal Delivery (ID 11)":"11",
    "Premature Delivery (ID 12)":"12",
    "Sick Baby (ID 13)":"13",
    "Extramural Birth (ID 14)":"14",
    "Not Available (ID 15)":"15",
    "Null / Missing (ID 17)":"17",
    "Home Health Agency (ID 18)":"18",
    "Readmission to Same Hospital (ID 19)":"19",
    "Not Mapped (ID 20)":"20",
    "Unknown / Invalid (ID 21)":"21",
    "Hospital Inpatient Transfer (ID 22)":"22",
    "Born Inside Hospital (ID 23)":"23",
    "Born Outside Hospital (ID 24)":"24",
    "Ambulatory Surgery Center (ID 25)":"25",
    "Transfer from Hospice (ID 26)":"26"
}

def prediction_input(form_key):
    with st.form(form_key):
        st.subheader("01 · Patient Demographics")
        c1,c2,c3=st.columns(3)
        race=c1.selectbox("Race",["Caucasian","AfricanAmerican","Asian","Hispanic","Other","Unknown"])
        gender=c2.selectbox("Gender",["Female","Male","Unknown/Invalid"])
        age=c3.selectbox("Age Group",["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"])
        st.divider()
        st.subheader("02 · Hospital Admission Details")
        c1,c2,c3=st.columns(3)
        admission_label=c1.selectbox("Admission Type",list(ADMISSION_TYPES.keys()),index=0)
        discharge_label=c2.selectbox("Discharge Disposition",list(DISCHARGE_TYPES.keys()),index=0)
        source_label=c3.selectbox("Admission Source",list(ADMISSION_SOURCES.keys()),index=0)
        diag_1=st.text_input("Primary Diagnosis Code","250.83")
        st.divider()
        st.subheader("03 · Hospital Encounter Statistics")
        c1,c2,c3,c4=st.columns(4)
        time_in_hospital=c1.number_input("Days in Hospital",1,14,3)
        num_lab_procedures=c2.number_input("Lab Procedures",0,150,40)
        num_procedures=c3.number_input("Procedures",0,6,0)
        num_medications=c4.number_input("Medications",0,100,10)
        c1,c2,c3,c4=st.columns(4)
        number_outpatient=c1.number_input("Previous Outpatient Visits",0,100,0)
        number_emergency=c2.number_input("Previous Emergency Visits",0,100,0)
        number_inpatient=c3.number_input("Previous Inpatient Visits",0,50,0)
        number_diagnoses=c4.number_input("Number of Diagnoses",1,16,5)
        st.divider()
        st.subheader("04 · Diabetes and Laboratory Information")
        c1,c2,c3=st.columns(3)
        max_glu_serum=c1.selectbox("Maximum Glucose Serum",["None","Norm",">200",">300"])
        A1Cresult=c2.selectbox("HbA1c Result",["None","Norm",">7",">8"])
        insulin=c3.selectbox("Insulin Status",["No","Steady","Up","Down"])
        c1,c2,c3=st.columns(3)
        metformin=c1.selectbox("Metformin Status",["No","Steady","Up","Down"])
        change_label=c2.selectbox("Medication Changed",["No","Yes"])
        diabetesMed=c3.selectbox("Diabetes Medication Prescribed",["Yes","No"])
        submitted=st.form_submit_button("Generate Prediction",use_container_width=True)
    if not submitted:
        return None
    return {
        "time_in_hospital":time_in_hospital,
        "num_lab_procedures":num_lab_procedures,
        "num_procedures":num_procedures,
        "num_medications":num_medications,
        "number_outpatient":number_outpatient,
        "number_emergency":number_emergency,
        "number_inpatient":number_inpatient,
        "number_diagnoses":number_diagnoses,
        "race":np.nan if race=="Unknown" else race,
        "gender":gender,
        "age":age,
        "admission_type_id":ADMISSION_TYPES[admission_label],
        "discharge_disposition_id":DISCHARGE_TYPES[discharge_label],
        "admission_source_id":ADMISSION_SOURCES[source_label],
        "diag_1":diag_1.strip() or "Unknown",
        "max_glu_serum":max_glu_serum,
        "A1Cresult":A1Cresult,
        "insulin":insulin,
        "change":"Ch" if change_label=="Yes" else "No",
        "diabetesMed":diabetesMed,
        "metformin":metformin
    }

def prepare_input(input_data):
    data=pd.DataFrame([input_data],columns=feature_cols)
    for col in numeric_features:
        data[col]=pd.to_numeric(data[col],errors="coerce")
    for col in categorical_features:
        data[col]=data[col].fillna("Unknown").astype(str)
    return data

def get_model_probability(name,model,input_df):
    probability=float(model.predict_proba(input_df)[0,1])
    prediction=int(probability>=0.5)
    return probability,prediction

with st.sidebar:
    st.markdown("## 🏥 ReadmitAI")
    st.caption("HOSPITAL ANALYTICS PLATFORM")
    st.divider()
    page=st.radio("NAVIGATION",["Dashboard","Patient Prediction","Compare All Models","Model Analytics","About"])
    st.divider()
    st.caption("Diabetes 130-US Hospitals Dataset")
    st.caption("Machine Learning Mini Project")
    st.caption("Educational prototype")

st.markdown("""
<div class="hero">
<div class="eyebrow">MACHINE LEARNING · HEALTHCARE ANALYTICS</div>
<h1>ReadmitAI</h1>
<p>Hospital Readmission Risk Prediction using Machine Learning</p>
</div>
""",unsafe_allow_html=True)

st.info("Educational research prototype only. Predictions are not medical advice or clinically validated risk assessments.")

if page=="Dashboard":
    st.title("Project Dashboard")
    st.write("A machine learning application that analyses historical diabetic hospital encounters to estimate recorded readmission probability within 30 days.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Hospital Encounters","101,766")
    c2.metric("Dataset Attributes","50")
    c3.metric("ML Models","4")
    c4.metric("30-Day Positive Class","11.16%")
    st.subheader("Project Workflow")
    workflow=[("01","Dataset Collection"),("02","Data Cleaning"),("03","Feature Processing"),("04","Model Training"),("05","Performance Evaluation"),("06","Prediction")]
    cols=st.columns(3)
    for i,(number,title) in enumerate(workflow):
        with cols[i%3]:
            st.markdown(f'<div class="section-card"><span style="color:#38BDF8;font-size:13px;font-weight:700">{number}</span><br><b>{title}</b></div>',unsafe_allow_html=True)
    st.subheader("Trained Machine Learning Models")
    cols=st.columns(4)
    for col,name in zip(cols,models.keys()):
        with col:
            st.markdown(f'<div class="section-card" style="text-align:center"><b>{name}</b></div>',unsafe_allow_html=True)
    if not metrics.empty:
        st.subheader("Model Performance Snapshot")
        st.dataframe(metrics.style.format("{:.4f}"),use_container_width=True)
        best=metrics["PR-AUC"].idxmax()
        st.success(f"Highest PR-AUC in current evaluation: {best} ({metrics.loc[best,'PR-AUC']:.4f})")
    st.subheader("Dataset Information")
    st.write("Dataset: Diabetes 130-US Hospitals for Years 1999–2008")
    st.write("Source: Kaggle mirror of the UCI Machine Learning Repository dataset.")
    st.write("Prediction target: Readmission within 30 days.")

elif page=="Patient Prediction":
    st.title("Patient Readmission Prediction")
    st.write("Enter an illustrative hospital encounter and select a trained model.")
    selected_model=st.selectbox("Select Machine Learning Model",list(models.keys()))
    input_data=prediction_input("single_prediction_form")
    if input_data is not None:
        input_df=prepare_input(input_data)
        try:
            probability,prediction=get_model_probability(selected_model,models[selected_model],input_df)
            st.divider()
            st.subheader("Prediction Result")
            c1,c2=st.columns(2)
            c1.metric("Estimated 30-Day Probability",f"{probability*100:.2f}%")
            c2.metric("Classification Threshold","50%")
            st.progress(probability)
            if prediction==1:
                st.error("Model Prediction: Readmission within 30 days")
            else:
                st.success("Model Prediction: No readmission within 30 days recorded")
            st.caption(f"Selected model: {selected_model}")
            st.download_button("Download Input Summary",input_df.to_csv(index=False),"prediction_input.csv","text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page=="Compare All Models":
    st.title("Compare All Models")
    st.write("Enter one hospital encounter and compare estimated probabilities from all four trained models.")
    input_data=prediction_input("compare_models_form")
    if input_data is not None:
        input_df=prepare_input(input_data)
        comparison=[]
        try:
            for name,model in models.items():
                probability,prediction=get_model_probability(name,model,input_df)
                comparison.append({
                    "Model":name,
                    "Estimated Probability (%)":round(probability*100,2),
                    "Prediction":"Readmitted" if prediction==1 else "No Readmission",
                    "Threshold (%)":50
                })
            comparison_df=pd.DataFrame(comparison)
            st.subheader("Four-Model Prediction Comparison")
            st.dataframe(comparison_df,use_container_width=True,hide_index=True)
            best=comparison_df.loc[comparison_df["Estimated Probability (%)"].idxmax()]
            st.info(f"Highest estimated probability: {best['Model']} — {best['Estimated Probability (%)']:.2f}%")
            st.download_button("Download Model Comparison",comparison_df.to_csv(index=False),"model_comparison.csv","text/csv")
        except Exception as e:
            st.error(f"Model comparison failed: {e}")

elif page=="Model Analytics":
    st.title("Model Performance Analytics")
    st.write("Evaluation results from the complete held-out test dataset.")
    if not metrics.empty:
        st.subheader("Performance Comparison")
        st.dataframe(metrics.style.format("{:.4f}"),use_container_width=True)
        c1,c2=st.columns(2)
        best_pr=metrics["PR-AUC"].idxmax()
        best_recall=metrics["Recall"].idxmax()
        c1.metric("Highest PR-AUC",best_pr,f"{metrics.loc[best_pr,'PR-AUC']:.4f}")
        c2.metric("Highest Recall",best_recall,f"{metrics.loc[best_recall,'Recall']:.4f}")
    else:
        st.warning("Model performance CSV is unavailable. Run Phase 5 evaluation first.")
    charts=[
        ("Model Performance Comparison","model_performance_comparison.png"),
        ("Confusion Matrices","confusion_matrices.png"),
        ("ROC Curve Comparison","roc_curve_comparison.png"),
        ("Precision-Recall Curve Comparison","precision_recall_comparison.png"),
        ("XGBoost Feature Importance","xgboost_feature_importance.png")
    ]
    for title,filename in charts:
        path=FIGURE_DIR/filename
        if path.exists():
            st.subheader(title)
            st.image(str(path),use_container_width=True)
    if metrics.empty:
        st.caption("Run the model evaluation notebook to generate analytics.")

else:
    st.title("About ReadmitAI")
    st.write("ReadmitAI is an academic machine learning mini project focused on predicting recorded 30-day hospital readmission among diabetic patients.")
    st.subheader("Problem Statement")
    st.write("Hospital readmission is an important healthcare analytics problem. This project uses historical encounter data to study patterns associated with readmission within 30 days.")
    st.subheader("Dataset")
    st.write("Diabetes 130-US Hospitals for Years 1999–2008")
    st.write("Source: UCI Machine Learning Repository, accessed through a Kaggle mirror.")
    st.subheader("Methodology")
    for item in ["Dataset collection and exploration","Missing-value analysis and preprocessing","Patient-wise training and testing split","Training four classification models","Performance evaluation using six metrics","Streamlit application development"]:
        st.markdown(f"- {item}")
    st.subheader("Models Used")
    for name in models:
        st.markdown(f"- **{name}**")
    st.subheader("Limitations")
    st.write("The dataset represents historical US hospital encounters from 1999–2008. Its results may not generalise to current hospitals or other populations. The application is an educational prototype, not a clinically validated tool.")
    st.subheader("Disclaimer")
    st.warning("Predictions are model estimates for educational demonstration only. They must not be used for diagnosis, treatment or real patient-care decisions.")

st.divider()
st.caption("ReadmitAI | Hospital Readmission Risk Predictor | Educational Machine Learning Prototype")

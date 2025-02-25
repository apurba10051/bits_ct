import streamlit as st
import joblib
import pandas as pd
import numpy as np
import torch
import re
from datetime import datetime
import warnings
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Medical context validation with enhanced term recognition
class MedicalTermRecognizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # Expanded disease patterns
        self.disease_patterns = re.compile(
            r'\b(heart attack|myocardial infarction|covid(-19)?|coronavirus|diabetes|'
            r'stroke|sepsis|pneumonia|arrhythmia|hypertension|asthma|arthritis|'
            r'leukemia|melanoma|alzheimer\'?s|parkinson\'?s|tuberculosis|'
            r'epilepsy|migraine|osteoporosis)\b|'
            r'\b([A-Z][a-z]+ (disease|syndrome|disorder))\b',
            re.IGNORECASE
        )

        # Broad medical context references
        self.reference_texts = [
            "Clinical trial investigating disease treatment outcomes",
            "Patient study on cardiovascular health interventions",
            "Randomized controlled trial of infectious disease therapies",
            "Phase III study of chronic condition management",
            "Multicenter research protocol for emergency medical treatments"
        ]

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt",
                              padding=True, truncation=True,
                              max_length=256)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def validate_medical_context(self, text, threshold=0.55):
        """Hybrid validation with enhanced medical term recognition"""
        # Direct disease term matching
        if self.disease_patterns.search(text):
            return True, "Recognized medical terms"

        # Semantic similarity check
        text_emb = self.get_embeddings(text)
        ref_embs = self.get_embeddings(self.reference_texts)
        similarity = np.max(cosine_similarity(text_emb, ref_embs))

        if similarity > threshold:
            return True, f"Medical context detected (confidence: {similarity:.2f})"
        return False, "No clear medical context identified"

# Load prediction pipeline
try:
    pipeline = joblib.load('clinical_trial_predictor.pkl')
    model = pipeline['model']
    clinical_bert_tokenizer = pipeline['clinical_bert_tokenizer']
    clinical_bert_model = pipeline['clinical_bert_model']
    pca = pipeline['pca']
    label_encoders = pipeline['label_encoders']
    target_encoder = pipeline['target_encoder']
    feature_names = pipeline['feature_names']
    scaler = pipeline['scaler']
    imputer = pipeline['imputer']
except Exception as e:
    st.error(f"Pipeline loading failed: {str(e)}")
    st.stop()

# Initialize components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clinical_bert_model = clinical_bert_model.to(device)
validator = MedicalTermRecognizer()

# Streamlit UI
st.title('Clinical Trial Status Predictor')

with st.form("trial_form"):
    # Input fields
    brief_title = st.text_input("Brief Title*")
    description = st.text_area("Study Description*")
    keywords = st.text_input("Keywords (comma-separated)*")
    inclusion = st.text_area("Inclusion Criteria Summary*")
    country = st.text_input("Country")

    start_date = st.date_input("Start Date*")
    end_date = st.date_input("End Date*")

    enrollment = st.number_input("Enrollment*", min_value=0)

    study_type = st.selectbox("Study Type*",
                            options=label_encoders['Study Type'].classes_)
    phases = st.selectbox("Phase*",
                        options=label_encoders['Phases'].classes_)

    submitted = st.form_submit_button("Predict")
    st.markdown("*Required fields")

if submitted:
    try:
        # Validate medical context
        validation_results = {
            "Title": brief_title,
            "Description": description,
            "Inclusion Criteria": inclusion
        }

        for field, text in validation_results.items():
            is_valid, message = validator.validate_medical_context(text)
            if not is_valid:
                matches = validator.disease_patterns.findall(text)
                detected_terms = ", ".join({m[0].lower() for m in matches if m[0]})
                raise ValueError(
                    f"{field} validation failed: {message}\n"
                    f"Detected terms: {detected_terms or 'None'}"
                )

        # Feature engineering
        duration_days = (end_date - start_date).days
        combined_text = f"{brief_title} {description} {keywords} {inclusion} {country}"

        # Generate embeddings
        inputs = clinical_bert_tokenizer(
            combined_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = clinical_bert_model(**inputs)

        text_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()
        text_reduced = pca.transform(text_embeddings)

        # Prepare features
        feature_values = [
            duration_days,
            enrollment,
            label_encoders['Study Type'].transform([study_type])[0],
            label_encoders['Phases'].transform([phases])[0],
            *text_reduced[0].tolist()
        ]

        features = pd.DataFrame([feature_values], columns=feature_names)
        features = pd.DataFrame(imputer.transform(features), columns=features.columns)
        features[['Enrollment', 'duration_days']] = scaler.transform(
            features[['Enrollment', 'duration_days']]
        )

        # Prediction
        proba = model.predict_proba(features)
        prediction = model.predict(features)
        confidence = np.max(proba)
        status = target_encoder.inverse_transform(prediction)[0]

        # Display results
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Status", status)

        with col2:
            # Determine color based on confidence
            if confidence > 0.7:
                color = "green"
            elif confidence > 0.5:
                color = "orange"
            else:
                color = "red"

            # Create custom HTML for colored confidence display
            confidence_html = f"""
                <div style="color: {color}; font-size: 2rem; font-weight: bold;">
                    {confidence:.1%}
                </div>
                <div style="color: gray; font-size: 0.8rem;">
                    Confidence Score
                </div>
            """
            st.markdown(confidence_html, unsafe_allow_html=True)

        if confidence < 0.6:
            st.warning("Low confidence prediction - verify input quality")

    except Exception as e:
        st.error(str(e))
        st.stop()

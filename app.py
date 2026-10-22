
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("Diabetes Prediction System")

uploaded_file = st.file_uploader(
    "Upload Diabetes CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    if "Outcome" not in df.columns:
        st.error("CSV file mein Outcome column hona chahiye.")

    else:
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        if X.isnull().values.any() or y.isnull().any():
            st.error("Dataset mein missing values hain. Pehle unhe handle karein.")

        elif y.nunique() < 2:
            st.error("Outcome column mein 0 aur 1 dono classes honi chahiye.")

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            st.metric(
                "Model Accuracy",
                f"{accuracy * 100:.2f}%"
            )

            with st.expander("Confusion Matrix"):
                st.write(confusion_matrix(y_test, y_pred))

            st.subheader("Enter Patient Details")

            with st.form("prediction_form"):

                patient_data = {}

                for column in X.columns:
                    patient_data[column] = st.number_input(
                        column,
                        value=0.0,
                        min_value=0.0
                    )

                submit = st.form_submit_button("Predict Diabetes")

            if submit:

                patient = pd.DataFrame(
                    [patient_data],
                    columns=X.columns
                )

                patient_scaled = scaler.transform(patient)

                prediction = model.predict(patient_scaled)

                st.subheader("Prediction Result")

                if prediction[0] == 1:
                    st.warning(
                        "Model predicts a higher likelihood of diabetes."
                    )
                else:
                    st.success(
                        "Model predicts a lower likelihood of diabetes."
                    )

                st.info(
                    "This is an educational prediction, not a medical diagnosis."
                )

else:
    st.info("Please upload your diabetes CSV file to start.")

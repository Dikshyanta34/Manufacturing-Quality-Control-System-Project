import streamlit as st
import pandas as pd
import numpy as np

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def main():
    st.set_page_config(page_title="Manufacturing Quality Control System", layout="wide")

    st.sidebar.title("⚙️ Settings")
    st.sidebar.subheader("Step 1: Upload and Train")

    uploaded_file = st.sidebar.file_uploader("📤 Upload CSV file", type=["csv"])

    allowed_targets = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Data Uploaded Successfully!")

        # Filter only columns that match allowed targets
        target_options = [col for col in data.columns if col in allowed_targets]

        if target_options:
            target_column = st.sidebar.selectbox("🎯 Select Target Column", target_options)

            if st.sidebar.button("Train XGBoost Model 🏋️‍♂️"):
                with st.spinner("🔄 Training... please wait!"):
                    # Preprocess
                    data = data.dropna(subset=[target_column])
                    X = data.drop(columns=[target_column])
                    y = data[target_column]

                    le = LabelEncoder()
                    y = le.fit_transform(y)

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    num_classes = len(np.unique(y))

                    if num_classes == 2:
                        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    else:
                        model = XGBClassifier(
                            use_label_encoder=False,
                            eval_metric='mlogloss',
                            objective='multi:softprob',
                            num_class=num_classes
                        )

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    st.sidebar.success(f"✅ Model Trained! Accuracy: {acc:.2f}")

                    st.session_state.model = model
                    st.session_state.le = le
                    st.session_state.X_columns = X.columns
        else:
            st.sidebar.error("🚫 No valid target columns found! Please upload correct data.")

    # ---- Main UI
    st.title("🏭 Manufacturing Quality Control System")
    st.markdown("---")

    if 'model' in st.session_state:
        st.subheader("🔧 Enter Product Features")

        input_features = {}
        cols = st.columns(2)  # split into two columns

        for idx, feature in enumerate(st.session_state.X_columns):
            with cols[idx % 2]:
                input_features[feature] = st.number_input(f"➤ {feature}", value=0.0)

        input_df = pd.DataFrame([input_features])

        st.markdown("---")

        if st.button("🔮 Predict Defect Status"):
            model = st.session_state.model
            le = st.session_state.le

            prediction = model.predict(input_df)
            predicted_label = le.inverse_transform(prediction)[0]

            st.markdown("---")
            st.subheader("📝 Prediction Result")

            if str(predicted_label).lower() in ['defect', 'defective', 'yes', '1']:
                st.error("🚨 Oh no! Your item is **DEFECTIVE**! Please check the manufacturing process. 😞")
            else:
                st.success(f"✅ Prediction: **{predicted_label}** - Your item is **NOT defective**. 🎉")
                st.balloons()

    else:
        st.info("👈 Please upload your data and train a model first!")

if __name__ == "__main__":
    main()

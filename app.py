import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from disease_info import DISEASE_INFO
from utlis import preprocess_image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI-Powered Crop Health Monitoring",
    page_icon="🌱",
    layout="wide"
)

# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = [
    "Banded Chlorosis",
    "Brown Spot",
    "BrownRust",
    "Grassy shoot",
    "Pokkah Boeng",
    "Sett Rot",
    "Viral Disease",
    "Yellow Leaf",
    "smut"
]

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "best_model.keras"
    )
    return model

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌱 Project Information")

st.sidebar.info("""
### AI-Powered Crop Health Monitoring System For Disease

**About Project:**
"This project uses Artificial Intelligence and Deep Learning to detect diseases in sugarcane leaves. Farmers can upload an image of a sugarcane leaf and receive instant disease predictions, helping improve crop health and productivity."

**Model:** MobileNetV3Small  

**Classes:** 9

**Accuracy:** 88.24%

**Technology:**
- TensorFlow
- Streamlit
- Deep Learning
- MobileNetV3Small
""")

st.sidebar.success(
    "Final Year Project/"

    "Project Team : Annesha Panda / Suman Das/ Mritunjoy Paul"
)

# =====================================================
# MAIN TITLE
# =====================================================

st.title(
    "🌿 AI-Powered Crop Health Monitoring System For Disease"
)

st.write(
    "Upload one or more sugarcane leaf images to detect diseases."
)

# =====================================================
# MULTIPLE IMAGE UPLOAD
# =====================================================

uploaded_files = st.file_uploader(
    "Upload Sugarcane Leaf Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_files:

    results = []

    for uploaded_file in uploaded_files:

        st.divider()

        image, rgb_input, _ = preprocess_image(
            uploaded_file
        )

        col1, col2 = st.columns(2)

        # =============================================
        # IMAGE
        # =============================================

        with col1:

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

        # =============================================
        # PREDICTION
        # =============================================

        with st.spinner(
            f"Predicting {uploaded_file.name}..."
        ):

            prediction = model.predict(
                rgb_input,
                verbose=0
            )

            probs = prediction[0]

            pred_index = np.argmax(probs)

            disease = CLASS_NAMES[pred_index]

            confidence = probs[pred_index] * 100

            top3_indices = np.argsort(
                probs
            )[-3:][::-1]

        results.append({
            "Image": uploaded_file.name,
            "Disease": disease,
            "Confidence (%)": round(
                confidence,
                2
            )
        })

        # =============================================
        # RESULTS
        # =============================================

        with col2:

            st.success(
                f"Prediction: {disease}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

            st.subheader(
                "Top 3 Predictions"
            )

            for idx in top3_indices:

                st.write(
                    f"{CLASS_NAMES[idx]} : "
                    f"{probs[idx]*100:.2f}%"
                )

            # =========================================
            # DISEASE INFORMATION
            # =========================================

            if disease in DISEASE_INFO:

                with st.expander(
                    "Disease Description"
                ):

                    st.write(
                        DISEASE_INFO[disease][
                            "description"
                        ]
                    )

                with st.expander(
                    "Recommended Treatment"
                ):

                    st.write(
                        DISEASE_INFO[disease][
                            "treatment"
                        ]
                    )

        # =============================================
        # PROBABILITY GRAPH
        # =============================================

        st.subheader(
            f"Prediction Probabilities - {uploaded_file.name}"
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.barh(
            CLASS_NAMES,
            probs * 100
        )

        ax.set_xlabel(
            "Probability (%)"
        )

        ax.set_title(
            "Prediction Confidence"
        )

        st.pyplot(fig)

    # =================================================
    # SUMMARY TABLE
    # =================================================

    st.divider()

    st.header(
        "📊 Batch Prediction Summary"
    )

    df = pd.DataFrame(
        results
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    # =================================================
    # DISEASE DISTRIBUTION
    # =================================================

    st.subheader(
        "Disease Distribution"
    )

    disease_counts = (
        df["Disease"]
        .value_counts()
    )

    st.bar_chart(
        disease_counts
    )

    # =================================================
    # CSV DOWNLOAD
    # =================================================

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Upload one or more sugarcane leaf images to begin prediction."
    )
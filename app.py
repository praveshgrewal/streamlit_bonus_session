import os
import joblib
import pandas as pd
import streamlit as st


# Needed for unpickling the pipeline created in the notebook.
# The FunctionTransformer inside the pipeline references add_features
# from __main__, so we define it here too.
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "sibsp" in X.columns and "parch" in X.columns:
        X["family_size"] = X["sibsp"] + X["parch"] + 1
        X["is_alone"] = (X["family_size"] == 1).astype(int)
    return X


st.set_page_config(
    page_title="Titanic Survival Studio",
    page_icon="T",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Work+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Work Sans', sans-serif;
    }

    .app-shell {
        background: radial-gradient(1200px 800px at 10% 10%, #f7f2ea 0%, #eef2f6 45%, #e4edf6 100%);
        padding: 1.6rem 2rem 2rem 2rem;
        border-radius: 20px;
        border: 1px solid rgba(10, 20, 40, 0.06);
        box-shadow: 0 14px 40px rgba(10, 20, 40, 0.08);
    }

    .hero {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(10, 20, 40, 0.08);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: 2.7rem;
        font-weight: 700;
        color: #1b2a3a;
        letter-spacing: 0.3px;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #4a5a6a;
        margin-top: 0.2rem;
    }

    .pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        background: #edf2f7;
        color: #2a3b4f;
        border: 1px solid rgba(10, 20, 40, 0.08);
        margin-right: 0.4rem;
        margin-bottom: 0.2rem;
    }

    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(10, 20, 40, 0.08);
        box-shadow: 0 10px 26px rgba(10, 20, 40, 0.06);
    }

    .card-title {
        font-family: 'Fraunces', serif;
        font-size: 1.2rem;
        margin: 0 0 0.6rem 0;
        color: #1b2a3a;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #5b6b7a;
    }

    .prob {
        font-family: 'Fraunces', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #17324d;
        margin: 0.1rem 0 0.8rem 0;
    }

    .note {
        font-size: 0.9rem;
        color: #6b7b8a;
    }

    .status {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-good {
        background: #e7f6ee;
        color: #1f5f3a;
        border: 1px solid rgba(31, 95, 58, 0.2);
    }

    .status-bad {
        background: #fdecec;
        color: #8a2c2c;
        border: 1px solid rgba(138, 44, 44, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


model = load_model("titanic_model.pkl")

with st.container():
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="hero-title">Titanic Survival Studio</div>
                <div class="hero-subtitle">Design a passenger profile and see the model's survival estimate.</div>
            </div>
            <div>
                <span class="pill">Model: Random Forest</span>
                <span class="pill">Dataset: Seaborn Titanic</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("Model file `titanic_model.pkl` not found. Please run the notebook to train and save the model.")
        st.stop()

    preset_options = [
        "Custom",
        "Third-class young adult",
        "First-class wealthy adult",
        "Child traveling with family",
    ]

    preset = st.selectbox("Quick Preset", preset_options, index=0, key="preset")

    preset_values = {
        "Third-class young adult": {
            "pclass": 3,
            "sex": "male",
            "age": 25.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 8.0,
            "embarked": "S",
            "tclass": "Third",
            "deck": None,
            "embark_town": "Southampton",
            "alone": True,
        },
        "First-class wealthy adult": {
            "pclass": 1,
            "sex": "female",
            "age": 34.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 120.0,
            "embarked": "C",
            "tclass": "First",
            "deck": "B",
            "embark_town": "Cherbourg",
            "alone": True,
        },
        "Child traveling with family": {
            "pclass": 2,
            "sex": "female",
            "age": 8.0,
            "sibsp": 1,
            "parch": 1,
            "fare": 18.0,
            "embarked": "S",
            "tclass": "Second",
            "deck": None,
            "embark_town": "Southampton",
            "alone": False,
        },
    }

    if preset != st.session_state.get("active_preset", "Custom"):
        if preset in preset_values:
            for key, value in preset_values[preset].items():
                st.session_state[key] = value
        st.session_state["active_preset"] = preset

    left, right = st.columns([1.1, 1])

    def selectbox_with_state(label, options, key, default):
        current = st.session_state.get(key, default)
        index = options.index(current) if current in options else 0
        return st.selectbox(label, options, index=index, key=key)

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Passenger Profile</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            pclass = selectbox_with_state("Ticket Class", [1, 2, 3], "pclass", 3)
            sex = selectbox_with_state("Sex", ["male", "female"], "sex", "male")
            age = st.slider(
                "Age",
                min_value=0.0,
                max_value=80.0,
                value=float(st.session_state.get("age", 29.0)),
                step=0.5,
                key="age",
            )
            sibsp = st.number_input(
                "Siblings/Spouses",
                min_value=0,
                max_value=8,
                value=int(st.session_state.get("sibsp", 0)),
                step=1,
                key="sibsp",
            )
            parch = st.number_input(
                "Parents/Children",
                min_value=0,
                max_value=6,
                value=int(st.session_state.get("parch", 0)),
                step=1,
                key="parch",
            )

        with col2:
            fare = st.number_input(
                "Fare",
                min_value=0.0,
                max_value=600.0,
                value=float(st.session_state.get("fare", 32.0)),
                step=1.0,
                key="fare",
            )
            embarked = selectbox_with_state(
                "Embarked",
                ["S", "C", "Q", "Unknown"],
                "embarked",
                "S",
            )
            tclass = selectbox_with_state(
                "Class",
                ["First", "Second", "Third"],
                "tclass",
                "Third",
            )
            deck = selectbox_with_state(
                "Deck",
                ["A", "B", "C", "D", "E", "F", "G", "Unknown"],
                "deck",
                "Unknown",
            )
            embark_town = selectbox_with_state(
                "Embark Town",
                ["Southampton", "Cherbourg", "Queenstown", "Unknown"],
                "embark_town",
                "Southampton",
            )

        alone = st.checkbox("Traveling Alone", value=bool(st.session_state.get("alone", True)), key="alone")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top: 1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Profile Snapshot</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <span class="pill">Class: {tclass}</span>
            <span class="pill">Embarked: {embarked}</span>
            <span class="pill">Deck: {deck}</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Prediction</div>", unsafe_allow_html=True)

        input_df = pd.DataFrame(
            [
                {
                    "pclass": pclass,
                    "sex": sex,
                    "age": age,
                    "sibsp": sibsp,
                    "parch": parch,
                    "fare": fare,
                    "embarked": None if embarked == "Unknown" else embarked,
                    "class": tclass,
                    "deck": None if deck == "Unknown" else deck,
                    "embark_town": None if embark_town == "Unknown" else embark_town,
                    "alone": alone,
                }
            ]
        )

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.markdown("<div class='metric-label'>Survival Probability</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prob'>{proba*100:.1f}%</div>", unsafe_allow_html=True)
        st.progress(min(max(proba, 0.0), 1.0))

        outcome = "Likely Survived" if pred == 1 else "Likely Not Survived"
        status_class = "status-good" if pred == 1 else "status-bad"
        st.markdown(
            f"<span class='status {status_class}'>{outcome}</span>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='note' style='margin-top: 0.8rem;'>This is a demo model trained on the Seaborn Titanic dataset. Results are probabilistic, not deterministic.</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Show Input Data"):
            st.dataframe(input_df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

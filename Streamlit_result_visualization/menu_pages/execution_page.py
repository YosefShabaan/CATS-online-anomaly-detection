from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from keras import layers, models
from streamlit_option_menu import option_menu
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

SENSOR_COLUMNS = [
    "aimp",
    "amud",
    "arnd",
    "asin1",
    "asin2",
    "adbr",
    "adfl",
    "bed1",
    "bed2",
    "bfo1",
    "bfo2",
    "bso1",
    "bso2",
    "bso3",
    "ced1",
    "cfo1",
    "cso1",
]

PLOT_GROUPS = [
    ("AIMP", ["aimp"]),
    ("AMUD and ARND", ["amud", "arnd"]),
    ("ASIN1 and ASIN2", ["asin1", "asin2"]),
    ("ADBR and ADFL", ["adbr", "adfl"]),
    ("BED1 and BED2", ["bed1", "bed2"]),
    ("BFO1 and BFO2", ["bfo1", "bfo2"]),
    ("BSO1, BSO2 and BSO3", ["bso1", "bso2", "bso3"]),
    ("CED1, CFO1 and CSO1", ["ced1", "cfo1", "cso1"]),
]

MODEL_CONFIGS = {
    ("Model 1", "Adam", "30", "90"): ("m1_adam_e30.h5", 0.712901719027131),
    ("Model 1", "Adam", "30", "95"): ("m1_adam_e30.h5", 0.8540586741485884),
    ("Model 1", "Adam", "60", "90"): ("m1_adam_e60.h5", 0.7195667774220919),
    ("Model 1", "Adam", "60", "95"): ("m1_adam_e60.h5", 0.8690659052845591),
    ("Model 1", "RMSprop", "30", "90"): ("m1_rms_e30.h5", 0.738507739374793),
    ("Model 1", "RMSprop", "30", "95"): ("m1_rms_e30.h5", 0.880710482063847),
    ("Model 1", "RMSprop", "60", "90"): ("m1_rms_e60.h5", 0.729910393617794),
    ("Model 1", "RMSprop", "60", "95"): ("m1_rms_e60.h5", 0.8784755999776444),
    ("Model 2", "Adam", "30", "90"): ("m2_adam_e30.h5", 0.22526050543400736),
    ("Model 2", "Adam", "30", "95"): ("m2_adam_e30.h5", 0.37730381179387495),
    ("Model 2", "Adam", "60", "90"): ("m2_adam_e60.h5", 0.22454969771467106),
    ("Model 2", "Adam", "60", "95"): ("m2_adam_e60.h5", 0.37595087812035954),
    ("Model 2", "RMSprop", "30", "90"): ("m2_rms_e30.h5", 0.22683427400617032),
    ("Model 2", "RMSprop", "30", "95"): ("m2_rms_e30.h5", 0.37864416533085626),
    ("Model 2", "RMSprop", "60", "90"): ("m2_rms_e60.h5", 0.2267485741862487),
    ("Model 2", "RMSprop", "60", "95"): ("m2_rms_e60.h5", 0.38110223655544756),
}


def build_autoencoder(model_type):
    if model_type == "Model 1":
        model = models.Sequential()
        model.add(layers.Input(shape=(17,)))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(8, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())

        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(17, activation="sigmoid"))
        return model

    if model_type == "Model 2":
        model = models.Sequential()
        model.add(keras.Input(shape=(17,)))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(16, activation="relu"))

        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(17, activation="tanh"))
        return model

    raise ValueError(f"Unsupported model type: {model_type}")


def model_build(model_type, optimizer, epoch, percentile):
    config_key = (model_type, optimizer, epoch, percentile)
    if config_key not in MODEL_CONFIGS:
        raise ValueError("The selected model configuration is not available.")

    weights_name, threshold = MODEL_CONFIGS[config_key]
    weights_path = MODELS_DIR / weights_name

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights were not found: {weights_path}")

    autoencoder = build_autoencoder(model_type)
    autoencoder.load_weights(weights_path)
    return autoencoder, threshold


def read_sensor_csv(uploaded_file):
    """Read and validate an uploaded CATS sensor CSV without crashing the page."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read the CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    df.columns = [str(column).strip() for column in df.columns]

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    else:
        first_column = df.columns[0]
        timestamps = pd.to_datetime(df[first_column], errors="coerce")

        if timestamps.isna().any():
            raise ValueError(
                "A valid 'timestamp' column was not found. "
                "The CSV must contain a timestamp column."
            )

        df = df.drop(columns=[first_column])

    if timestamps.isna().any():
        invalid_count = int(timestamps.isna().sum())
        raise ValueError(f"{invalid_count} timestamp value(s) could not be parsed.")

    df.index = pd.DatetimeIndex(timestamps, name="timestamp")
    df = df.sort_index()

    required_columns = SENSOR_COLUMNS + ["y"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("Missing required column(s): " + ", ".join(missing_columns))

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_columns = [
        column for column in required_columns if df[column].isna().any()
    ]
    if invalid_columns:
        raise ValueError(
            "These columns contain missing or non-numeric values: "
            + ", ".join(invalid_columns)
        )

    if len(df) < 2:
        raise ValueError("The CSV must contain at least two rows.")

    return df[required_columns]


def get_slider_step(index):
    differences = index.to_series().diff().dropna()
    positive_differences = differences[differences > pd.Timedelta(0)]

    if positive_differences.empty:
        return pd.Timedelta(seconds=1).to_pytimedelta()

    return positive_differences.median().to_pytimedelta()


def execution_page():
    st.title("Execution of the selected model")

    st.sidebar.title("Model Selection:")
    with st.sidebar:
        model_type = option_menu(
            menu_title="Model",
            options=["Model 1", "Model 2"],
            icons=["", "star-fill"],
            menu_icon="pc-display",
            default_index=0,
        )

    st.sidebar.title("Parameter Selection:")
    with st.sidebar:
        optimizer = option_menu(
            menu_title="Optimizer",
            options=["Adam", "RMSprop"],
            icons=["", "star-fill"],
            menu_icon="speedometer",
            default_index=0,
        )

    with st.sidebar:
        epoch = option_menu(
            menu_title="Epoch",
            options=["30", "60"],
            icons=["star-fill"],
            menu_icon="speedometer",
            default_index=0,
        )

    with st.sidebar:
        percentile = option_menu(
            menu_title="Percentile",
            options=["90", "95"],
            icons=["star-fill"],
            menu_icon="speedometer",
            default_index=0,
        )

    uploaded_file = st.file_uploader(
        "Upload CSV file of sensor data",
        type=["csv"],
        key="sensor_csv_uploader",
    )

    if uploaded_file is None:
        return

    try:
        df = read_sensor_csv(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success(f"Loaded {len(df):,} rows from {uploaded_file.name}.")

    start_data = df.index.min().to_pydatetime()
    end_data = df.index.max().to_pydatetime()
    slider_step = get_slider_step(df.index)

    selected_range = st.slider(
        "Select range to analyze",
        min_value=start_data,
        max_value=end_data,
        value=(start_data, end_data),
        step=slider_step,
        format="YY/MM/DD - HH:mm:ss",
    )

    df_selected = df.loc[selected_range[0] : selected_range[1]]
    feature_df = df_selected[SENSOR_COLUMNS]

    if feature_df.empty:
        st.warning("The selected time range contains no rows.")
        return

    st.caption(
        f"Selected rows: {len(feature_df):,}. "
        "The table preview shows the first 1,000 rows to keep the page responsive."
    )
    st.dataframe(feature_df.head(1000))

    if not st.button("Show anomalies"):
        return

    progress_bar = st.progress(0, text="Creating model...")

    try:
        autoencoder, threshold = model_build(
            model_type,
            optimizer,
            epoch,
            percentile,
        )

        progress_bar.progress(10, text="Predicting...")

        inputs = feature_df.to_numpy(dtype=np.float32)
        predictions = autoencoder.predict(inputs, batch_size=1024, verbose=0)
        mse = pd.Series(
            np.mean(np.square(inputs - predictions), axis=1),
            index=feature_df.index,
            name="mse",
        )
        binary_predictions = mse > threshold
        anomaly_index = binary_predictions[binary_predictions].index
        anomaly_df = df_selected.loc[anomaly_index]

        progress_bar.progress(20, text="Generating results...")

        num_anomalies = int(binary_predictions.sum())
        if num_anomalies > 0:
            st.markdown(
                f"<span style='color:red'>Found {num_anomalies} anomalies.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:green'>No anomalies found.</span>",
                unsafe_allow_html=True,
            )

        for position, (title, columns) in enumerate(PLOT_GROUPS, start=1):
            progress = 20 + int((position / len(PLOT_GROUPS)) * 75)
            progress_bar.progress(progress, text=f"Creating {title} plots...")

            fig = px.line(
                feature_df,
                x=feature_df.index,
                y=columns,
                title=f"Sensor anomalies: {', '.join(columns)}",
            )

            for column in columns:
                if not anomaly_df.empty:
                    anomaly_trace = px.scatter(
                        anomaly_df,
                        x=anomaly_df.index,
                        y=column,
                        color_discrete_sequence=["red"],
                    ).data[0]
                    anomaly_trace.name = f"{column} anomaly"
                    fig.add_trace(anomaly_trace)

            st.plotly_chart(fig, use_container_width=True)

        progress_bar.progress(100, text="Done.")
        progress_bar.empty()

    except Exception as exc:
        progress_bar.empty()
        st.error(f"Analysis failed: {exc}")

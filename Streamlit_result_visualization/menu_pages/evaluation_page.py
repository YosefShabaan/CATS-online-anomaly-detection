from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_INFO_DIR = BASE_DIR / "models_info"


def load_evaluation_results(run_dir):
    """Load the saved training metrics and plot paths for one experiment."""
    files = {
        "training": run_dir / "training_results.csv",
        "confusion_matrix": run_dir / "confusion_matrix.png",
        "metrics": run_dir / "evaluation_metrics.csv",
        "roc_curve": run_dir / "roc_curve.png",
    }

    missing_files = [str(path) for path in files.values() if not path.exists()]
    if missing_files:
        st.error("Some evaluation files could not be found:")
        for path in missing_files:
            st.code(path)
        st.stop()

    return (
        pd.read_csv(files["training"]),
        str(files["confusion_matrix"]),
        pd.read_csv(files["metrics"]),
        str(files["roc_curve"]),
    )


# Function for the evaluation page
def evaluation_page():
    with st.sidebar:
        st.title("Select parameter to compare")
        evaluated = option_menu(
            menu_title="Evaluate",
            options=["Architecture", "Optimizer", "Percentile"],
            icons=[],
            menu_icon="pc-display",
            default_index=0,
        )

    # Comparison of Architectures
    match evaluated:
        case "Architecture":
            st.title("Model 1 vs Model 2")

            with st.sidebar:
                st.title("Select parameters to use")
                optimizer = option_menu(
                    menu_title="Optimizer",
                    options=["Adam", "RMSprop"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            with st.sidebar:
                epochs = option_menu(
                    menu_title="Epochs",
                    options=["30", "60"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            with st.sidebar:
                threshold = option_menu(
                    menu_title="Percentile",
                    options=["90", "95"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            st.write("Selected parameters:")
            st.write("- Optimizer: " + str(optimizer))
            st.write("- Epochs: " + str(epochs))
            st.write("- Percentile: " + str(threshold))

            run_name = f"{optimizer}_e{epochs}_p{threshold}"
            file_path1 = MODELS_INFO_DIR / "Model1" / run_name
            file_path2 = MODELS_INFO_DIR / "Model2" / run_name

            df1, img1, test1, roc1 = load_evaluation_results(file_path1)
            df2, img2, test2, roc2 = load_evaluation_results(file_path2)

            df1["Model"] = "Model 1"
            df2["Model"] = "Model 2"
            combined_df = pd.concat([df1, df2])

            for value in df1.columns:
                if value not in ["Epoch", "Model"]:
                    fig = px.line(
                        combined_df,
                        x="Epoch",
                        y=value,
                        color="Model",
                        title=str(value),
                    )
                    st.plotly_chart(fig)

            st.title("Confusion Matrix & Test Result")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Model 1:")
                st.image(img1)
                st.write(test1)
                st.image(roc1)

            with col2:
                st.write("Model 2:")
                st.image(img2)
                st.write(test2)
                st.image(roc2)

        case "Optimizer":
            st.title("Adam vs RMSprop")

            with st.sidebar:
                st.title("Select parameters to use")
                model = option_menu(
                    menu_title="Model",
                    options=["Model1", "Model2"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            with st.sidebar:
                epochs = option_menu(
                    menu_title="Epochs",
                    options=["30", "60"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            with st.sidebar:
                threshold = option_menu(
                    menu_title="Percentile",
                    options=["90", "95"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            st.write("Selected parameters:")
            st.write("- Model: " + str(model))
            st.write("- Epochs: " + str(epochs))
            st.write("- Percentile: " + str(threshold))

            file_path1 = MODELS_INFO_DIR / model / f"Adam_e{epochs}_p{threshold}"
            file_path2 = MODELS_INFO_DIR / model / f"RMSprop_e{epochs}_p{threshold}"

            df1, img1, test1, roc1 = load_evaluation_results(file_path1)
            df2, img2, test2, roc2 = load_evaluation_results(file_path2)

            df1["Optimizer"] = "Adam"
            df2["Optimizer"] = "RMSprop"
            combined_df = pd.concat([df1, df2])

            for value in df1.columns:
                if value not in ["Epoch", "Optimizer"]:
                    fig = px.line(
                        combined_df,
                        x="Epoch",
                        y=value,
                        color="Optimizer",
                        title=str(value),
                    )
                    st.plotly_chart(fig)

            st.title("Confusion Matrix & Test Result")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Adam:")
                st.image(img1)
                st.write(test1)
                st.image(roc1)

            with col2:
                st.write("RMSprop:")
                st.image(img2)
                st.write(test2)
                st.image(roc2)

        case "Percentile":
            st.title("Threshold 90% vs Threshold 95%")

            with st.sidebar:
                st.title("Select parameters to use")
                model = option_menu(
                    menu_title="Model",
                    options=["Model1", "Model2"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            with st.sidebar:
                optimizer = option_menu(
                    menu_title="Optimizer",
                    options=["Adam", "RMSprop"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            with st.sidebar:
                epochs = option_menu(
                    menu_title="Epochs",
                    options=["30", "60"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            st.write("Selected parameters:")
            st.write("- Model: " + str(model))
            st.write("- Optimizer: " + str(optimizer))
            st.write("- Epochs: " + str(epochs))

            file_path1 = MODELS_INFO_DIR / model / f"{optimizer}_e{epochs}_p90"
            file_path2 = MODELS_INFO_DIR / model / f"{optimizer}_e{epochs}_p95"

            df1, img1, test1, roc1 = load_evaluation_results(file_path1)
            df2, img2, test2, roc2 = load_evaluation_results(file_path2)

            df1["Percentile"] = "Threshold 90%"
            df2["Percentile"] = "Threshold 95%"
            combined_df = pd.concat([df1, df2])

            for value in df1.columns:
                if value not in ["Epoch", "Percentile"]:
                    fig = px.line(
                        combined_df,
                        x="Epoch",
                        y=value,
                        color="Percentile",
                        title=str(value),
                    )
                    st.plotly_chart(fig)

            st.title("Confusion Matrix & Test Result")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Threshold 90%:")
                st.image(img1)
                st.write(test1)
                st.image(roc1)

            with col2:
                st.write("Threshold 95%:")
                st.image(img2)
                st.write(test2)
                st.image(roc2)

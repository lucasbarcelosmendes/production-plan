import tempfile
from pathlib import Path

import streamlit as st

from models.main import main as run_main

st.title("Production Plan Calculator")

st.write(
    "Upload an `SPP.xlsx` file or use the bundled sample to run the production plan "
    "calculation. The results will be written to `SPP_results.xlsx`."
)

uploaded = st.file_uploader("SPP.xlsx", type=["xlsx"])

if st.button("Run calculation"):
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded.getbuffer())
            input_path = Path(tmp.name)
    else:
        input_path = Path("data/input/SPP.xlsx")
    output_dir = Path("web_outputs")
    run_main(spp_path=input_path, output_dir=output_dir)
    result_file = output_dir / "SPP_results.xlsx"
    if result_file.exists():
        st.success("Calculation complete.")
        with open(result_file, "rb") as fh:
            st.download_button(
                label="Download results", data=fh, file_name="SPP_results.xlsx"
            )
    else:
        st.error("No results produced.")

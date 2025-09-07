from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, render_template, request, send_file

from models.main import main as run_main

app = Flask(__name__)


@app.get("/")
def index():
    """Render file upload form."""
    return render_template("index.html")


@app.post("/run")
def run_calculation():
    """Accept an uploaded SPP.xlsx, run the model, and return results."""
    uploaded = request.files.get("spp_file")
    if uploaded is None or uploaded.filename == "":
        return ("No file uploaded", 400)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "SPP.xlsx"
        uploaded.save(input_path)
        output_dir = tmpdir_path
        run_main(spp_path=input_path, output_dir=output_dir)
        result_file = output_dir / "SPP_results.xlsx"
        if not result_file.exists():
            return ("No results produced", 500)
        return send_file(result_file, as_attachment=True, download_name="SPP_results.xlsx")


if __name__ == "__main__":
    app.run()

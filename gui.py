import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd

from main import run_transformations


class App(tk.Tk):
    """Simple GUI wrapper for ``run_transformations``."""

    def __init__(self):
        super().__init__()
        self.title("Production Plan")

        # ---- Input Excel ----
        tk.Label(self, text="Input Excel:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.input_var = tk.StringVar(value="SPP.xlsx")
        tk.Entry(self, textvariable=self.input_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        # ---- Output Excel ----
        tk.Label(self, text="Output Excel:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        default_out = Path("outputs") / "SPP_results.xlsx"
        self.output_var = tk.StringVar(value=str(default_out))
        tk.Entry(self, textvariable=self.output_var, width=40).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        # ---- Run Button ----
        tk.Button(self, text="Run", command=self.run).grid(row=2, column=0, columnspan=3, pady=10)

        # ---- Status Label ----
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(self, textvariable=self.status_var).grid(row=3, column=0, columnspan=3, pady=(0, 5))

    def browse_input(self) -> None:
        """Browse for the input Excel workbook."""
        path = filedialog.askopenfilename(
            title="Select input Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if path:
            self.input_var.set(path)

    def browse_output(self) -> None:
        """Browse for the output Excel location."""
        path = filedialog.asksaveasfilename(
            title="Select output Excel file",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
            initialfile=Path(self.output_var.get()).name,
        )
        if path:
            self.output_var.set(path)

    def run(self) -> None:
        """Assemble config and run the transformations."""
        input_path = self.input_var.get().strip()
        output_path = self.output_var.get().strip()
        try:
            sheets = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")
            config = {"output_xlsx": output_path, "sheets": sheets}
            run_transformations(config)
        except Exception as exc:  # pragma: no cover - GUI feedback only
            messagebox.showerror("Error", str(exc))
            self.status_var.set("Error")
        else:  # pragma: no cover - GUI feedback only
            messagebox.showinfo("Success", f"Output saved to {output_path}")
            self.status_var.set("Success")


if __name__ == "__main__":
    app = App()
    app.mainloop()

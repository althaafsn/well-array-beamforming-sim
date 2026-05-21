from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from well_array_sim.io.om_activity import (
    export_csv,
    filter_records,
    format_summary,
    load_om_activity,
    summarize,
)

COLUMN_HEADERS = (
    "Event",
    "Company",
    "Pipeline",
    "OD mm (NPS)",
    "Length m",
    "Commodity",
    "Integrity dig",
    "Dig count",
    "Commencement",
    "Province",
    "Nearest centre",
    "Circumstance",
)


class OmDataApp:
    """Standalone CER O&M data browser — separate from the pipe simulation GUI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CER Operation & Maintenance — Data Browser")
        self.root.geometry("1200x700")
        self.root.minsize(900, 500)

        self.records = load_om_activity()
        self.filtered: list = []

        self._build_filters()
        self._build_table()
        self._build_actions()
        self._apply_filters()

    def _build_filters(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Filters", padding=8)
        frame.pack(fill=tk.X, padx=8, pady=8)

        self.integrity_var = tk.BooleanVar(value=False)
        self.has_od_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Integrity dig only", variable=self.integrity_var, command=self._apply_filters).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 12)
        )
        ttk.Checkbutton(frame, text="Has outside diameter", variable=self.has_od_var, command=self._apply_filters).grid(
            row=0, column=1, sticky=tk.W, padx=(0, 12)
        )

        ttk.Label(frame, text="Province").grid(row=0, column=2, sticky=tk.W)
        self.province_var = tk.StringVar(value="")
        provinces = sorted({r.province_territory for r in self.records if r.province_territory})
        self.province_combo = ttk.Combobox(
            frame,
            textvariable=self.province_var,
            values=[""] + provinces,
            width=24,
        )
        self.province_combo.grid(row=0, column=3, sticky=tk.W, padx=(4, 12))
        self.province_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())

        ttk.Label(frame, text="Search").grid(row=0, column=4, sticky=tk.W)
        self.search_var = tk.StringVar(value="")
        search_entry = ttk.Entry(frame, textvariable=self.search_var, width=28)
        search_entry.grid(row=0, column=5, sticky=tk.W, padx=(4, 0))
        search_entry.bind("<Return>", lambda _e: self._apply_filters())
        ttk.Button(frame, text="Apply", command=self._apply_filters).grid(row=0, column=6, padx=(8, 0))

    def _build_table(self) -> None:
        outer = ttk.Frame(self.root, padding=(8, 0))
        outer.pack(fill=tk.BOTH, expand=True)

        self.summary_var = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.summary_var, justify=tk.LEFT).pack(fill=tk.X, pady=(0, 6))

        table_frame = ttk.Frame(outer)
        table_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(
            table_frame,
            columns=COLUMN_HEADERS,
            show="headings",
            selectmode="browse",
        )
        widths = (90, 160, 140, 110, 70, 100, 80, 70, 90, 110, 120, 260)
        for header, width in zip(COLUMN_HEADERS, widths, strict=True):
            self.tree.heading(header, text=header)
            self.tree.column(header, width=width, stretch=(header == "Circumstance"))

        yscroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def _build_actions(self) -> None:
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill=tk.X)
        ttk.Button(frame, text="Export filtered CSV", command=self._on_export).pack(side=tk.LEFT, padx=(0, 8))

    def _apply_filters(self) -> None:
        self.filtered = filter_records(
            self.records,
            integrity_only=self.integrity_var.get(),
            has_od=self.has_od_var.get(),
            province=self.province_var.get() or None,
            search=self.search_var.get() or None,
        )
        self.tree.delete(*self.tree.get_children())
        for idx, record in enumerate(self.filtered):
            self.tree.insert("", tk.END, iid=str(idx), values=record.to_display_row())
        self.summary_var.set(format_summary(summarize(self.filtered)))

    def _selected_record(self):
        selection = self.tree.selection()
        if not selection:
            return None
        return self.filtered[int(selection[0])]

    def _on_export(self) -> None:
        if not self.filtered:
            messagebox.showinfo("Export", "No rows to export.")
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".csv",
            initialfile="om_filtered.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        export_csv(self.filtered, Path(path))
        messagebox.showinfo("Export", f"Saved {len(self.filtered)} rows to:\n{path}")


def main() -> None:
    root = tk.Tk()
    OmDataApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Shared Plotly reporting helper for CE6021 lab evaluation scripts."""
import os
import plotly.io as pio


def save_html(*figs, output_path):
    """Write one or more Plotly figures to a single combined HTML file.

    Args:
        *figs: One or more plotly.graph_objects.Figure instances.
        output_path (str): Path to write the combined HTML report to.
    """
    parts = [
        pio.to_html(fig, full_html=False,
                    include_plotlyjs=("cdn" if i == 0 else False))
        for i, fig in enumerate(figs)
    ]
    title = os.path.basename(output_path)
    html = (
        "<!DOCTYPE html><html>"
        "<head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{{font-family:sans-serif;padding:20px}}"
        "hr{{margin:30px 0;border:1px solid #ccc}}</style></head>"
        "<body>" + "<hr>".join(parts) + "</body></html>"
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\nResults saved → {output_path}")
    print("Open in VS Code: right-click → Open with Live Server")
    print("         or run: python -m http.server 5000")

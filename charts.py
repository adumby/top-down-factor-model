"""Plotly styling: one template so every chart reads as one system (reference palette, light mode)."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CURRENT = "#2a78d6"     # slot 1 blue  -> the allocation you are editing
BASELINE = "#eb6834"    # slot 2 orange -> the baseline (sheet) allocation
THIRD = "#1baf7a"       # slot 3 aqua
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# diverging blue <-> gray <-> red (negative = red, positive = blue)
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
GOOD, CRITICAL = "#006300", "#d03b3b"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

pio.templates["macro"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, color=INK2, size=13),
        title=dict(font=dict(color=INK, size=15), x=0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        colorway=SERIES,
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
                   showgrid=False),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
                   zerolinewidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.25, bargroupgap=0.08,
    )
)
pio.templates.default = "macro"


def line(fig: go.Figure, x, y, name: str, color: str, dash: str | None = None, fmt: str = ".2f",
         width: float = 2) -> None:
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines",
                             line=dict(color=color, width=width, dash=dash),
                             hovertemplate=f"%{{y:{fmt}}}<extra>{name}</extra>"))


def bars(fig: go.Figure, x, y, name: str, color: str, horizontal: bool = False, fmt: str = ".2f") -> None:
    kw = dict(x=y, y=x, orientation="h") if horizontal else dict(x=x, y=y)
    hv = "%{x:" + fmt + "}" if horizontal else "%{y:" + fmt + "}"
    lab = "%{y}" if horizontal else "%{x}"
    fig.add_trace(go.Bar(name=name, marker=dict(color=color, line=dict(color=SURFACE, width=2),
                                                cornerradius=4),
                         hovertemplate=f"{lab}: {hv}<extra>{name}</extra>", **kw))


def finish(fig: go.Figure, title: str, height: int = 380, ytitle: str = "", xtitle: str = "",
           hover: str = "closest", legend: bool = True) -> go.Figure:
    fig.update_layout(template="macro", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      title=dict(text=title, y=1.0, yref="container", yanchor="top", pad=dict(t=8)),
                      height=height, hovermode=hover, showlegend=legend,
                      yaxis_title=ytitle, xaxis_title=xtitle,
                      margin=dict(l=10, r=10, t=(78 if legend else 48) if title else 30, b=10))
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig

from __future__ import annotations

import html
import json
import math
import zipfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_pack"
ZIP_PATH = ROOT / "data_pack.zip"

LANGUAGES = {
    "Українська": {
        "app_title": "Класифікація зображень за структурним порівнянням графів",
        "app_sub": "Інтерактивне дослідження результатів Graph Edit Distance (GED) для графів зображень та концептів.",
        "sidebar_filters": "Фільтри набору даних",
        "filter_picked": "Тільки відібрані",
        "pick_btn": "Відібрати",
        "unpick_btn": "Вилучити",
        "res_filter": "Результат класифікації",
        "correct": "Правильно",
        "incorrect": "Неправильно",
        "true_digit": "Справжня цифра",
        "pred_digit": "Передбачена цифра",
        "min_sim": "Мінімальна схожість",
        "max_sim": "Максимальна схожість",
        "sort_by": "Сортувати за",
        "asc_sort": "За зростанням",
        "metric_images": "Зображення",
        "metric_acc": "Точність",
        "metric_inc": "Неправильні",
        "metric_mean": "Сер. схожість",
        "nav_prev": "← Назад",
        "nav_next": "Вперед →",
        "viewing": "Перегляд {pos} з {total}",
        "vis_comp": "Візуальне порівняння",
        "test_img": "Тестове зображення",
        "concept_comp_sidebar": "Порівняння з концептом",
        "true_lbl": "Факт",
        "pred_lbl": "Прогноз",
        "best_sim_lbl": "Найкраща схож.",
        "concept_sims": "Схожість з концептами",
        "tab_graph": "Порівняння графів",
        "tab_ops": "Операції редагування",
        "tab_matrix": "Матриця вартості",
        "tab_data": "Огляд даних",
        "sim_score": "Схожість",
        "raw_cost": "Вартість GED",
        "img_complex": "Складність зобр.",
        "con_complex": "Складність конц.",
        "legend_subst": "Точка/заміна",
        "legend_vector": "Вектор",
        "legend_start": "Початок/вставка",
        "legend_del": "Видалення",
        "substitutions": "Заміни",
        "deletions": "Видалення",
        "insertions": "Вставки",
        "no_images": "Зображень не знайдено",
        "no_ops": "Операції відсутні",
        "anim_play": "Відтворити",
        "anim_stop": "Пауза",
        "tab_anim": "Анімація трансформації",
        "anim_step": "Крок трансформації",
        "anim_cost": "Поточна накопичена вартість",
        "anim_op": "Поточна операція",
        "anim_start": "Початковий стан (Зображення)",
        "anim_end": "Кінцевий стан (Концепт)",
        "anim_desc": "Цей розділ демонструє покроковий процес перетворення графа зображення на граф концепту.",
    },
    "English": {
        "app_title": "Image Classification by Structural Graph Comparison",
        "app_sub": "Interactive exploration of Graph Edit Distance results for MNIST-like image graphs and concept graphs.",
        "sidebar_filters": "Dataset Filters",
        "filter_picked": "Only picked images",
        "pick_btn": "Pick for testing",
        "unpick_btn": "Unpick image",
        "res_filter": "Classification result",
        "correct": "Correct",
        "incorrect": "Incorrect",
        "true_digit": "True digit",
        "pred_digit": "Predicted digit",
        "min_sim": "Minimum best similarity",
        "max_sim": "Maximum best similarity",
        "sort_by": "Sort images by",
        "asc_sort": "Ascending sort",
        "metric_images": "Images",
        "metric_acc": "Accuracy",
        "metric_inc": "Incorrect cases",
        "metric_mean": "Mean best similarity",
        "nav_prev": "← Previous",
        "nav_next": "Next →",
        "viewing": "Viewing {pos} of {total}",
        "vis_comp": "Visual Comparison",
        "test_img": "Test Image",
        "concept_comp_sidebar": "Concept comparison",
        "true_lbl": "True",
        "pred_lbl": "Predicted",
        "best_sim_lbl": "Best sim",
        "concept_sims": "Concept Similarities",
        "tab_graph": "Graph Comparison",
        "tab_ops": "Edit Operations",
        "tab_matrix": "Cost Matrix",
        "tab_data": "Dataset Overview",
        "sim_score": "Similarity",
        "raw_cost": "Raw cost",
        "img_complex": "Image complexity",
        "con_complex": "Concept complexity",
        "legend_subst": "Point/substitution",
        "legend_vector": "Vector",
        "legend_start": "StartPoint or insertion",
        "legend_del": "Deletion",
        "substitutions": "Substitutions",
        "deletions": "Deletions",
        "insertions": "Insertions",
        "no_images": "No images match the selected filters.",
        "no_ops": "No operations recorded.",
        "anim_play": "Play",
        "anim_stop": "Pause",
        "tab_anim": "Transformation Animation",
        "anim_step": "Transformation Step",
        "anim_cost": "Current Cumulative Cost",
        "anim_op": "Current Operation",
        "anim_start": "Initial State (Image)",
        "anim_end": "Final State (Concept)",
        "anim_desc": "This section demonstrates the step-by-step process of transforming the image graph into the concept graph.",
    }
}

NODE_COLORS = {
    "Point": "#2563eb",
    "Vector": "#059669",
    "StartPoint": "#dc2626",
    "EndPoint": "#7c3aed",
    "Corner": "#d97706",
}
OP_COLORS = {
    "substitution": "#2563eb",
    "deletion": "#f97316",
    "insertion": "#dc2626",
    "none": "#64748b",
}


st.set_page_config(
    page_title="GED Image Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --ink: #111827;
            --muted: #64748b;
            --line: #d9e1ea;
            --soft: #f6f8fb;
        }

        .block-container {
            padding-top: 1.25rem;
            max-width: 1480px;
        }

        h1, h2, h3 {
            letter-spacing: 0 !important;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.82rem;
            font-weight: 650;
            white-space: nowrap;
        }

        .status-correct {
            color: #166534;
            background: #ecfdf3;
            border-color: #bbf7d0;
        }

        .status-wrong {
            color: #991b1b;
            background: #fff1f2;
            border-color: #fecdd3;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .graph-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }

        .graph-panel {
            border: 1px solid #d9e1ea;
            border-radius: 8px;
            background: #ffffff;
            overflow: hidden;
        }

        .graph-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 10px 12px;
            border-bottom: 1px solid #d9e1ea;
            color: #111827;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 14px;
            font-weight: 700;
        }

        .graph-subtitle {
            color: #64748b;
            font-size: 12px;
            font-weight: 500;
        }

        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px 12px;
            margin: 8px 0 0;
            color: #475569;
            font-size: 0.84rem;
        }

        .legend span {
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }

        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }

        @media (max-width: 900px) {
            .graph-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_data_pack() -> None:
    if DATA_DIR.exists():
        return
    if not ZIP_PATH.exists():
        st.error("`data_pack` is missing. Put `data_pack.zip` in the project root.")
        st.stop()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.infolist():
            if member.filename.startswith("__MACOSX/"):
                continue
            zf.extract(member, ROOT)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_image_path(kind: str, item_id: str) -> Path | None:
    """Locate image using the exact same logic as graph loading."""
    sub = "images" if kind == "image" else "concepts"
    folder = DATA_DIR / sub
    if not folder.exists():
        # Fallback to checking root if subfolders don't exist
        folder = DATA_DIR
    
    for ext in [".png", ".jpg", ".jpeg", ".svg"]:
        p = folder / f"{item_id}{ext}"
        if p.is_file():
            return p
        # Handle potential prefixing in filenames
        p_prefixed = folder / f"{kind}_{item_id}{ext}"
        if p_prefixed.is_file():
            return p_prefixed

    return None


@st.cache_data(show_spinner=False)
def load_indexes() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_data_pack()
    images = pd.DataFrame(read_json(DATA_DIR / "images.json"))
    concepts = pd.DataFrame(read_json(DATA_DIR / "concepts.json"))

    images["result"] = images["correct"].map({True: "Correct", False: "Incorrect"})
    images["label"] = images.apply(
        lambda row: f"{row.digit_class} -> {row.predicted_class} | {row.best_similarity:.4f} | {short_id(row.image_id)}",
        axis=1,
    )
    concepts["label"] = concepts.apply(
        lambda row: f"{row.concept_id} | digit {row.digit_class} | complexity {row.complexity}",
        axis=1,
    )
    return images, concepts


@st.cache_data(show_spinner=False)
def load_graph(kind: str, item_id: str) -> dict[str, Any]:
    if kind == "image":
        return read_json(DATA_DIR / "images" / f"{item_id}.json")
    return read_json(DATA_DIR / "concepts" / f"{item_id}.json")


@st.cache_data(show_spinner=False)
def load_ged(image_id: str, concept_id: str) -> dict[str, Any]:
    return read_json(DATA_DIR / "ged_results" / image_id / f"{concept_id}.json")


@st.cache_data(show_spinner=False)
def load_all_similarities(image_id: str, concept_ids: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for concept_id in concept_ids:
        result = load_ged(image_id, concept_id)
        rows.append(
            {
                "concept_id": concept_id,
                "similarity": float(result.get("similarity") or 0.0),
                "raw_cost": float(result.get("raw_cost") or 0.0),
                "skipped": bool(result.get("skipped", False)),
            }
        )
    return pd.DataFrame(rows)


def short_id(value: str, size: int = 8) -> str:
    return value[:size]


def center_value(value: Any, default: float = 0.0) -> float:
    if isinstance(value, dict):
        value = value.get("center", value.get("min", default))
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def node_kind(node: dict[str, Any]) -> str:
    labels = node.get("labels") or []
    if "StartPoint" in labels:
        return "StartPoint"
    if "Vector" in labels:
        return "Vector"
    if "EndPoint" in labels:
        return "EndPoint"
    if "Corner" in labels:
        return "Corner"
    return labels[0] if labels else "Point"


def node_position(node: dict[str, Any], width: int, height: int) -> tuple[float, float]:
    nx = max(-1.15, min(1.15, center_value(node.get("normalized_x"))))
    ny = max(-1.15, min(1.15, center_value(node.get("normalized_y"))))
    x = 30 + ((nx + 1.15) / 2.3) * (width - 60)
    y = 30 + ((ny + 1.15) / 2.3) * (height - 60)
    return x, y


def operation_maps(ged: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    image_ops: dict[str, dict[str, Any]] = {}
    concept_ops: dict[str, dict[str, Any]] = {}
    for op in ged.get("node_operations") or []:
        if op.get("image_node"):
            image_ops[op["image_node"]] = op
        if op.get("concept_node"):
            concept_ops[op["concept_node"]] = op
    return image_ops, concept_ops


def property_cost_text(op: dict[str, Any]) -> str:
    costs = op.get("property_costs") or {}
    if not costs:
        return ""
    ranked = sorted(costs.items(), key=lambda item: item[1], reverse=True)
    return "; ".join(f"{key}: {value:.4f}" for key, value in ranked[:6])


def graph_svg(
    graph: dict[str, Any],
    title: str,
    subtitle: str,
    op_by_node: dict[str, dict[str, Any]],
    width: int = 560,
    height: int = 430,
) -> str:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_by_id = {node["id"]: node for node in nodes}
    positions = {node["id"]: node_position(node, width, height) for node in nodes}

    edge_markup: list[str] = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        edge_markup.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#94a3b8" stroke-width="2" stroke-linecap="round" opacity="0.74"/>'
        )

    node_markup: list[str] = []
    for node_id, node in node_by_id.items():
        x, y = positions[node_id]
        kind = node_kind(node)
        op = op_by_node.get(node_id, {})
        op_type = op.get("type", "none")
        cost = float(op.get("cost") or 0.0)
        base = NODE_COLORS.get(kind, NODE_COLORS["Point"])
        color = OP_COLORS.get(op_type, base) if op_type != "substitution" else base
        radius = 8 if kind != "Vector" else 7
        stroke = "#111827" if kind == "StartPoint" else "#ffffff"
        stroke_width = 3 if kind == "StartPoint" else 2
        labels = ", ".join(node.get("labels") or [])
        title_text = html.escape(
            f"{short_id(node_id, 14)} | {labels} | op={op_type} | cost={cost:.4f}"
            + (f" | {property_cost_text(op)}" if op else "")
        )
        if kind == "Vector":
            points = [
                f"{x:.1f},{(y - radius):.1f}",
                f"{(x + radius):.1f},{y:.1f}",
                f"{x:.1f},{(y + radius):.1f}",
                f"{(x - radius):.1f},{y:.1f}",
            ]
            node_markup.append(
                f'<polygon points="{" ".join(points)}" fill="{color}" stroke="{stroke}" '
                f'stroke-width="{stroke_width}"><title>{title_text}</title></polygon>'
            )
        else:
            node_markup.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}"><title>{title_text}</title></circle>'
            )

        if cost > 0:
            node_markup.append(
                f'<text x="{x + 10:.1f}" y="{y - 9:.1f}" fill="#334155" font-size="10" '
                f'font-family="Inter,Arial">{cost:.2f}</text>'
            )

    return f"""
    <div class="graph-panel">
      <div class="graph-title">
        <span>{html.escape(title)}</span>
        <span class="graph-subtitle">{html.escape(subtitle)}</span>
      </div>
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">
        <rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>
        <line x1="30" y1="{height / 2:.1f}" x2="{width - 30}" y2="{height / 2:.1f}" stroke="#e2e8f0"/>
        <line x1="{width / 2:.1f}" y1="30" x2="{width / 2:.1f}" y2="{height - 30}" stroke="#e2e8f0"/>
        {''.join(edge_markup)}
        {''.join(node_markup)}
      </svg>
    </div>
    """


def graph_component_html(left_graph: str, right_graph: str) -> str:
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        html, body {{
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .graph-grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px;
          padding: 0 1px 10px;
          box-sizing: border-box;
        }}

        .graph-panel {{
          border: 1px solid #d9e1ea;
          border-radius: 8px;
          background: #ffffff;
          overflow: hidden;
          min-width: 0;
        }}

        .graph-title {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          padding: 10px 12px;
          border-bottom: 1px solid #d9e1ea;
          color: #111827;
          font-size: 14px;
          font-weight: 700;
        }}

        .graph-title span:first-child {{
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }}

        .graph-subtitle {{
          color: #64748b;
          font-size: 12px;
          font-weight: 500;
          flex: 0 0 auto;
        }}

        svg {{
          display: block;
          width: 100%;
          max-width: 100%;
        }}

        @media (max-width: 760px) {{
          .graph-grid {{
            grid-template-columns: 1fr;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="graph-grid">{left_graph}{right_graph}</div>
    </body>
    </html>
    """


def operation_dataframe(ged: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for op in ged.get("node_operations") or []:
        rows.append(
            {
                "scope": "node",
                "type": op.get("type"),
                "image_item": short_id(str(op.get("image_node") or ""), 18),
                "concept_item": short_id(str(op.get("concept_node") or ""), 18),
                "cost": float(op.get("cost") or 0.0),
                "largest_property_costs": property_cost_text(op),
            }
        )
    for op in ged.get("edge_operations") or []:
        rows.append(
            {
                "scope": "edge",
                "type": op.get("type"),
                "image_item": " -> ".join(short_id(str(item), 8) for item in (op.get("image_edge") or [])),
                "concept_item": " -> ".join(short_id(str(item), 8) for item in (op.get("concept_edge") or [])),
                "cost": float(op.get("cost") or 0.0),
                "largest_property_costs": "",
            }
        )
    return pd.DataFrame(rows)


def cost_matrix_dataframe(ged: dict[str, Any]) -> pd.DataFrame:
    matrix = ged.get("cost_matrix") or {}
    values = matrix.get("matrix") or []
    image_ids = [short_id(str(item), 8) for item in matrix.get("image_node_ids") or []]
    concept_ids = [short_id(str(item), 8) for item in matrix.get("concept_node_ids") or []]
    if not values:
        return pd.DataFrame()
    return pd.DataFrame(values, index=concept_ids, columns=image_ids)


def similarity_chart(similarities: pd.DataFrame, concepts: pd.DataFrame, true_class: int, predicted_class: int) -> alt.Chart:
    if "digit_class" in similarities.columns:
        chart_df = similarities.copy()
    else:
        chart_df = similarities.merge(concepts[["concept_id", "digit_class"]], on="concept_id", how="left")
    chart_df["concept"] = chart_df["concept_id"].astype(str)
    chart_df["role"] = "other"
    chart_df.loc[chart_df["digit_class"] == true_class, "role"] = "true class"
    chart_df.loc[chart_df["digit_class"] == predicted_class, "role"] = "predicted class"
    chart_df.loc[
        (chart_df["digit_class"] == true_class) & (chart_df["digit_class"] == predicted_class),
        "role",
    ] = "true + predicted"

    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("concept:N", sort="-y", title="Concept Graph"),
            y=alt.Y(
                "similarity:Q", 
                title="Similarity Score", 
                # Starting the scale at 0.5 makes the "gap" between 
                # 0.7 and 0.9 much more obvious visually.
                scale=alt.Scale(domain=[0.5, 1.0], clamp=True)
            ),
            color=alt.Color(
                "role:N",
                scale=alt.Scale(
                    domain=["true + predicted", "predicted class", "true class", "other"],
                    range=["#059669", "#2563eb", "#d97706", "#94a3b8"],
                ),
                title="Role",
            ),
            tooltip=[
                "concept_id:N",
                "digit_class:N",
                alt.Tooltip("similarity:Q", format=".4f"),
                alt.Tooltip("raw_cost:Q", format=".4f"),
            ],
        )
        .properties(height=285)
    )


def render_header(images: pd.DataFrame, t: dict) -> None:
    st.title(t["app_title"])
    st.caption(t["app_sub"])

    total = len(images)
    correct = int(images["correct"].sum())
    incorrect = total - correct
    accuracy = correct / total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        t["metric_images"], 
        f"{total}", 
        help="The total number of images currently visible after applying your sidebar filters."
    )
    c2.metric(
        t["metric_acc"], 
        f"{accuracy:.1%}", 
        help="Percentage of images where the algorithm's best match belongs to the correct digit class."
    )
    c3.metric(
        t["metric_inc"], 
        f"{incorrect}", 
        help="Total number of images that were misclassified based on structural similarity."
    )
    c4.metric(
        t["metric_mean"], 
        f"{images['best_similarity'].mean():.4f}", 
        help="The average similarity score of the top match for every image in the current selection."
    )


def filter_images(images: pd.DataFrame, t: dict) -> pd.DataFrame:
    if "picked_images" not in st.session_state:
        st.session_state.picked_images = []

    with st.sidebar:
        st.header(t["sidebar_filters"])
        
        show_picked = st.checkbox(t.get("filter_picked", "Show only picked ⭐️"), value=False)
        
        res_options = {t["correct"]: "Correct", t["incorrect"]: "Incorrect"}
        result_filter = st.multiselect(
            t["res_filter"],
            options=list(res_options.keys()),
            default=list(res_options.keys()),
        )
        mapped_res = [res_options[r] for r in result_filter]

        true_classes = st.multiselect(
            t["true_digit"],
            sorted(images["digit_class"].unique().tolist()),
            default=sorted(images["digit_class"].unique().tolist()),
        )
        predicted_classes = st.multiselect(
            t["pred_digit"],
            sorted(images["predicted_class"].unique().tolist()),
            default=sorted(images["predicted_class"].unique().tolist()),
        )
        min_similarity = st.slider(t["min_sim"], 0.0, 1.0, 0.0, 0.01)
        max_similarity = st.slider(t["max_sim"], 0.0, 1.0, 1.0, 0.01)
        
        sort_by = st.selectbox(
            t["sort_by"],
            ["best_similarity", "complexity", "digit_class", "predicted_class"],
            index=0,
        )
        ascending = st.toggle(t["asc_sort"], value=False)

    mask = (
        images["result"].isin(mapped_res)
        & images["digit_class"].isin(true_classes)
        & images["predicted_class"].isin(predicted_classes)
        & (images["best_similarity"] >= min_similarity)
        & (images["best_similarity"] <= max_similarity)
    )
    
    if show_picked:
        mask = mask & images["image_id"].isin(st.session_state.picked_images)

    filtered = images[mask].copy()
    filtered = filtered.sort_values(sort_by, ascending=ascending)
    return filtered


def render_selected_case(images: pd.DataFrame, concepts: pd.DataFrame, t: dict) -> None:
    filtered = filter_images(images, t)
    if filtered.empty:
        st.warning(t["no_images"])
        return

    options = filtered["image_id"].tolist()
    labels = dict(zip(filtered["image_id"], filtered["label"], strict=False))

    # Manage selection state for easier navigation
    if "sel_img_id" not in st.session_state or st.session_state.sel_img_id not in options:
        st.session_state.sel_img_id = options[0]

    def nav_callback(delta: int):
        curr_idx = options.index(st.session_state.sel_img_id)
        st.session_state.sel_img_id = options[(curr_idx + delta) % len(options)]
        st.session_state.anim_step_idx = 0
        st.session_state.anim_playing = False
        st.session_state.anim_autoplay_next = False

    # Determine current position for the "X of Y" label
    try:
        current_pos = options.index(st.session_state.sel_img_id) + 1
    except ValueError:
        current_pos = 1

    # Navigation and selection hub
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 3, 1])
        with c1:
            st.button(t["nav_prev"], on_click=nav_callback, args=(-1,), use_container_width=True, key="main_nav_prev")
        
        with c2:
            st.selectbox(
                t["viewing"].format(pos=current_pos, total=len(options)),
                options,
                index=options.index(st.session_state.sel_img_id),
                key="sel_img_id",
                format_func=lambda item: labels.get(item, item),
                label_visibility="collapsed"
            )
        with c3:
            st.button(t["nav_next"], on_click=nav_callback, args=(1,), use_container_width=True, key="main_nav_next")

    selected_image_id = st.session_state.sel_img_id
    selected = images.loc[images["image_id"] == selected_image_id].iloc[0]

    concept_ids = tuple(concepts["concept_id"].tolist())
    similarities = load_all_similarities(selected_image_id, concept_ids)
    similarities = similarities.merge(concepts[["concept_id", "digit_class", "label"]], on="concept_id", how="left")
    best_concept_id = similarities.sort_values("similarity", ascending=False).iloc[0]["concept_id"]

    st.sidebar.divider()
    concept_options = similarities.sort_values("similarity", ascending=False)["concept_id"].tolist()
    selected_concept_id = st.sidebar.selectbox(
        t["concept_comp_sidebar"],
        concept_options,
        index=concept_options.index(best_concept_id),
        format_func=lambda cid: f"{cid} | sim {similarities.loc[similarities.concept_id == cid, 'similarity'].iloc[0]:.4f}",
    )

    concept_row = concepts.loc[concepts["concept_id"] == selected_concept_id].iloc[0]
    ged = load_ged(selected_image_id, selected_concept_id)
    image_graph = load_graph("image", selected_image_id)
    concept_graph = load_graph("concept", selected_concept_id)
    image_ops, concept_ops = operation_maps(ged)

    result_class = "status-correct" if bool(selected.correct) else "status-wrong"
    result_text = t["correct"] if bool(selected.correct) else t["incorrect"]

    top_left, top_right = st.columns([1.2, 1.0], gap="large")
    with top_left:
        st.subheader(t["vis_comp"])
        
        img_c1, img_c2 = st.columns(2)
        with img_c1:
            test_img_path = find_image_path("image", selected_image_id)
            if test_img_path:
                st.caption(t["test_img"])
                st.image(str(test_img_path), use_container_width=True)
        
        with img_c2:
            concept_img_path = find_image_path("concept", selected_concept_id)
            if concept_img_path:
                st.caption(f"{t['concept_comp_sidebar']} {selected_concept_id}")
                st.image(str(concept_img_path), use_container_width=True)

        st.markdown(
            f"""
            <span class="status-pill {result_class}" title="Whether the predicted digit matches the ground truth.">{result_text}</span>
            <div class="small-muted" style="margin-top: 0.7rem;">
                id: <code>{html.escape(selected_image_id)}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Pick/Unpick button for bookmarking test cases
        is_picked = selected_image_id in st.session_state.get("picked_images", [])
        if st.button(
            ("⭐ " + t.get("unpick_btn", "Unpick")) if is_picked else ("☆ " + t.get("pick_btn", "Pick")), 
            key="btn_pick_toggle",
            use_container_width=True
        ):
            if is_picked:
                st.session_state.picked_images.remove(selected_image_id)
            else:
                st.session_state.picked_images.append(selected_image_id)
            st.rerun()

        a, b, c = st.columns(3)
        a.metric(
            t["true_lbl"], 
            int(selected.digit_class), 
            help="The actual label assigned to this image in the dataset."
        )
        b.metric(
            t["pred_lbl"], 
            int(selected.predicted_class), 
            help="The label of the concept that achieved the highest structural similarity score."
        )
        c.metric(
            t["best_sim_lbl"], 
            f"{float(selected.best_similarity):.4f}", 
            help="The highest similarity score found across all available reference concepts."
        )

    with top_right:
        st.subheader(t["concept_sims"])
        st.altair_chart(
            similarity_chart(
                similarities,
                concepts,
                int(selected.digit_class),
                int(selected.predicted_class),
            ),
            width="stretch",
        )

    tabs = st.tabs([t["tab_graph"], t["tab_ops"], t["tab_anim"], t["tab_matrix"], t["tab_data"]])

    with tabs[0]:
        st.subheader(f"{t['tab_graph']}: {selected_concept_id}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            t["sim_score"], 
            f"{float(ged.get('similarity') or 0):.4f}", 
            help="Normalized similarity score (0-1) between the test graph and this specific concept."
        )
        m2.metric(
            t["raw_cost"], 
            f"{float(ged.get('raw_cost') or 0):.4f}", 
            help="The total mathematical cost of the graph edit operations (GED) needed to transform one graph into the other."
        )
        m3.metric(
            t["img_complex"], 
            int(selected.complexity), 
            help="Structural weight of the test image (typically nodes + edges)."
        )
        m4.metric(
            t["con_complex"], 
            int(concept_row.complexity), 
            help="Structural weight of the reference concept graph."
        )

        st.markdown(
            f"""
            <div class="legend">
              <span><i class="dot" style="background:#2563eb"></i>{t['legend_subst']}</span>
              <span><i class="dot" style="background:#059669"></i>{t['legend_vector']}</span>
              <span><i class="dot" style="background:#dc2626"></i>{t['legend_start']}</span>
              <span><i class="dot" style="background:#f97316"></i>{t['legend_del']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left_graph = graph_svg(
            image_graph,
            "Image graph",
            f"{len(image_graph.get('nodes', []))} nodes, {len(image_graph.get('edges', []))} edges",
            image_ops,
        )
        right_graph = graph_svg(
            concept_graph,
            f"Concept {selected_concept_id}",
            f"digit {int(concept_row.digit_class)}",
            concept_ops,
        )
        components.html(graph_component_html(left_graph, right_graph), height=515, scrolling=False)

    with tabs[1]:
        st.subheader(t["tab_ops"])
        ops = operation_dataframe(ged)
        if ops.empty:
            st.info(t["no_ops"])
        else:
            op_counts = Counter(ops["type"])
            o1, o2, o3 = st.columns(3)
            o1.metric(t["substitutions"], op_counts.get("substitution", 0))
            o2.metric(t["deletions"], op_counts.get("deletion", 0))
            o3.metric(t["insertions"], op_counts.get("insertion", 0))
            st.dataframe(
                ops.sort_values(["scope", "cost"], ascending=[True, False]),
                width="stretch",
                hide_index=True,
            )

    with tabs[2]:
        st.subheader(t["tab_anim"])
        st.info(t["anim_desc"])

        # 1. Prepare ordered transformation steps
        node_ops = ged.get("node_operations") or []
        edge_ops = ged.get("edge_operations") or []
        
        steps = []
        # Logical order: Edge Deletions -> Node Deletions -> Substitutions -> Node Insertions -> Edge Insertions
        steps.extend([op for op in edge_ops if op['type'] == 'deletion'])
        steps.extend([op for op in node_ops if op['type'] == 'deletion'])
        steps.extend([op for op in node_ops if op['type'] == 'substitution'])
        steps.extend([op for op in node_ops if op['type'] == 'insertion'])
        steps.extend([op for op in edge_ops if op['type'] == 'insertion'])

        if not steps:
            st.info(t["no_ops"])
        else:
            # Animation State Management
            if "anim_step_idx" not in st.session_state:
                st.session_state.anim_step_idx = 0
            if "anim_playing" not in st.session_state:
                st.session_state.anim_playing = False

            # Autoplay logic
            if st.session_state.get("anim_autoplay_next") and st.session_state.anim_playing:
                st.session_state.anim_autoplay_next = False
                if st.session_state.anim_step_idx < len(steps):
                    st.session_state.anim_step_idx += 1
                if st.session_state.anim_step_idx >= len(steps):
                    st.session_state.anim_playing = False

            st.session_state.anim_step_idx = min(st.session_state.anim_step_idx, len(steps))

            c_play, c_prev, c_next, _ = st.columns([1, 1, 1, 3])
            
            if c_play.button(t["anim_stop"] if st.session_state.anim_playing else t["anim_play"], use_container_width=True, key="anim_play_toggle"):
                st.session_state.anim_playing = not st.session_state.anim_playing
                if st.session_state.anim_playing and st.session_state.anim_step_idx >= len(steps):
                    st.session_state.anim_step_idx = 0
                st.rerun()
            
            if c_prev.button(t["nav_prev"], use_container_width=True, key="anim_nav_prev"):
                st.session_state.anim_step_idx = max(0, st.session_state.anim_step_idx - 1)
                st.session_state.anim_playing = False
                st.rerun()

            if c_next.button(t["nav_next"], use_container_width=True, key="anim_nav_next"):
                st.session_state.anim_step_idx = min(len(steps), st.session_state.anim_step_idx + 1)
                st.session_state.anim_playing = False
                st.rerun()

            step_idx = st.slider(t["anim_step"], 0, len(steps), key="anim_step_idx")
            current_cost = sum(float(op.get('cost', 0)) for op in steps[:step_idx])
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(t["anim_cost"], f"{current_cost:.4f}")
            
            if step_idx > 0:
                last_op = steps[step_idx - 1]
                op_type = last_op.get('type', '').capitalize()
                scope = "Node" if "node" in str(last_op.keys()) else "Edge"
                col_m2.metric(t["anim_op"], f"{op_type} {scope}", delta=f"+{float(last_op.get('cost',0)):.4f}", delta_color="inverse")
            else:
                col_m2.metric(t["anim_op"], t["anim_start"])

            # 2. Reconstruct graph state at step_idx
            image_nodes = {n['id']: n for n in image_graph.get('nodes', [])}
            concept_nodes = {n['id']: n for n in concept_graph.get('nodes', [])}
            
            # If we are at the end, show the exact concept graph
            if step_idx == len(steps):
                temp_graph = concept_graph
            else:
                # Track what each original image node became (to handle edges correctly)
                deleted_node_ids = {op['image_node'] for op in steps[:step_idx] if op['type'] == 'deletion' and 'image_node' in op}
                node_subst_map = {op['image_node']: op['concept_node'] for op in steps[:step_idx] if op['type'] == 'substitution' and 'image_node' in op}
                inserted_node_ids = {op['concept_node'] for op in steps[:step_idx] if op['type'] == 'insertion' and 'concept_node' in op}
                
                current_nodes = []
                node_id_map = {} # image_id -> current_id

                for nid, node in image_nodes.items():
                    if nid in deleted_node_ids:
                        continue
                    if nid in node_subst_map:
                        con_id = node_subst_map[nid]
                        current_nodes.append(concept_nodes[con_id])
                        node_id_map[nid] = con_id
                    else:
                        current_nodes.append(node)
                        node_id_map[nid] = nid
                
                for nid in inserted_node_ids:
                    if nid not in [n['id'] for n in current_nodes]:
                        current_nodes.append(concept_nodes[nid])

                deleted_edge_tuples = {tuple(sorted(op['image_edge'])) for op in steps[:step_idx] if op['type'] == 'deletion' and 'image_edge' in op}
                inserted_edges = [op['concept_edge'] for op in steps[:step_idx] if op['type'] == 'insertion' and 'concept_edge' in op]
                
                current_edges = []
                for edge in image_graph.get('edges', []):
                    e_tuple = tuple(sorted((edge['source'], edge['target'])))
                    if e_tuple not in deleted_edge_tuples:
                        s, t_ = edge['source'], edge['target']
                        if s in node_id_map and t_ in node_id_map:
                            current_edges.append({"source": node_id_map[s], "target": node_id_map[t_]})
                
                for e_tuple in inserted_edges:
                    current_edges.append({"source": e_tuple[0], "target": e_tuple[1]})

                temp_graph = {"nodes": current_nodes, "edges": current_edges}
            
            # Highlight current operation
            active_op_map = {}
            if step_idx > 0:
                active_op = steps[step_idx-1]
                if active_op.get("image_node"): active_op_map[active_op["image_node"]] = active_op
                if active_op.get("concept_node"): active_op_map[active_op["concept_node"]] = active_op
                if active_op.get("image_edge"):
                    for node_id in active_op["image_edge"]: active_op_map[node_id] = active_op
                if active_op.get("concept_edge"):
                    for node_id in active_op["concept_edge"]: active_op_map[node_id] = active_op

            anim_svg = graph_svg(
                temp_graph,
                f"{t['tab_anim']} - {t['metric_images'] if step_idx == 0 else t['concept_comp_sidebar']}",
                f"Step {step_idx}/{len(steps)}",
                active_op_map,
            )
            components.html(graph_component_html(anim_svg, ""), height=515, scrolling=False)

            if st.session_state.anim_playing:
                if st.session_state.anim_step_idx < len(steps):
                    time.sleep(0.4)
                    st.session_state.anim_autoplay_next = True
                    st.rerun()
                else:
                    st.session_state.anim_playing = False
                    st.rerun()

    with tabs[3]:
        st.subheader(t["tab_matrix"])
        matrix = cost_matrix_dataframe(ged)
        if matrix.empty:
            st.info(t["no_ops"])
        else:
            st.caption("Rows are concept nodes; columns are image nodes. Lower values indicate a better node match.")
            matrix_long = (
                matrix.reset_index(names="concept_node")
                .melt(id_vars="concept_node", var_name="image_node", value_name="cost")
                .copy()
            )
            st.altair_chart(
                alt.Chart(matrix_long)
                .mark_rect()
                .encode(
                    x=alt.X("image_node:N", title="Image node"),
                    y=alt.Y("concept_node:N", title="Concept node"),
                    color=alt.Color(
                        "cost:Q",
                        scale=alt.Scale(scheme="yelloworangered"),
                        title="Cost",
                    ),
                    tooltip=[
                        "concept_node:N",
                        "image_node:N",
                        alt.Tooltip("cost:Q", format=".4g"),
                    ],
                )
                .properties(height=min(520, 80 + 28 * len(matrix.index))),
                width="stretch",
            )
            st.dataframe(matrix.round(4), width="stretch")

    with tabs[4]:
        st.subheader(t["tab_data"])
        st.caption(f"{len(filtered)} of {len(images)} images shown after sidebar filters.")
        display = filtered[
            [
                "image_id",
                "digit_class",
                "predicted_class",
                "best_similarity",
                "complexity",
                "node_count",
                "edge_count",
                "result",
            ]
        ].copy()
        display["image_id"] = display["image_id"].map(lambda value: short_id(value, 12))
        st.dataframe(display, width="stretch", hide_index=True)

        by_digit = (
            images.groupby(["digit_class", "correct"])
            .size()
            .reset_index(name="count")
            .replace({"correct": {True: "Correct", False: "Incorrect"}})
        )
        st.altair_chart(
            alt.Chart(by_digit)
            .mark_bar()
            .encode(
                x=alt.X("digit_class:O", title="True digit"),
                y=alt.Y("count:Q", title="Images"),
                color=alt.Color(
                    "correct:N",
                    scale=alt.Scale(domain=["Correct", "Incorrect"], range=["#059669", "#dc2626"]),
                    title="Result",
                ),
                tooltip=["digit_class:O", "correct:N", "count:Q"],
            )
            .properties(height=260),
            width="stretch",
        )


def main() -> None:
    inject_styles()
    ensure_data_pack()
    
    with st.sidebar:
        lang_choice = st.selectbox("Language / Мова", list(LANGUAGES.keys()), index=0)
    t = LANGUAGES[lang_choice]
    
    images, concepts = load_indexes()
    render_header(images, t)
    render_selected_case(images, concepts, t)


if __name__ == "__main__":
    main()

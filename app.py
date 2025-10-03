# ----------------------------------
# Malaysia’s GDP Streamlit Dashboard
# ----------------------------------
# Required libraries
import os
import json
import base64
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------
# Helper (Load data + Preprocessing)
# ----------------------------------
@st.cache_data
def load_gdp(path="data/gdp_yearly_state_sector.csv"):
    # Read data from file
    df = pd.read_csv(path)

    # Drop missing values
    df = df.dropna()

    # Convert string columns to category
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("category")

    # Reorder states: Malaysia first, others alphabetically, Supranational last
    unique_states = sorted(df['state'].unique())
    states = [s for s in unique_states if s not in ("Malaysia", "Supranational")]
    ordered = ["Malaysia"] + states + (["Supranational"] if "Supranational" in unique_states else [])

    df['state'] = pd.Categorical(df['state'], categories=ordered, ordered=True)

    return df

@st.cache_data
def load_malaysia_shapefile(path="data/MYS-ADM1_simplified.geojson"):
    gdf = gpd.read_file(path)

    # Standardize state names
    gdf["shapeName"] = gdf["shapeName"].replace({
        "Kuala Lumpur": "W.P. Kuala Lumpur",
        "Labuan": "W.P. Labuan",
        "Putrajaya": "W.P. Putrajaya",
        "Malacca": "Melaka",
        "Penang": "Pulau Pinang"
    })
    return gdf

# Load the data (with error check)
DATA_PATH = "data/gdp_yearly_state_sector.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at `{DATA_PATH}`. Please put your CSV there.")
    st.stop()
gdp_df = load_gdp(DATA_PATH)

SHAPE_PATH = "data/MYS-ADM1_simplified.geojson"
if not os.path.exists(SHAPE_PATH):
    st.error(f"Shapefile not found at `{SHAPE_PATH}`. Please put your GeoJSON there.")
    st.stop()
malaysia_states = load_malaysia_shapefile(SHAPE_PATH)

# ----------------------------------
# UI (Header + Title + Linked Logo)
# ----------------------------------
st.set_page_config(
    page_title="Malaysia’s GDP Report Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://mediasmart.my/v2/main-page/",
        "Report a bug": "mailto:mohdaizad115@gmail.com",
        "About": "Malaysia’s GDP Report Dashboard\ncreated with Streamlit by MSRSB"
    })

def make_header():
    # Logo path & URL
    logo_url = "https://mediasmart.my/v2/main-page/"
    logo_path = "assets/msrsb_logo4.png"

    # Convert image to base64
    logo_base64 = None
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
    
    # Inject custom CSS for header styling
    st.markdown(
        """
        <style>
            /* remove default padding on top */
            .block-container {
                padding-top: 1.5rem;
            }
            /* header styling */
            .dashboard-header {
                display: flex;
                align-items: center;
                padding: 0px 20px;
                border-bottom: 1px solid #444;
                margin-bottom: -31px;
            }
            /* title styling */
            .dashboard-header h1 {
                display: flex;
                align-items: center;
                font-size: 40px;
                font-weight: bold;
                color: white;
                margin: 0;
                margin-left: 15px;
            }
            .dashboard-header img {
                height: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render header
    if logo_base64:
        st.markdown(
            f"""
            <div class="dashboard-header">
                <a href="{logo_url}" target="_blank">
                    <img src="data:image/png;base64,{logo_base64}" alt="Logo">
                </a>
                <h1>Malaysia’s GDP Report Dashboard</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="dashboard-header">
                <h1>Malaysia’s GDP Report Dashboard</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

make_header()

# ----------------------------------
# UI (Sidebar)
# ----------------------------------
# Constants
YEARS = sorted(gdp_df["year"].unique())
DEFAULT_YEAR = max(YEARS)

STATES = ["Malaysia"] + [
    s for s in gdp_df["state"].cat.categories 
    if s not in ("Malaysia", "Supranational")
]

SECTORS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
SECTOR_LABELS = {
    "S0": "S0 (Total GDP)",
    "S1": "S1 (Agriculture)",
    "S2": "S2 (Mining & Quarrying)",
    "S3": "S3 (Manufacturing)",
    "S4": "S4 (Construction)",
    "S5": "S5 (Services)",
    "S6": "S6 (Import Duties)",
}

# Inject custom CSS for sidebar styling
st.markdown(
    """
    <style>
        /* Reduce top padding inside sidebar */
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 0rem;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.markdown("### Settings & Filters")

# GDP indicator selection (GDP vs GDP per capita)
gdp_type = st.sidebar.radio(
    "GDP Indicators Mode:",
    options=["gdp1", "gdp2"],
    format_func=lambda x: "GDP" if x == "gdp1" else "GDP per capita"
)

# Year mode (Single Year vs Year Range)
year_mode = st.sidebar.radio(
    "Year Selection Mode:",
    options=["single", "range"],
    format_func=lambda x: "Single Year" if x == "single" else "Year Range",
    index=0
)

# Inject CSS for slider styling
st.markdown("""
    <style>
    /* Fix cutoff issue for slider values in sidebar */
    .stSlider > div[data-baseweb="slider"] {
        padding-right: 0.6rem;
        padding-left: 0.5rem;
    }

    /* Make tick labels more readable */
    .stSlider > div[data-baseweb="slider"] span {
        font-size: 0.9rem; 
        color: #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Year slider (Single Year or Year Range)
DEFAULT_YEAR = 2023
if year_mode == "single":
    year_single = st.sidebar.slider(
        "Select Year:",
        min_value=min(YEARS),
        max_value=max(YEARS),
        value=DEFAULT_YEAR,
        format="%d",
        step=1
    )
    year_range = (min(YEARS), max(YEARS))
else:
    year_range = st.sidebar.slider(
        "Select Year Range:",
        min_value=min(YEARS),
        max_value=max(YEARS),
        value=(min(YEARS), max(YEARS)),
        format="%d",
        step=1
    )
    year_single = DEFAULT_YEAR

# Year range for trend chart (ignore single-year mode)
year_range_for_trend = (
    year_range if year_mode == "range"
    else (min(YEARS), max(YEARS))
)

# Custom evenly spaced tick labels for slider
tick_interval = 2
tick_years = [y for y in YEARS if (y - min(YEARS)) % tick_interval == 0]

def get_alignment(year):
    if year == min(YEARS):
        return "left"
    elif year == max(YEARS):
        return "right"
    elif year == 2017:
        return "left"
    elif year == 2021:
        return "right"
    else:
        return "center"

# Render ticks aligned with slider positions
ticks_html = "".join(
    f"<span style='flex:1; text-align:{get_alignment(y)}; font-size:12px; color:#ccc'>{y}</span>"
    for y in tick_years
)

st.sidebar.markdown(
    f"""
    <div style='display:flex; justify-content:space-between; width:100%; margin-top:-20px;'>
        {ticks_html}
    </div>
    """,
    unsafe_allow_html=True
)

# State selection
state_sel = st.sidebar.selectbox(
    "Select State:",
    options=STATES,
    index=0
)

# Sector selection
sector_sel_label = st.sidebar.selectbox(
    "Select Sector:",
    options=[SECTOR_LABELS[s] for s in SECTORS],
    index=0
)
# Convert label back to sector code (e.g., "S0")
sector_sel = next(k for k, v in SECTOR_LABELS.items() if v == sector_sel_label)

# ----------------------------------
# Data Selection Functions
# ----------------------------------
def selected_gdp_col(gdp_type_flag: str) -> str:
    """Return GDP column name based on mode"""
    return "gdp" if gdp_type_flag == "gdp1" else "gdp_capita"

def filter_for_current_year(df, year: int):
    """Filter dataframe for a single year"""
    return df[df['year'] == int(year)].copy()

def filter_for_range(df, year_min: int, year_max: int):
    """Filter dataframe for a range of years"""
    return df[(df['year'] >= int(year_min)) & (df['year'] <= int(year_max))].copy()

# Selected GDP column
gdp_col = selected_gdp_col(gdp_type)

# Filtered datasets
if year_mode == "single":
    # Single-year mode
    df_current_year = filter_for_current_year(gdp_df, year_single)
    df_previous_year = filter_for_current_year(gdp_df, year_single - 1)
    df_filtered = df_current_year  # same as current year
    # filter_for_range(gdp_df, year_single, year_single) = filter_for_current_year(gdp_df, year_single)
else:
    # Year-range mode
    df_filtered = filter_for_range(gdp_df, year_range[0], year_range[1])
    df_current_year = (
        df_filtered
        .groupby(['state', 'sector'], as_index=False)
        .agg({gdp_col: 'sum'})
        .rename(columns={gdp_col: gdp_col})
    )
    df_previous_year = None

# Special case filtering
df_single_year = filter_for_current_year(gdp_df, year_single)

# ----------------------------------
# KPIs
# ----------------------------------
# Define KPI datasets depending on mode
DEFAULT_KPI_YEAR = 2023
if year_mode == "single":
    kpi_current_year = filter_for_current_year(gdp_df, year_single)
    kpi_previous_year = filter_for_current_year(gdp_df, year_single - 1)
else:
    kpi_current_year = filter_for_current_year(gdp_df, DEFAULT_KPI_YEAR)
    kpi_previous_year = filter_for_current_year(gdp_df, DEFAULT_KPI_YEAR - 1)

# Floating, Format & Calculations
def format_myr(x):
    if pd.isna(x): return "N/A"
    try: val = float(x)
    except: return "N/A"   # check here later
    return f"RM{val:,.0f} Million"

def compute_national_gdp(df, sector, col):
    row = df[(df['state']=="Malaysia") & (df['sector']==sector)]
    return float(row.iloc[0][col]) if not row.empty else np.nan

def compute_top_state(df, sector, col):
    tmp = df[(df['sector']==sector) & (~df['state'].isin(["Malaysia","Supranational"]))]
    if tmp.empty: return None, np.nan
    top = tmp.sort_values(by=col, ascending=False).iloc[0]
    return top['state'], float(top[col])

def compute_top_sector(df, state, col):
    tmp = df[(df['state']==state) & (df['sector']!="S0")]
    if tmp.empty: return None, np.nan
    top = tmp.sort_values(by=col, ascending=False).iloc[0]
    return top['sector'], float(top[col])

def calc_growth(df_cur, df_prev, sector, col):
    cur = df_cur[(df_cur['state']=="Malaysia") & (df_cur['sector']==sector)]
    prev = df_prev[(df_prev['state']=="Malaysia") & (df_prev['sector']==sector)]
    if cur.empty or prev.empty: return None
    curv, prevv = float(cur.iloc[0][col]), float(prev.iloc[0][col])
    if prevv == 0: return None
    return (curv - prevv)/prevv * 100.0

# KPI values
nat_val = compute_national_gdp(kpi_current_year, sector_sel, gdp_col)
nat_val_prev = compute_national_gdp(kpi_previous_year, sector_sel, gdp_col)
top_state_name, top_state_val = compute_top_state(kpi_current_year, sector_sel, gdp_col)
top_sector_name, top_sector_val = compute_top_sector(kpi_current_year, state_sel, gdp_col)
growth_pct = calc_growth(kpi_current_year, kpi_previous_year, sector_sel, gdp_col)

# Load Font Awesome icons
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# KPI display
k1,k2,k3,k4 = st.columns(4)
kpi_style = """
    border-radius:8px; padding:14px 18px; color: white; height:85px;
    display:flex; flex-direction:row; align-items:center; justify-content:space-between;
    position:relative; overflow:hidden;
"""

icon_style = """
    font-size:40px; opacity:0.4;
    position:absolute; z-index:0; right:13px;
    top:50%; transform:translateY(-50%); 
    pointer-events:none;
"""

with k1:
    bg = "#1abc9c" if nat_val >= nat_val_prev else "#e74c3c"
    display_val = format_myr(nat_val)
    st.markdown(f"""
        <div style='background:{bg}; {kpi_style}'>
            <div style='display:flex; flex-direction:column;'>
                <div style='font-weight:700; font-size:20px'>{display_val}</div>
                <div style='font-size:13px; opacity:0.9'>
                    National {'GDP' if gdp_type=='gdp1' else 'GDP per capita'} for {'2023' if year_mode=='range' else year_single}
                </div>
            </div>
            <i class="fas fa-dollar-sign" style="{icon_style}"></i>
        </div>
    """, unsafe_allow_html=True)

with k2:
    bg = "#f39c12"
    disp = "N/A" if top_state_name is None else f"{top_state_name}: {format_myr(top_state_val)}"
    st.markdown(f"""
        <div style='background:{bg}; {kpi_style}'>
            <div style='display:flex; flex-direction:column;'>
                <div style='font-weight:700; font-size:18px'>{disp}</div>
                <div style='font-size:13px; opacity:0.9'>
                    Top State ({'GDP' if gdp_type=='gdp1' else 'GDP per capita'}) in {'2023' if year_mode=='range' else year_single}
                </div>
            </div>
            <i class="fas fa-award" style="{icon_style}"></i>
        </div>
    """, unsafe_allow_html=True)

with k3:
    bg = "#2980b9"
    disp = "N/A" if top_sector_name is None else f"{top_sector_name}: {format_myr(top_sector_val)}"
    st.markdown(f"""
        <div style='background:{bg}; {kpi_style}'>
            <div style='display:flex; flex-direction:column;'>
                <div style='font-weight:700; font-size:18px'>{disp}</div>
                <div style='font-size:13px; opacity:0.9'>
                    Top Sector ({'GDP' if gdp_type=='gdp1' else 'GDP per capita'}) in {'2023' if year_mode=='range' else year_single}
                </div>
            </div>
            <i class="fas fa-industry" style="{icon_style}"></i>
        </div>
    """, unsafe_allow_html=True)

with k4:
    bg = "#2ecc71" if (growth_pct is None or growth_pct>=0) else "#e74c3c"
    disp = f"{round(growth_pct,2)}%" if growth_pct is not None else "N/A"
    icon = "fa-arrow-up" if (growth_pct is None or growth_pct >= 0) else "fa-arrow-down"
    st.markdown(f"""
        <div style='background:{bg}; {kpi_style}'>
            <div style='display:flex; flex-direction:column;'>
                <div style='font-weight:700; font-size:20px'>{disp}</div>
                <div style='font-size:13px; opacity:0.9'>
                    {'GDP' if gdp_type=='gdp1' else 'GDP per capita'} Growth in {'2023' if year_mode=='range' else year_single}
                </div>
            </div>
            <i class="fas {icon}" style="{icon_style}"></i>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------
# First Row Charts
# ----------------------------------
col_a, col_b = st.columns([1.4,1.1])

# Trend Plot over Time
with col_a:
    st.markdown(f"### {('GDP' if gdp_type=='gdp1' else 'GDP per capita')} Trend over Time for {sector_sel} in {state_sel}")
    trend_df = filter_for_range(gdp_df, year_range_for_trend[0], year_range_for_trend[1])
    trend_df = trend_df[(trend_df['state']==state_sel) & (trend_df['sector']==sector_sel)].copy()
    if trend_df.empty:
        st.info("No trend data available for selected combination.")
    else:
        trend_df[gdp_col] = pd.to_numeric(trend_df[gdp_col], errors='coerce')
        min_y = trend_df[gdp_col].min(skipna=True)
        max_y = trend_df[gdp_col].max(skipna=True)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend_df['year'], y=trend_df[gdp_col], mode='lines+markers',
                                       name=f"{'GDP' if gdp_type=='gdp1' else 'GDP per capita'}",
                                       line=dict(color="#3b528b", width=3), marker=dict(color="#EFC000", size=8)))
        fig_trend.add_trace(go.Scatter(x=np.concatenate([trend_df['year'], trend_df['year'][::-1]]),
                                       y=np.concatenate([trend_df[gdp_col], np.repeat(min_y, len(trend_df))]),
                                       fill='toself', fillcolor='rgba(0,115,194,0.18)', line=dict(color='rgba(255,255,255,0)'),
                                       hoverinfo='skip', showlegend=False))
        fig_trend.update_traces(hovertemplate=(
                                    f"<b>Year:</b> %{{x}}<br>"
                                    f"<b>{'GDP' if gdp_type=='gdp1' else 'GDP per capita'}:</b> %{{y}}"
                                    "<extra></extra>"),
                                hoverlabel= dict(
                                    bgcolor="#3b528b",
                                    bordercolor="#EFC000", 
                                    font=dict(color="#EFC000", size=13)))
        fig_trend.update_layout(height=360, margin=dict(t=30,b=10,l=10,r=10),
                                xaxis_title="Year",
                                yaxis_title=f"{('GDP' if gdp_type=='gdp1' else 'GDP per capita')} (MYR Million)")
        st.plotly_chart(fig_trend, use_container_width=True)

# Correlation Scatter Plot
with col_b:
    st.markdown(f"### GDP vs GDP per capita for {sector_sel} in {year_single}")
    scatter_df = df_single_year[(df_single_year['sector']==sector_sel) & (~df_single_year['state'].isin(["Malaysia","Supranational"]))].copy()
    if 'gdp' not in scatter_df.columns or 'gdp_capita' not in scatter_df.columns:
        st.info("Scatter requires 'gdp' and 'gdp_capita' columns in your data.")
    else:
        scatter_df = scatter_df.dropna(subset=['gdp','gdp_capita'])
        if scatter_df.empty:
            st.info("No data for scatter plot.")
        else:
            scatter_df['gdp'] = pd.to_numeric(scatter_df['gdp'], errors='coerce')
            scatter_df['gdp_capita'] = pd.to_numeric(scatter_df['gdp_capita'], errors='coerce')
            fig_scat = px.scatter(scatter_df, x='gdp', y='gdp_capita', text='state',
                                  labels={'gdp':'GDP (MYR Million)', 'gdp_capita':'GDP per capita (MYR Million)'},
                                  trendline="ols", trendline_color_override="#3b528b")
            fig_scat.update_traces(marker=dict(size=15, opacity=0.8, color="#EFC000"))
            fig_scat.update_traces(hovertemplate="<b>State:</b> %{text}<br><b>GDP:</b> %{x}<br><b>GDP per capita:</b> %{y}<extra></extra>", 
                                   hoverlabel= dict(
                                       bgcolor="#3b528b",
                                       bordercolor="#EFC000",
                                       font=dict(color="#EFC000", size=13)))
            fig_scat.update_traces(selector=dict(mode="lines"),
                                   hovertemplate="<b>GDP:</b> %{x}<br><b>GDP per capita:</b> %{y}<extra></extra>", 
                                   hoverlabel= dict(
                                       bgcolor="#3b528b",
                                       bordercolor="#EFC000",
                                       font=dict(color="#EFC000", size=13)))
            fig_scat.update_layout(height=360, margin=dict(t=30,b=10,l=10,r=10))
            st.plotly_chart(fig_scat, use_container_width=True)

st.markdown("---")

# ----------------------------------
# Second Row Charts
# ----------------------------------
# Choropleth Map
st.markdown(f"### {('GDP' if gdp_type=='gdp1' else 'GDP per capita')} Choropleth Map of Malaysia in {year_single}")
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
try:
    # Prepare data for mapping
    map_data = df_single_year[
        (df_single_year['sector']==sector_sel) &
        (~df_single_year['state'].isin(["Malaysia","Supranational"]))
    ][['state', gdp_col]].copy()

    # Merge shapefile with current year GDP data
    map_df = malaysia_states.merge(map_data, left_on='shapeName', right_on='state', how='left')

    geojson_obj = json.loads(map_df.to_json())
    plot_df = map_df[['shapeName', gdp_col]].rename(
        columns={'shapeName':'region', gdp_col:'gdp_val'}
    )
    fig_map = px.choropleth_mapbox(
        plot_df,
        geojson=geojson_obj,
        locations="region",
        color="gdp_val",
        featureidkey="properties.shapeName",
        mapbox_style="carto-darkmatter",
        zoom=5.0,
        center={"lat":4.1, "lon":109.2},
        opacity=0.4,
        color_continuous_scale="Viridis",
        labels={'gdp_val': f"{('GDP' if gdp_type=='gdp1' else 'GDP per capita')} (MYR Million)"}
    )
    fig_map.update_geos(
        fitbounds="geojson", 
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showland=False
    )
    fig_map.update_traces(
        marker_line_width=1,
        marker_line_color="white",
        hovertemplate="<b>%{location}</b><br><b>GDP:</b> %{z}<extra></extra>",
        hoverlabel=dict(
            bgcolor="rgba(255,0,0,0.2)", 
            bordercolor="red",
            font_color="black"
        )
    )
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=[
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="grey", width=2),
                fillcolor="rgba(0,0,0,0)"
            )
        ]
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})
except Exception as e:
    st.error(f"Error rendering choropleth map: {e}")

st.markdown("---")

# ----------------------------------
# Third Row Charts
# ----------------------------------
col_c, col_d = st.columns([0.9,0.6])

# Donut Chart
with col_c:
    st.markdown(f"### {('GDP' if gdp_type=='gdp1' else 'GDP per capita')} Contribution by Sector for {state_sel} in {year_single}")
    df_donut = df_single_year[(df_single_year['state']==state_sel) & (df_single_year['sector']!="S0")].copy()
    if df_donut.empty:
        st.info("No data for pie chart.")
    else:
        df_donut[gdp_col] = pd.to_numeric(df_donut[gdp_col], errors='coerce')
        df_donut = df_donut.dropna(subset=[gdp_col])
        df_donut['percentage'] = (df_donut[gdp_col] / df_donut[gdp_col].sum()) * 100
        labels = df_donut['sector'].astype(str) + ": " + df_donut['percentage'].round(1).astype(str) + "%"
        fig_donut = go.Figure(data=[go.Pie(labels=df_donut['sector'], values=df_donut[gdp_col], hole=.48, 
                                           domain=dict(x=[0.225, 0.575], y=[0.10, 1]), 
                                           marker=dict(colors=px.colors.sequential.Viridis[:len(df_donut)]),
                                           hovertemplate=(
                                               "<b>Sector:</b> %{label}<br>"
                                               f"<b>{'GDP' if gdp_type=='gdp1' else 'GDP per capita'}:</b> %{{value}}<br>"
                                               "<b>Percentage:</b> %{percent}"
                                               "<extra></extra>"))])
        fig_donut.update_traces(hoverlabel=dict(bgcolor="rgba(68,1,84,0.85)", bordercolor="white", font=dict(color="white", size=13)))
        fig_donut.update_layout(margin=dict(t=30,b=10,l=10,r=10), height=340, legend=dict(orientation="v", x=0.75, xanchor="left"))
        st.plotly_chart(fig_donut, use_container_width=True)

# Bar Chart
with col_d:
    st.markdown(f"### {('GDP' if gdp_type=='gdp1' else 'GDP per capita')} by State for {sector_sel} in {year_single}")
    df_bar = df_single_year[(df_single_year['sector']==sector_sel) & (~df_single_year['state'].isin(["Malaysia","Supranational"]))].copy()
    if df_bar.empty:
        st.info("No data for bar chart.")
    else:
        df_bar[gdp_col] = pd.to_numeric(df_bar[gdp_col], errors='coerce')
        df_bar = df_bar.sort_values(by=gdp_col, ascending=False)
        fig_bar = px.bar(df_bar, x='state', y=gdp_col, labels={gdp_col: f"{('GDP' if gdp_type=='gdp1' else 'GDP per capita')} (MYR Million)", 'state':'State'})
        fig_bar.update_traces(marker_color="#3b528b",
                              hovertemplate=(
                                    f"<b>State:</b> %{{x}}<br>"
                                    f"<b>{'GDP' if gdp_type=='gdp1' else 'GDP per capita'}:</b> %{{y}}"
                                    "<extra></extra>"),
                              hoverlabel=dict(
                                  bgcolor="rgba(59,82,139,0.85)",
                                  bordercolor="white",
                                  font=dict(color="white", size=13)))
        fig_bar.update_layout(xaxis_tickangle=-45, height=340, margin=dict(t=30,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.markdown(
    "Data Source: [Department of Statistics Malaysia (DOSM)](https://open.dosm.gov.my/data-catalogue?search=gdp), 2015–2023"
)

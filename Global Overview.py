import streamlit as st
from utils.utils import load_data, get_dataset_names
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Life Quality Dashboard",
    page_icon="🏠",
    layout="wide"
)

# -- Custom CSS Styling --
st.markdown("""
    <style>
    .kpi-card {
        background-color: white;
        padding: 12px 24px;
        border-radius: 10px;
        text-align: center;
        color: black;
        font-family: "Segoe UI", sans-serif;
        box-shadow: 0px 10px 15px -3px rgba(0,0,0,0.1);
        margin-top: 32px;
        margin-bottom: 22px;
    }

    .kpi-value {
        font-size: 24px;
        font-weight: bold;
    }

    .kpi-delta {
        font-size: 16px;
        color: black;
    }

    .kpi-title {
        font-size: 16px;
        margin-bottom: 8px;
        color: black;
    }

    .block-container {
        padding: 1.5rem 1rem 1rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -- Sidebar --
with st.sidebar:
    st.header("Choose factor to visualise")
    selected_dataset = st.sidebar.selectbox(
        "Choose Dataset",
        get_dataset_names(),
        index=0
    )

# -- Load Data --
df = load_data(selected_dataset)
df = df[df["Year"] >= 1980]

# -- Dataset Column Mapping --
DATA_PLOT_MAPPING = {
    "air_quality": "Air Quality",
    "gdp_capita": "GDP Per Capita",
    "life_expectancy": "life expectancy",
    "unemployment_rate": "Unemployment"
}

data_plot = DATA_PLOT_MAPPING.get(selected_dataset, df.columns[-1])

# -- KPI Cards --
df_air = load_data("air_quality")
df_air = df_air[df_air["Year"] >= 1980].dropna(subset=["Year", "Air Quality"])

df_gdp = load_data("gdp_capita")
df_gdp = df_gdp[df_gdp["Year"] >= 1980].dropna(subset=["Year", "GDP Per Capita"])

df_life = load_data("life_expectancy")
df_life = df_life[df_life["Year"] >= 1980].dropna(subset=["Year", "life expectancy"])

df_unemp = load_data("unemployment_rate")
df_unemp = df_unemp[df_unemp["Year"] >= 1980].dropna(subset=["Year", "Unemployment"])

# Compute metrics
latest_year_air = df_air["Year"].max()
prev_year_air = latest_year_air - 1
avg_pm25 = df_air[df_air["Year"] == latest_year_air]["Air Quality"].mean()
prev_pm25 = df_air[df_air["Year"] == prev_year_air]["Air Quality"].mean()
delta_pm25 = avg_pm25 - prev_pm25

latest_year_gdp = df_gdp["Year"].max()
prev_year_gdp = latest_year_gdp - 1
gdp_avg = df_gdp[df_gdp["Year"] == latest_year_gdp]["GDP Per Capita"].mean()
gdp_prev = df_gdp[df_gdp["Year"] == prev_year_gdp]["GDP Per Capita"].mean()
gdp_delta = ((gdp_avg - gdp_prev) / gdp_prev) * 100

latest_year_life = df_life["Year"].max()
prev_year_life = latest_year_life - 1
life_exp = df_life[df_life["Year"] == latest_year_life]["life expectancy"].mean()
life_exp_prev = df_life[df_life["Year"] == prev_year_life]["life expectancy"].mean()
life_exp_delta = life_exp - life_exp_prev

latest_year_unemp = df_unemp["Year"].max()
prev_year_unemp = latest_year_unemp - 1
unemployment = df_unemp[df_unemp["Year"] == latest_year_unemp]["Unemployment"].mean()
unemployment_prev = df_unemp[df_unemp["Year"] == prev_year_unemp]["Unemployment"].mean()
unemployment_delta = unemployment - unemployment_prev

# -- KPI Display --
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🌫️ Avg Air Quality (PM2.5)</div>
        <div class="kpi-value">{avg_pm25:.1f}</div>
        <div class="kpi-delta">{delta_pm25:+.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">💰 GDP per Capita</div>
        <div class="kpi-value">${gdp_avg:,.0f}</div>
        <div class="kpi-delta">{gdp_delta:+.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🧓 Life Expectancy</div>
        <div class="kpi-value">{life_exp:.1f} yrs</div>
        <div class="kpi-delta">{life_exp_delta:+.1f} yrs</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📉 Unemployment Rate</div>
        <div class="kpi-value">{unemployment:.1f}%</div>
        <div class="kpi-delta">{unemployment_delta:+.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# -- Visualization --
lower_bound = df[data_plot].quantile(0.05)
upper_bound = df[data_plot].quantile(0.95)

col = st.columns((4,4), gap='small')

with col[0]:
    if "Year" in df.columns and len(df["Year"].unique()) > 1:
        df = df.sort_values("Year", ascending=True)
        fig = px.choropleth(
            df,
            locations="Entity",
            locationmode="country names",
            color=data_plot,
            animation_frame="Year",
            range_color=[lower_bound, upper_bound],
            color_continuous_scale="RdBu_r",
            title=f"{selected_dataset.replace('_', ' ').title()} Over Time"
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                orientation="h",
                yanchor="bottom",
                y= 1,
                xanchor="center",
                x=0.5,
            ),
            height=600
        )
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500
        fig.layout.sliders[0].currentvalue["prefix"] = "Year: "
        st.plotly_chart(fig, use_container_width=True)

with col[1]:
    st.markdown('Trend Over Time')
    selected_countries = st.multiselect(
        'Select Countries to Compare:',
        options=sorted(df['Entity'].unique()),
        default=[df['Entity'].iloc[0]],
        key='country_selector'
    )

    if selected_countries:
        filtered_df = df[df['Entity'].isin(selected_countries)]
        st.line_chart(
            filtered_df,
            x='Year',
            y=data_plot,
            color='Entity',
            use_container_width=True
        )
    else:
        st.warning("Please select at least one country")

# -- Top/Bottom Performers --
st.markdown("### 🌍 Country Rankings by Aspect")

aspects = ["air_quality", "gdp_capita", "life_expectancy", "unemployment_rate"]
aspect_names = {
    "air_quality": "Air Quality",
    "gdp_capita": "GDP Per Capita",
    "life_expectancy": "life expectancy",
    "unemployment_rate": "Unemployment"
}
low_is_better = ["air_quality", "unemployment_rate"]

# Filter year options to 1980 onwards
all_years = set()
for aspect in aspects:
    df_temp = load_data(aspect)
    df_temp = df_temp[df_temp["Year"] >= 1980]
    if "Year" in df_temp.columns:
        all_years.update(df_temp["Year"].dropna().unique())

year_options = sorted(all_years)
selected_year = st.selectbox("Select Year:", year_options, index=len(year_options)-1)

top_row = st.columns((1, 1, 1, 1), gap="medium")
bottom_row = st.columns((1, 1, 1, 1), gap="medium")

for i, aspect in enumerate(aspects):
    df_aspect = load_data(aspect)
    df_aspect = df_aspect[df_aspect["Year"] >= 1980]
    column = aspect_names.get(aspect)

    if "Year" in df_aspect.columns:
        df_aspect = df_aspect[df_aspect["Year"] == selected_year]
        display_year = selected_year
    else:
        display_year = "Unknown"

    df_grouped = df_aspect.groupby("Entity", as_index=False)[column].mean()
    ascending = aspect in low_is_better

    df_top = df_grouped.sort_values(column, ascending=ascending).head(5)
    df_top["label"] = df_top.apply(lambda row: f"{row['Entity']}: {row[column]:.2f}", axis=1)

    df_bottom = df_grouped.sort_values(column, ascending=not ascending).head(5)
    df_bottom["label"] = df_bottom.apply(lambda row: f"{row['Entity']}: {row[column]:.2f}", axis=1)

    with top_row[i]:
        fig_top = px.bar(
            df_top,
            x=column,
            y="Entity",
            orientation="h",
            text="label",
            title=f"{aspect_names.get(aspect)} (Top 5 - {display_year})"
        )
        fig_top.update_traces(
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="black")
        )
        fig_top.update_layout(
            yaxis=dict(autorange="reversed", showticklabels=False),
            xaxis=dict(showticklabels=False),
            yaxis_title="",
            xaxis_title="",
            height=300,
            margin=dict(l=0, r=0, t=30, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with bottom_row[i]:
        fig_bottom = px.bar(
            df_bottom,
            x=column,
            y="Entity",
            orientation="h",
            text="label",
            title=f"{aspect_names.get(aspect)} (Bottom 5 - {display_year})"
        )
        fig_bottom.update_traces(
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="black")
        )
        fig_bottom.update_layout(
            yaxis=dict(autorange="reversed", showticklabels=False),
            xaxis=dict(showticklabels=False),
            yaxis_title="",
            xaxis_title="",
            height=300,
            margin=dict(l=0, r=0, t=30, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_bottom, use_container_width=True)

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
st.markdown("""<style>
.kpi-card {background-color: white; padding: 12px 24px; border-radius: 10px; text-align: center; color: black; font-family: "Segoe UI", sans-serif; box-shadow: 0px 10px 15px -3px rgba(0,0,0,0.1); margin-top: 32px; margin-bottom: 22px;}
.kpi-value {font-size: 24px; font-weight: bold;}
.kpi-delta {font-size: 16px; color: black;}
.kpi-title {font-size: 16px; margin-bottom: 8px; color: black;}
.block-container {padding: 1.5rem 1rem 1rem 1rem;}
</style>""", unsafe_allow_html=True)

# -- Sidebar --
with st.sidebar:
    st.header("Choose factor to visualise")
    selected_dataset = st.selectbox("Choose Dataset", get_dataset_names(), index=0)

# -- Load main data --
@st.cache_data
def load_and_filter(dataset):
    df = load_data(dataset)
    return df[df["Year"] >= 1980]

df = load_and_filter(selected_dataset)

# -- Dataset Column Mapping --
DATA_PLOT_MAPPING = {
    "air_quality": "Air Quality",
    "gdp_capita": "GDP Per Capita",
    "life_expectancy": "life expectancy",
    "unemployment_rate": "Unemployment"
}
data_plot = DATA_PLOT_MAPPING.get(selected_dataset, df.columns[-1])

# -- Compute KPIs --
@st.cache_data
def compute_kpis():
    def latest_and_prev(df, col):
        latest = df["Year"].max()
        prev = latest - 1
        curr_val = df[df["Year"] == latest][col].mean()
        prev_val = df[df["Year"] == prev][col].mean()
        return curr_val, curr_val - prev_val, prev_val

    df_air = load_and_filter("air_quality").dropna(subset=["Air Quality"])
    df_gdp = load_and_filter("gdp_capita").dropna(subset=["GDP Per Capita"])
    df_life = load_and_filter("life_expectancy").dropna(subset=["life expectancy"])
    df_unemp = load_and_filter("unemployment_rate").dropna(subset=["Unemployment"])

    avg_pm25, delta_pm25, _ = latest_and_prev(df_air, "Air Quality")
    gdp_avg, gdp_delta_abs, gdp_prev = latest_and_prev(df_gdp, "GDP Per Capita")
    gdp_delta = ((gdp_avg - gdp_prev) / gdp_prev) * 100 if gdp_prev else 0
    life_exp, life_exp_delta, _ = latest_and_prev(df_life, "life expectancy")
    unemployment, unemployment_delta, _ = latest_and_prev(df_unemp, "Unemployment")

    return avg_pm25, delta_pm25, gdp_avg, gdp_delta, life_exp, life_exp_delta, unemployment, unemployment_delta

avg_pm25, delta_pm25, gdp_avg, gdp_delta, life_exp, life_exp_delta, unemployment, unemployment_delta = compute_kpis()

# -- KPI Display --
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">🌫️ Avg Air Quality (PM2.5)</div><div class="kpi-value">{avg_pm25:.1f}</div><div class="kpi-delta">{delta_pm25:+.1f}</div></div>""", unsafe_allow_html=True)
with kpi_cols[1]:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">💰 GDP per Capita</div><div class="kpi-value">${gdp_avg:,.0f}</div><div class="kpi-delta">{gdp_delta:+.1f}%</div></div>""", unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">🧓 Life Expectancy</div><div class="kpi-value">{life_exp:.1f} yrs</div><div class="kpi-delta">{life_exp_delta:+.1f} yrs</div></div>""", unsafe_allow_html=True)
with kpi_cols[3]:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">📉 Unemployment Rate</div><div class="kpi-value">{unemployment:.1f}%</div><div class="kpi-delta">{unemployment_delta:+.1f}%</div></div>""", unsafe_allow_html=True)

# -- Choropleth + Line chart --
lower_bound = df[data_plot].quantile(0.05)
upper_bound = df[data_plot].quantile(0.95)
col = st.columns((4, 4), gap='small')

with col[0]:
    st.markdown(f"#### **{selected_dataset.replace('_', ' ').title()} Over Time**")
    if "Year" in df.columns and len(df["Year"].unique()) > 1:
        df = df.sort_values("Year", ascending=True)
        fig = px.choropleth(
            df,
            locations="Entity",
            locationmode="country names",
            color=data_plot,
            animation_frame="Year",
            range_color=[lower_bound, upper_bound],
            color_continuous_scale="RdBu_r"
        )
        fig.update_layout(
            coloraxis_colorbar=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
            height=600
        )
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500
        fig.layout.sliders[0].currentvalue["prefix"] = "Year: "
        st.plotly_chart(fig, use_container_width=True)

with col[1]:
    st.markdown("#### **Trend Over Time**")
    selected_countries = st.multiselect('Select Countries to Compare:', options=sorted(df['Entity'].unique()), default=[df['Entity'].iloc[0]], key='country_selector')
    if selected_countries:
        filtered_df = df[df['Entity'].isin(selected_countries)]
        st.line_chart(filtered_df, x='Year', y=data_plot, color='Entity', use_container_width=True)
    else:
        st.warning("Please select at least one country")

# -- Rankings --
st.markdown("#### **Rank Chart**")

aspects = ["air_quality", "gdp_capita", "life_expectancy", "unemployment_rate"]
aspect_names = {"air_quality": "Air Quality", "gdp_capita": "GDP Per Capita", "life_expectancy": "life expectancy", "unemployment_rate": "Unemployment"}
low_is_better = ["air_quality", "unemployment_rate"]

@st.cache_data
def compute_rankings(selected_year):
    results = {}
    for aspect in aspects:
        df_aspect = load_and_filter(aspect)
        column = aspect_names.get(aspect)
        df_aspect = df_aspect[df_aspect["Year"] == selected_year]
        df_grouped = df_aspect.groupby("Entity", as_index=False)[column].mean()
        ascending = aspect in low_is_better
        df_top = df_grouped.sort_values(column, ascending=ascending).head(5)
        df_bottom = df_grouped.sort_values(column, ascending=not ascending).head(5)
        results[aspect] = (df_top, df_bottom)
    return results

all_years = sorted({year for aspect in aspects for year in load_and_filter(aspect)["Year"].unique()})
selected_year = st.selectbox("Select Year:", all_years, index=len(all_years)-1)
rankings = compute_rankings(selected_year)

top_row = st.columns((1, 1, 1, 1), gap="medium")
bottom_row = st.columns((1, 1, 1, 1), gap="medium")


for i, aspect in enumerate(aspects):
    df_top, df_bottom = rankings[aspect]
    column = aspect_names[aspect]

    with top_row[i]:
        fig_top = px.bar(df_top.assign(label=lambda d: d.apply(lambda r: f"{r['Entity']}: {r[column]:.2f}", axis=1)),
                         x=column, y="Entity", orientation="h", text="label",
                         title=f"{column} (Top 5 - {selected_year})")
        fig_top.update_traces(textposition="inside", insidetextanchor="start", textfont=dict(color="black"))
        fig_top.update_layout(yaxis=dict(autorange="reversed", showticklabels=False), xaxis=dict(showticklabels=False), height=300, margin=dict(l=0, r=0, t=30, b=20), showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with bottom_row[i]:
        fig_bottom = px.bar(df_bottom.assign(label=lambda d: d.apply(lambda r: f"{r['Entity']}: {r[column]:.2f}", axis=1)),
                            x=column, y="Entity", orientation="h", text="label",
                            title=f"{column} (Bottom 5 - {selected_year})")
        fig_bottom.update_traces(textposition="inside", insidetextanchor="start", textfont=dict(color="black"))
        fig_bottom.update_layout(yaxis=dict(autorange="reversed", showticklabels=False), xaxis=dict(showticklabels=False), height=300, margin=dict(l=0, r=0, t=30, b=20), showlegend=False)
        st.plotly_chart(fig_bottom, use_container_width=True)

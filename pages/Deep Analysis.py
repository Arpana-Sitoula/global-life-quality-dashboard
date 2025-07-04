import pandas as pd
import pycountry
import pycountry_convert as pc
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.utils import load_data
import pydeck as pdk
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_echarts import st_echarts

st.set_page_config(page_title="Global Dashboard", layout="wide")


#---CSS Styling---
st.markdown("""
<style>
.insight-box-left {
    background-color: #f9f9f9;
    padding: 12px 18px;
    border-left: 4px solid #2b83ba;
    border-radius: 6px;
    font-size: 0.95rem;
    color: #333;
    box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
}
.insight-box-right {
    background-color: #f9f9f9;
    padding: 12px 18px;
    border-right: 4px solid #2b83ba;
    border-radius: 6px;
    font-size: 0.95rem;
    color: #333;
    box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


# Helper
def get_continent(country_name):
    try:
        c = pycountry.countries.lookup(country_name)
        code = pc.country_alpha2_to_continent_code(c.alpha_2)
        return {'AF':'Africa','NA':'North America','OC':'Oceania','AN':'Antarctica',
                'AS':'Asia','EU':'Europe','SA':'South America'}.get(code, 'Unknown')
    except:
        return 'Unknown'

@st.cache_data
def build_country_metrics(year):
    df_air = load_data("air_quality")
    df_gdp = load_data("gdp_capita")
    df_life = load_data("life_expectancy")
    df_unemp = load_data("unemployment_rate")
    
    rows = []
    for country in sorted(df_gdp['Entity'].unique()):
        air = df_air[(df_air["Entity"] == country) & (df_air["Year"] == year)]
        gdp = df_gdp[(df_gdp["Entity"] == country) & (df_gdp["Year"] == year)]
        life = df_life[(df_life["Entity"] == country) & (df_life["Year"] == year)]
        unemp = df_unemp[(df_unemp["Entity"] == country) & (df_unemp["Year"] == year)]
        
        row = {
            "Country": country,
            "Continent": get_continent(country),
            "GDP_per_Capita": gdp["GDP Per Capita"].mean() if not gdp.empty else None,
            "Life_Expectancy": life["life expectancy"].mean() if not life.empty else None,
            "Air_Quality_PM25": air["Air Quality"].mean() if not air.empty else None,
            "Unemployment_Rate": unemp["Unemployment"].mean() if not unemp.empty else None
        }
        if all(v is not None for k,v in row.items() if k not in ["Country", "Continent"]):
            rows.append(row)
    return pd.DataFrame(rows)

# Sidebar controls
df_gdp = load_data("gdp_capita")
years = sorted(df_gdp["Year"].unique())
available_countries = sorted(df_gdp["Entity"].unique())

with st.sidebar:
    st.header("📅 Year Selection")
    selected_year = st.selectbox("Select Year", years, index=years.index(2019) if 2019 in years else len(years)-1)

# Build data
df_metrics = build_country_metrics(selected_year)

if df_metrics.empty:
    st.warning("No complete data available for the selected year.")
else:
    st.markdown(f"### Global Analysis for {selected_year} ({len(df_metrics)} countries)")

    # ---- Radar + report ----
        # ---- Radar + report ----
    col1, col2,col3 = st.columns([1.8,0.2,1])
    with col1:
        st.markdown("#### **Radar Comparision**")
        selected_countries = st.multiselect(
            "Select Countries for Radar Comparison", 
            sorted(df_metrics["Country"].unique()), 
            default=['Germany', 'Italy']
        )

        fig_radar = go.Figure()
        def normalize(value, kind):
            ranges = {'GDP per Capita': (0, 100000), 'Life Expectancy': (40, 85),
                      'Air Quality (PM2.5)': (0, 100), 'Unemployment Rate': (0, 30)}
            if value is None: return 0
            min_v, max_v = ranges[kind]
            if kind in ['Air Quality (PM2.5)', 'Unemployment Rate']:
                return 100 - (value - min_v)/(max_v - min_v)*100
            else:
                return (value - min_v)/(max_v - min_v)*100

        for country in selected_countries:
            df_c = df_metrics[df_metrics["Country"] == country]
            if not df_c.empty:
                row = df_c.iloc[0]
                vals = [
                    normalize(row["GDP_per_Capita"], 'GDP per Capita'),
                    normalize(row["Life_Expectancy"], 'Life Expectancy'),
                    normalize(row["Air_Quality_PM25"], 'Air Quality (PM2.5)'),
                    normalize(row["Unemployment_Rate"], 'Unemployment Rate')
                ]
                fig_radar.add_trace(go.Scatterpolar(r=vals, theta=['GDP per Capita', 'Life Expectancy', 'Air Quality', 'Employment'],
                                                    name=country, opacity=0.7))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.title('')
    with col3:
        st.markdown("#### **Radar Summary**")
        
        summary_lines = []
        
        radar_scores = {}
        for country in selected_countries:
            df_c = df_metrics[df_metrics["Country"] == country]
            if not df_c.empty:
                row = df_c.iloc[0]
                vals = [
                    normalize(row["GDP_per_Capita"], 'GDP per Capita'),
                    normalize(row["Life_Expectancy"], 'Life Expectancy'),
                    normalize(row["Air_Quality_PM25"], 'Air Quality (PM2.5)'),
                    normalize(row["Unemployment_Rate"], 'Unemployment Rate')
                ]
                avg_score = np.mean(vals)
                radar_scores[country] = avg_score
        
        if radar_scores:
            best_country = max(radar_scores, key=radar_scores.get)
            best_score = radar_scores[best_country]
            
            # Rank all
            sorted_scores = sorted(radar_scores.items(), key=lambda x: x[1], reverse=True)
            rank_lines = [f"🔹 {country}: {score:.1f}%" for country, score in sorted_scores]
            
            summary_lines.append(f"🏆 {best_country} leads with an overall performance of {best_score:.1f}% across all categories.")
            summary_lines.extend(rank_lines)
        else:
            summary_lines.append("No data available for selected countries.")

        # Display nicely
        st.markdown("""
        <div class="insight-box-left">
        {}
        </div>
        """.format("<br>".join(summary_lines)), unsafe_allow_html=True)

        from PIL import Image
        img = Image.open("arrow1.png")
        rotated = img.rotate(-90, expand=True)

        st.markdown("<div style='margin-top: 120px;'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(rotated, width=220)

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)



    # ---- Correlation + report ----
    col3, col4,col5 = st.columns([1,0.2,1.8])
    with col5:
        st.markdown("#### **Correlation Heatmap**")
        corr = df_metrics[['GDP_per_Capita', 'Life_Expectancy', 'Air_Quality_PM25', 'Unemployment_Rate']].corr()
        fig_heatmap = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                           colorscale='RdBu', zmid=0, text=corr.round(2).values,
                                           texttemplate="%{text}"))
        fig_heatmap.update_layout( height=500, margin=dict(t=50))
        st.plotly_chart(fig_heatmap, use_container_width=True)

        
    with col4:
        st.title('')
    with col3:
        st.markdown("#### **Correlation Summary**")
        
        # Identify strongest positive and negative correlation
        corr_flat = corr.where(~np.eye(corr.shape[0], dtype=bool)).unstack().dropna()
        corr_flat = corr_flat.reset_index()
        corr_flat.columns = ['Var1', 'Var2', 'Correlation']
        
        strongest = corr_flat.iloc[corr_flat['Correlation'].abs().idxmax()]
        
        summary = (f"📈 Strongest correlation: {strongest['Var1']} & {strongest['Var2']} "
                f"({strongest['Correlation']:.2f})")
        
        st.markdown(f"""
        <div class="insight-box-right">
        {summary}<br>
        Positive means they rise together, negative means one rises as the other falls.
        </div>
    """, unsafe_allow_html=True)
        
        from PIL import Image
        img = Image.open("arrow2.png")
        rotated = img.rotate(-90, expand=True)

        st.markdown("<div style='margin-top: 120px;'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(rotated, width=210)

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)



    st.markdown('---')
    # ---- Bubble + mini heatmap ----
    st.markdown("#### **Bubble Chart + Mini Heatmap**")
    selected_continent = st.selectbox("Select Continent for bubble chart:", sorted(df_metrics["Continent"].unique()))
    df_cont = df_metrics[df_metrics["Continent"] == selected_continent]

    xg, yg = df_metrics["GDP_per_Capita"], df_metrics["Life_Expectancy"]
    xc, yc = df_cont["GDP_per_Capita"], df_cont["Life_Expectancy"]

    fig_bubble = go.Figure()

    # Main bubbles for countries
    fig_bubble.add_trace(go.Scatter(
        x=xc, y=yc, mode='markers+text',
        name=f"{selected_continent} Countries",  
        text=df_cont["Country"], textposition='top center',
        marker=dict(
            size=np.clip(100 - df_cont["Air_Quality_PM25"], 5, 30),
            color=df_cont["Unemployment_Rate"],
            colorscale='Plasma',
            colorbar=dict(title="Unemployment Rate (%)"),
            opacity=0.85
        )
    ))

    # Average point
    fig_bubble.add_trace(go.Scatter(
        x=[xc.mean()], y=[yc.mean()], mode='markers+text',
        name="Continent Average",  
        marker=dict(
            symbol='star',
            size=30,
            color='gold',
            line=dict(color='black', width=2)
        ),
        text=["Avg"], textposition='bottom right',
        hoverinfo='skip'
    ))

    # Mini heatmap
    heat = go.Histogram2d(
        x=xc, y=yc,
        name="Density Heatmap", 
        xbins=dict(start=xg.min(), end=xg.max(), size=(xg.max()-xg.min())/20),
        ybins=dict(start=yg.min(), end=yg.max(), size=(yg.max()-yg.min())/20),
        colorscale='Hot',
        opacity=0.5,
        showscale=False
    )
    heat.xaxis = 'x2'
    heat.yaxis = 'y2'
    fig_bubble.add_trace(heat)

    # Layout
    fig_bubble.update_layout(
        xaxis=dict(title="GDP per Capita"),
        yaxis=dict(title="Life Expectancy"),
        xaxis2=dict(domain=[0.75,0.95], anchor='y2', range=[xg.min(), xg.max()]),
        yaxis2=dict(domain=[0.75,0.95], anchor='x2', range=[yg.min(), yg.max()]),
        height=700,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.09,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.5)",  
            font=dict(size=10) 
        )
    )

    st.plotly_chart(fig_bubble, use_container_width=True)


st.markdown('---')
st.markdown("####  **Country Classification Sunburst**")
st.markdown("""
<style>
.sticky-note {
    display: inline-block;
    background-color: #fffa8b;
    color: #333;
    padding: 0.5rem 1rem;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    font-size: 0.9rem;
    height: 110px;
    width: 170px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="sticky-note">📌 <b>Tip:</b> Click on a continent or country to zoom. Click center to zoom out.</div>', unsafe_allow_html=True)


# Compute classes
median_gdp = df_metrics["GDP_per_Capita"].median()
median_life = df_metrics["Life_Expectancy"].median()
df_metrics["GDP_Class"] = df_metrics["GDP_per_Capita"].apply(lambda x: "High GDP" if x >= median_gdp else "Low GDP")
df_metrics["Life_Class"] = df_metrics["Life_Expectancy"].apply(lambda x: "High Life Exp" if x >= median_life else "Low Life Exp")

labels = []
parents = []
values = []

for _, row in df_metrics.iterrows():
    continent = row["Continent"]
    gdp_class = f"{continent} - {row['GDP_Class']}"
    life_class = f"{gdp_class} - {row['Life_Class']}"
    country = row["Country"]

    if continent not in labels:
        labels.append(continent)
        parents.append("")
        values.append(0)

    if gdp_class not in labels:
        labels.append(gdp_class)
        parents.append(continent)
        values.append(0)

    if life_class not in labels:
        labels.append(life_class)
        parents.append(gdp_class)
        values.append(0)

    labels.append(country)
    parents.append(life_class)
    values.append(row["GDP_per_Capita"])

fig_sunburst = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="remainder",
    hovertemplate="<b>%{label}</b><br>GDP per Capita: %{value:.0f}<extra></extra>"
))

fig_sunburst.update_layout(
    height=700,
    margin=dict(t=10, l=10, r=10, b=10)
)

st.plotly_chart(fig_sunburst, use_container_width=True)

st.markdown('---')
st.markdown("#### **Combined Similarity Network Map**")

# Load your coordinates
df_coords = pd.read_csv("datasets/longitude-latitude.csv")

# Merge metrics + coordinates
df_merged = df_metrics.merge(
    df_coords[['Country', 'Latitude', 'Longitude']], 
    on="Country", 
    how="inner"
)

@st.cache_data
def compute_similarity(df):
    features = df[['GDP_per_Capita', 'Life_Expectancy', 'Air_Quality_PM25', 'Unemployment_Rate']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    sim_matrix = cosine_similarity(scaled)

    pairs = []
    countries = df["Country"].tolist()
    lats = df["Latitude"].tolist()
    lons = df["Longitude"].tolist()

    for i in range(len(countries)):
        for j in range(i + 1, len(countries)):
            sim = sim_matrix[i, j]
            if sim > 0.95:
                pairs.append({
                    "from": countries[i],
                    "to": countries[j],
                    "similarity": sim,
                    "from_lat": lats[i],
                    "from_lon": lons[i],
                    "to_lat": lats[j],
                    "to_lon": lons[j]
                })
    return pairs

# Compute all pairs once
all_pairs = compute_similarity(df_merged)

# --- UI Controls ---
selected_country = st.selectbox(
    "Country",
    options=[""] + sorted(df_merged["Country"].unique()),
    index=0
)

# Logic: filter if selected, else show all
if selected_country:
    pairs_to_show = [p for p in all_pairs if p["from"] == selected_country or p["to"] == selected_country]
    focus_row = df_merged[df_merged["Country"] == selected_country].iloc[0]
    view_state = pdk.ViewState(latitude=focus_row["Latitude"], longitude=focus_row["Longitude"], zoom=2)
else:
    pairs_to_show = all_pairs
    view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)

# --- Map ---
arc_layer = pdk.Layer(
    "ArcLayer",
    data=pairs_to_show,
    get_source_position='[from_lon, from_lat]',
    get_target_position='[to_lon, to_lat]',
    get_source_color=[0, 0, 200],
    get_target_color=[0, 0, 200],
    width_scale=1,
    width_min_pixels=1,
    pickable=True,
    auto_highlight=True
)

point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_merged,
    get_position='[Longitude, Latitude]',
    get_fill_color=[0, 0, 200],
    get_radius=30000,
    pickable=True
)

r = pdk.Deck(
    layers=[arc_layer, point_layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/light-v9',
    tooltip={
        "html": "<b>{from}</b> → <b>{to}</b><br>Similarity: {similarity}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
)

st.pydeck_chart(r)


# --- Network graph + summary ---
col1, col2 = st.columns([1, 2])

with col2:
    G = nx.Graph()
    for pair in pairs_to_show:
        G.add_edge(pair["from"], pair["to"], weight=pair["similarity"])

    nodes = [{
        "name": node,
        "symbolSize": 10,
        "itemStyle": {"color": "#2b83ba"}
    } for node in G.nodes()]

    edges = [{
        "source": edge[0],
        "target": edge[1],
        "value": f"{edge[2]['weight']:.2f}",
        "lineStyle": {
            "width": 1 + 4 * (edge[2]['weight'] - 0.95) / 0.05,
            "color": "#2b83ba",
            "opacity": 0.7
        }
    } for edge in G.edges(data=True)]

    option = {
        "tooltip": {"formatter": "{b}"},
        "series": [{
            "type": "graph",
            "layout": "force",
            "roam": True,
            "label": {"show": True},
            "force": {"repulsion": 100, "edgeLength": [50, 200]},
            "data": nodes,
            "links": edges,
            "lineStyle": {"opacity": 0.9}
        }]
    }

    st_echarts(option, height="700px", key="network_chart")

with col1:
    st.markdown("#### **Summary**")
    st.markdown(f"**Nodes:** {len(nodes)}")
    st.markdown(f"**Edges:** {len(edges)}")
    summary_df = pd.DataFrame([{
        "From": e["source"],
        "To": e["target"],
        "Similarity": e["value"]
    } for e in edges])
    st.dataframe(summary_df, use_container_width=True, height=200)

import pandas as pd
import pycountry
import streamlit as st
from utils.utils import load_data
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import plotly.graph_objects as go

# Configure Streamlit page to use full width
st.set_page_config(
    page_title="Global Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove default padding and margins for full screen
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
    }
    .country-title {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .flag-title {
        font-size: 2.5rem;
        margin-left: 1rem;
    }
    .year-subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 2rem;
        font-weight: bold;
        color: #34495e;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -- Sidebar for country and year selection --
with st.sidebar:
    st.header("🌍 Country Selection")
    
    # Get all available countries
    df_sample = load_data("gdp_capita")
    available_countries = sorted(df_sample['Entity'].unique())
    
    selected_country = st.selectbox(
        "Select Country:",
        available_countries,
        index=available_countries.index('Germany') if 'Germany' in available_countries else 0
    )
    
    st.header("📅 Year Selection")
    
    # Find common years across all datasets
    datasets = ["air_quality", "gdp_capita", "life_expectancy", "unemployment_rate"]
    common_years = None
    
    for dataset in datasets:
        df = load_data(dataset)
        country_years = set(df[df["Entity"] == selected_country]["Year"].unique())
        if common_years is None:
            common_years = country_years
        else:
            common_years = common_years.intersection(country_years)
    
    if common_years:
        year_options = sorted(common_years)
    else:
        year_options = sorted(set(df_sample["Year"].unique()))
    
    selected_year = st.selectbox(
        "Select Year:",
        year_options,
        index=len(year_options)-1 if year_options else 0
    )

# -- Function to get country flag emoji --
def get_country_flag(country_name):
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        alpha2 = country.alpha_2.upper()
        return ''.join([chr(ord(c) + 127397) for c in alpha2])
    except:
        return "🌍"

# Get country flag
country_flag = get_country_flag(selected_country)

# -- Page Header --
st.markdown(
    f'<div class="country-title">'
    f'{selected_country}'
    f'<span class="flag-title">{country_flag}</span>'
    f'</div>',
    unsafe_allow_html=True
)
st.markdown(f'<div class="year-subtitle">Economic and Social Indicators • {selected_year}</div>', unsafe_allow_html=True)

# -- Function to get country metrics --
def get_country_metrics(country, year):
    metrics = {}
    
    # Air Quality
    df_air = load_data("air_quality")
    df_air = df_air[(df_air["Entity"] == country) & (df_air["Year"] == year)]
    metrics["Air Quality (PM2.5)"] = df_air["Air Quality"].mean() if not df_air.empty else None
    
    # GDP per Capita
    df_gdp = load_data("gdp_capita")
    df_gdp = df_gdp[(df_gdp["Entity"] == country) & (df_gdp["Year"] == year)]
    metrics["GDP per Capita"] = df_gdp["GDP Per Capita"].mean() if not df_gdp.empty else None
    
    # Life Expectancy
    df_life = load_data("life_expectancy")
    df_life = df_life[(df_life["Entity"] == country) & (df_life["Year"] == year)]
    metrics["Life Expectancy"] = df_life["life expectancy"].mean() if not df_life.empty else None
    
    # Unemployment Rate
    df_unemp = load_data("unemployment_rate")
    df_unemp = df_unemp[(df_unemp["Entity"] == country) & (df_unemp["Year"] == year)]
    metrics["Unemployment Rate"] = df_unemp["Unemployment"].mean() if not df_unemp.empty else None
    
    return metrics

# Get metrics for selected country and year
country_metrics = get_country_metrics(selected_country, selected_year)

# -- Advanced Country Similarity Network --
st.markdown('<div class="section-title">🌐 Global Country Similarity Network</div>', unsafe_allow_html=True)

# Get comprehensive country data for network analysis
@st.cache_data
def prepare_network_data(year):
    """Prepare comprehensive country data for network analysis"""
    network_data = []
    
    # Get all available countries from all datasets
    all_countries = set()
    datasets = ["air_quality", "gdp_capita", "life_expectancy", "unemployment_rate"]
    
    for dataset in datasets:
        df = load_data(dataset)
        all_countries.update(df['Entity'].unique())
    
    # Filter out non-country entities
    country_filters = [
        'World', 'OECD', 'European Union', 'Africa', 'Asia', 'Europe', 
        'North America', 'South America', 'Oceania', 'Sub-Saharan Africa',
        'Middle East', 'East Asia', 'South Asia', 'Latin America'
    ]
    
    all_countries = [c for c in all_countries if not any(filter_term in c for filter_term in country_filters)]
    all_countries = sorted(all_countries)
    
    for country in all_countries:
        country_data = get_country_metrics(country, year)
        
        # Include countries with at least 2 metrics available
        available_metrics = sum(1 for v in country_data.values() if v is not None)
        if available_metrics >= 2:
            processed_data = {
                'Country': country,
                'GDP': country_data.get("GDP per Capita", 25000),
                'Life_Exp': country_data.get("Life Expectancy", 72),
                'Air_Quality': country_data.get("Air Quality (PM2.5)", 25),
                'Unemployment': country_data.get("Unemployment Rate", 6),
                'Data_Quality': available_metrics / 4
            }
            network_data.append(processed_data)
    
    return pd.DataFrame(network_data)

# Create network data
df_network = prepare_network_data(selected_year)

if len(df_network) > 10:
    # Network analysis controls
    col_net1, col_net2, col_net3 = st.columns(3)
    
    with col_net1:
        similarity_threshold = st.slider(
            "Connection Sensitivity", 
            min_value=0.1, max_value=0.9, value=0.3, step=0.1,
            help="Lower values = more connections"
        )
    
    with col_net2:
        layout_type = st.selectbox(
            "Layout Style",
            ["Force-directed", "Circular"],
            help="Choose visualization layout"
        )
    
    with col_net3:
        color_metric = st.selectbox(
            "Color by Metric",
            ["Life_Exp", "GDP", "Air_Quality", "Unemployment"],
            help="Metric for node coloring"
        )
    
    # Prepare features for similarity calculation
    features = ['GDP', 'Life_Exp', 'Air_Quality', 'Unemployment']
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_network[features])
    
    # Calculate similarity matrix using correlation
    similarity_matrix = np.zeros((len(df_network), len(df_network)))
    for i in range(len(df_network)):
        for j in range(len(df_network)):
            if i != j:
                try:
                    corr, _ = pearsonr(scaled_features[i], scaled_features[j])
                    similarity_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0
                except:
                    similarity_matrix[i, j] = 0
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for i, row in df_network.iterrows():
        G.add_node(row['Country'], 
                  gdp=row['GDP'],
                  life_exp=row['Life_Exp'],
                  air_quality=row['Air_Quality'],
                  unemployment=row['Unemployment'])
    
    # Add edges based on similarity threshold
    edge_count = 0
    for i in range(len(df_network)):
        for j in range(i+1, len(df_network)):
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                G.add_edge(df_network.iloc[i]['Country'], 
                          df_network.iloc[j]['Country'],
                          weight=similarity)
                edge_count += 1
    
    # Choose layout
    if layout_type == "Circular":
        pos = nx.circular_layout(G, scale=2)
    else:  # Force-directed
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    # Prepare data for Plotly visualization
    edge_x, edge_y = [], []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Prepare node data
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node attributes
        attrs = G.nodes[node]
        gdp = attrs['gdp']
        life_exp = attrs['life_exp']
        air_qual = attrs['air_quality']
        unemployment = attrs['unemployment']
        
        # Create hover text
        node_text.append(
            f"<b>{node}</b><br>" +
            f"GDP per Capita: ${gdp:,.0f}<br>" +
            f"Life Expectancy: {life_exp:.1f} years<br>" +
            f"Air Quality: {air_qual:.1f} μg/m³<br>" +
            f"Unemployment: {unemployment:.1f}%<br>" +
            f"Connections: {len(list(G.neighbors(node)))}"
        )
        
        # Node size based on GDP
        node_size.append(max(15, min(50, gdp/2000)))
        
        # Node color based on selected metric
        if color_metric == "Life_Exp":
            node_color.append(life_exp)
        elif color_metric == "GDP":
            node_color.append(gdp)
        elif color_metric == "Air_Quality":
            node_color.append(air_qual)
        else:  # Unemployment
            node_color.append(unemployment)
    
    # Create the network visualization
    fig_network = go.Figure()
    
    # Add edges
    if edge_x and edge_y:
        fig_network.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125,125,125,0.4)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
    
    # Add nodes
    if node_x and node_y:
        fig_network.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[country[:3].upper() if len(country) > 10 else country for country in G.nodes()],
            textposition="middle center",
            textfont=dict(size=8, color="white", family="Arial Black"),
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_metric.replace('_', ' ').title()),
                line=dict(width=2, color='rgba(255,255,255,0.8)'),
                opacity=0.8
            ),
            showlegend=False
        ))
    
    # Update layout for full width
    fig_network.update_layout(
        title=dict(
            text=f"Global Country Similarity Network - {selected_year}<br>" +
                 f"<sub>{len(G.nodes())} countries, {edge_count} connections</sub>",
            x=0.5,
            font=dict(size=16, color="#2c3e50")
        ),
        font_size=12,
        showlegend=False,
        height=700,
        plot_bgcolor='rgba(240,248,255,0.8)',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=80, b=10)
    )
    
    st.plotly_chart(fig_network, use_container_width=True)
    
    # Network insights
    col_insight1, col_insight2, col_insight3 = st.columns(3)
    
    with col_insight1:
        density = nx.density(G) if len(G.nodes()) > 1 else 0
        st.metric(
            "🔗 Network Density", 
            f"{density:.2%}",
            help="Percentage of possible connections that exist"
        )
    
    with col_insight2:
        if G.nodes():
            most_connected = max(G.nodes(), key=lambda x: len(list(G.neighbors(x))))
            st.metric(
                "🌟 Most Connected", 
                most_connected,
                delta=f"{len(list(G.neighbors(most_connected)))} connections"
            )
        else:
            st.metric("🌟 Most Connected", "N/A")
    
    with col_insight3:
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            st.metric(
                "🎯 Communities", 
                len(communities),
                help="Number of distinct country clusters"
            )
        except:
            st.metric("🎯 Communities", "1")

else:
    st.warning(f"Insufficient data for network analysis. Found data for {len(df_network)} countries.")

# -- Country Comparison Radar Chart --
st.markdown('<div class="section-title">📊 Country Comparison</div>', unsafe_allow_html=True)

col6, col7 = st.columns([1, 1])

with col6:
    st.markdown("**📊 Country Comparison Radar Chart**")
    
    # Country selection for comparison
    comparison_countries = st.multiselect(
        "Select countries to compare:",
        available_countries,
        default=[selected_country] + [c for c in ['United States', 'China', 'Japan'] 
                                    if c in available_countries and c != selected_country][:3],
        max_selections=5
    )
    
    if comparison_countries:
        # Create radar chart
        fig_radar = go.Figure()
        
        # Normalize metrics for radar chart
        def normalize_metric(value, metric_type):
            ranges = {
                'GDP per Capita': (0, 100000),
                'Life Expectancy': (40, 85),
                'Air Quality (PM2.5)': (0, 100),
                'Unemployment Rate': (0, 30)
            }
            
            if value is None:
                return 0
            
            min_val, max_val = ranges[metric_type]
            
            # Invert for metrics where lower is better
            if metric_type in ['Air Quality (PM2.5)', 'Unemployment Rate']:
                normalized = 100 - ((value - min_val) / (max_val - min_val) * 100)
            else:
                normalized = (value - min_val) / (max_val - min_val) * 100
            
            return max(0, min(100, normalized))
        
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Add trace for each country
        for idx, country in enumerate(comparison_countries):
            country_data = get_country_metrics(country, selected_year)

            categories = ['GDP per Capita', 'Life Expectancy', 'Air Quality', 'Employment']
            values = [
                normalize_metric(country_data.get("GDP per Capita"), 'GDP per Capita'),
                normalize_metric(country_data.get("Life Expectancy"), 'Life Expectancy'),
                normalize_metric(country_data.get("Air Quality (PM2.5)"), 'Air Quality (PM2.5)'),
                normalize_metric(country_data.get("Unemployment Rate"), 'Unemployment Rate')
            ]

            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=country,
                line=dict(color=color_palette[idx % len(color_palette)], width=3),
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickmode='linear',
                    tick0=0,
                    dtick=20,
                )
            ),
            showlegend=True,
            height=500,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Please select at least one country for comparison.")

with col7:
    st.markdown("**🔥 Correlation Heatmap**")
    
    # Prepare data for correlation analysis
    correlation_data = []
    
    # Get data for all countries for the selected year
    for country in available_countries:
        country_data = get_country_metrics(country, selected_year)
        
        # Only include countries with complete data
        if all(v is not None for v in country_data.values()):
            correlation_data.append({
                'Country': country,
                'GDP_per_Capita': country_data["GDP per Capita"],
                'Life_Expectancy': country_data["Life Expectancy"],
                'Air_Quality_PM25': country_data["Air Quality (PM2.5)"],
                'Unemployment_Rate': country_data["Unemployment Rate"]
            })
    
    if len(correlation_data) > 5:
        df_corr = pd.DataFrame(correlation_data)
        
        # Calculate correlation matrix
        numeric_cols = ['GDP_per_Capita', 'Life_Expectancy', 'Air_Quality_PM25', 'Unemployment_Rate']
        correlation_matrix = df_corr[numeric_cols].corr()
        
        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=['GDP per Capita', 'Life Expectancy', 'Air Quality (PM2.5)', 'Unemployment Rate'],
            y=['GDP per Capita', 'Life Expectancy', 'Air Quality (PM2.5)', 'Unemployment Rate'],
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            showscale=True
        ))
        
        fig_heatmap.update_layout(
            title="Attribute Correlations",
            height=500,
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info(f"Insufficient data for correlation analysis. Need data from at least 6 countries, found {len(correlation_data)}.")
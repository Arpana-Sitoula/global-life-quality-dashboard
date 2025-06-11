import streamlit as st
from utils.utils import load_data
import plotly.graph_objects as go
import pandas as pd
import pycountry
import emoji  # For flag emojis

st.set_page_config(
    page_title="Country Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# -- Clean CSS styling --
st.markdown("""
    <style>
    /* Main container */
    .stApp {
        background-color: #f8f9fa;
    }
     
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #e9ecef;
    }
    
    .metric-row:last-child {
        border-bottom: none;
    }
    
    .metric-label {
        font-weight: 600;
        color: #495057;
        font-size: 16px;
    }
    
    .metric-value {
        font-weight: 700;
        color: #212529;
        font-size: 18px;
    }
    
    /* Titles */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1.5rem;
    }
    
    .country-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 0.2rem;
    }
    
    .year-subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    .flag-title {
        font-size: 2.5rem;
        vertical-align: middle;
        margin-right: 15px;
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
        # Fallback if no common years found
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

# -- Main Layout --
col1, col2,col3 = st.columns([2.2, 0.6,1.2], gap="large")

with col1:
    # Get country ISO code
    try:
        country_iso = pycountry.countries.search_fuzzy(selected_country)[0].alpha_3
    except:
        country_iso = "DEU"  # Default to Germany if country not found
    
    # Create minimal country-only map
    fig = go.Figure()
    
    # Add country outline - this ensures ONLY the country is shown
    fig.add_trace(go.Choropleth(
        locations=[country_iso],
        z=[1],  # Single value to color the country
        locationmode='ISO-3',
        colorscale=[[0, '#e6f3ff'], [1, '#1e88e5']],  # Simple blue color
        showscale=False,
        hoverinfo='none',
        marker_line_color='#0d47a1',
        marker_line_width=1.5
    ))
    
    # Add metric labels with connecting lines
    annotations = []
    shapes = []  # For drawing lines
    y_positions = [0.85, 0.65, 0.45, 0.25]  # Vertical positions for labels
    
    # Country center point (approximate for line connection)
    country_center_x = 0.25  # Adjust based on your country's position
    country_center_y = 0.55  # Adjust based on your country's position
    
    metrics_list = [
        ("GDP per Capita", "💰", country_metrics.get("GDP per Capita"), "${:,.0f}"),
        ("Life Expectancy", "🧓", country_metrics.get("Life Expectancy"), "{:.1f} yrs"),
        ("Air Quality", "🌫️", country_metrics.get("Air Quality (PM2.5)"), "{:.1f} μg/m³"),
        ("Unemployment", "📉", country_metrics.get("Unemployment Rate"), "{:.1f}%")
    ]
    
    for i, (name, icon, value, fmt) in enumerate(metrics_list):
        if value is not None:
            label_x = 1  # Position for labels (right side)
            label_y = y_positions[i]
            
            # Add connecting line from country to label
            shapes.append(dict(
                type="line",
                x0=country_center_x, y0=country_center_y,
                x1=label_x - 0, y1=label_y,
                xref='paper', yref='paper',
                line=dict(
                    color="rgba(30, 136, 229, 0.6)",
                    width=1.5,
                    dash="dot"
                )
            ))
            
            # Add label with background
            annotations.append(dict(
                x=label_x,
                y=label_y,
                xref='paper',
                yref='paper',
                text=f"<b>{icon} {name}:</b> {fmt.format(value)}",
                showarrow=False,
                align='left',
                font=dict(size=14, color="#333"),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(30, 136, 229, 0.3)",
                borderwidth=1,
                borderpad=8
            ))
    
    # Update layout for the map - position it to the left
    fig.update_geos(
        visible=False,
        resolution=50,
        showcountries=False,  
        showocean=False,
        showland=False,
        showframe=False,
        fitbounds="locations",
        projection_type="mercator",
        projection_scale=3,
        # Position the map to the left side
        center=dict(lat=0, lon=-70)  # Adjust longitude to shift left
    )
    
    fig.update_layout(
        annotations=annotations,
        shapes=shapes,  # Add the connecting lines
        margin=dict(l=0, r=0, t=0, b=0),  # Extra right margin for labels
        height=400,
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            subunitwidth=1,
            # Position the geo plot to the left
            domain=dict(x=[0, 0.7], y=[0, 1])  # Map takes left 70% of space
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Display in a container
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.title("")

with col3:
    st.markdown("**📌 Quick Insights**")
    
    insights = []
    
    if country_metrics.get("GDP per Capita") is not None:
        gdp = country_metrics["GDP per Capita"]
        if gdp > 50000:
            insights.append("💎 High-income economy")
        elif gdp > 25000:
            insights.append("🏗️ Developed economy")
        else:
            insights.append("🌱 Developing economy")
    
    if country_metrics.get("Unemployment Rate") is not None:
        unemp = country_metrics["Unemployment Rate"]
        if unemp > 10:
            insights.append("🔴 High unemployment")
        elif unemp > 5:
            insights.append("🟡 Moderate unemployment")
        else:
            insights.append("🟢 Low unemployment")
    
    if country_metrics.get("Air Quality (PM2.5)") is not None:
        air = country_metrics["Air Quality (PM2.5)"]
        if air > 35:
            insights.append("⚠️ Poor air quality")
        elif air < 15:
            insights.append("🌿 Good air quality")
    
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.markdown("- No insights available")

# Second Row: Basic Visualisation
col4, col5 = st.columns([1,1], gap="large")

with col4:
    st.markdown("**🎯 Performance Score**")

    # Calculate comprehensive performance score
    if country_metrics:
        scores = {}
        
        # Normalize each metric to 0-100 scale
        def calculate_score(value, metric_type, all_values):
            if value is None or not all_values:
                return 50  # Neutral score
            
            # Remove None values
            clean_values = [v for v in all_values if v is not None]
            if not clean_values:
                return 50
            
            min_val, max_val = min(clean_values), max(clean_values)
            if min_val == max_val:
                return 50
            
            # For metrics where higher is better
            if metric_type in ['GDP per Capita', 'Life Expectancy']:
                score = ((value - min_val) / (max_val - min_val)) * 100
            # For metrics where lower is better
            else:
                score = (1 - (value - min_val) / (max_val - min_val)) * 100
            
            return max(0, min(100, score))
        
        # Get all values for normalization
        all_gdp = [get_country_metrics(c, selected_year).get("GDP per Capita") 
                    for c in available_countries[:20]]
        all_life = [get_country_metrics(c, selected_year).get("Life Expectancy") 
                    for c in available_countries[:20]]
        all_air = [get_country_metrics(c, selected_year).get("Air Quality (PM2.5)") 
                    for c in available_countries[:20]]
        all_unemp = [get_country_metrics(c, selected_year).get("Unemployment Rate") 
                    for c in available_countries[:20]]
        
        # Calculate scores
        gdp_score = calculate_score(country_metrics.get("GDP per Capita"), "GDP per Capita", all_gdp)
        life_score = calculate_score(country_metrics.get("Life Expectancy"), "Life Expectancy", all_life)
        air_score = calculate_score(country_metrics.get("Air Quality (PM2.5)"), "Air Quality (PM2.5)", all_air)
        unemp_score = calculate_score(country_metrics.get("Unemployment Rate"), "Unemployment Rate", all_unemp)
        
        # Overall score (weighted average)
        overall_score = (gdp_score * 0.25 + life_score * 0.25 + air_score * 0.25 + unemp_score * 0.25)
        
        # Create gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{selected_country} Performance Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

with col5: 
    # Score breakdown
    st.markdown("**📊 Score Breakdown:**")
   # Score dictionary
    score_data = {
        '💰 Economic': gdp_score,
        '🧓 Health': life_score,
        '🌫️ Environment': air_score,
        '📉 Employment': unemp_score
    }

    # Extract categories and scores
    categories = list(score_data.keys())
    scores = list(score_data.values())

    # Color by score
    bar_colors = ["#2ecc71" if s > 75 else "#f1c40f" if s > 50 else "#e74c3c" for s in scores]

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=scores,
        y=categories,
        orientation='h',
        marker_color=bar_colors,
        text=[f"{s:.1f}/100" for s in scores],
        textposition='auto'
    ))

    # Styling
    fig.update_layout(
        title="",
        xaxis=dict(title="Score (0-100)", range=[0, 100]),
        yaxis=dict(title="Category"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',      
        paper_bgcolor='white'
    )

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# -- Second Row: Box plot finding outliers
# Add this code after your performance metrics row (after col4 and col5)

# -- Third Row: Box Plot Analysis --
st.markdown("---")
st.markdown('<div class="section-title">📊 Box Plot: Outlier Analysis</div>', unsafe_allow_html=True)

# Collect data for all countries for the selected year
box_plot_data = []

for country in available_countries:
    country_data = get_country_metrics(country, selected_year)
    if any(v is not None for v in country_data.values()):  # Include countries with at least some data
        box_plot_data.append({
            'Country': country,
            'GDP_per_Capita': country_data.get("GDP per Capita"),
            'Life_Expectancy': country_data.get("Life Expectancy"),
            'Air_Quality_PM25': country_data.get("Air Quality (PM2.5)"),
            'Unemployment_Rate': country_data.get("Unemployment Rate")
        })

if len(box_plot_data) > 10:  # Need sufficient data for meaningful box plot
    df_box = pd.DataFrame(box_plot_data)
    
    # Create subplot for box plots
    from plotly.subplots import make_subplots
    
    fig_box = make_subplots(
        rows=2, cols=2,
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # GDP per Capita Box Plot
    gdp_data = df_box['GDP_per_Capita'].dropna()
    fig_box.add_trace(
        go.Box(y=gdp_data, name='GDP', marker_color='#3498db', showlegend=False),
        row=1, col=1
    )
    
    # Add selected country point
    if country_metrics.get("GDP per Capita") is not None:
        fig_box.add_trace(
            go.Scatter(x=[0], y=[country_metrics["GDP per Capita"]], 
                        mode='markers', marker=dict(size=12, color='red', symbol='diamond'),
                        name=f'{selected_country}', showlegend=False),
            row=1, col=1
        )
    
    # Life Expectancy Box Plot
    life_data = df_box['Life_Expectancy'].dropna()
    fig_box.add_trace(
        go.Box(y=life_data, name='Life Exp', marker_color='#2ecc71', showlegend=False),
        row=1, col=2
    )
    
    if country_metrics.get("Life Expectancy") is not None:
        fig_box.add_trace(
            go.Scatter(x=[0], y=[country_metrics["Life Expectancy"]], 
                        mode='markers', marker=dict(size=12, color='red', symbol='diamond'),
                        name=f'{selected_country}', showlegend=False),
            row=1, col=2
        )
    
    # Air Quality Box Plot
    air_data = df_box['Air_Quality_PM25'].dropna()
    fig_box.add_trace(
        go.Box(y=air_data, name='Air Quality', marker_color='#e74c3c', showlegend=False),
        row=2, col=1
    )
    
    if country_metrics.get("Air Quality (PM2.5)") is not None:
        fig_box.add_trace(
            go.Scatter(x=[0], y=[country_metrics["Air Quality (PM2.5)"]], 
                        mode='markers', marker=dict(size=12, color='red', symbol='diamond'),
                        name=f'{selected_country}', showlegend=False),
            row=2, col=1
        )
    
    # Unemployment Rate Box Plot
    unemp_data = df_box['Unemployment_Rate'].dropna()
    fig_box.add_trace(
        go.Box(y=unemp_data, name='Unemployment', marker_color='#f39c12', showlegend=False),
        row=2, col=2
    )
    
    if country_metrics.get("Unemployment Rate") is not None:
        fig_box.add_trace(
            go.Scatter(x=[0], y=[country_metrics["Unemployment Rate"]], 
                        mode='markers', marker=dict(size=12, color='red', symbol='diamond'),
                        name=f'{selected_country}', showlegend=False),
            row=2, col=2
        )
    
    fig_box.update_layout(
        height=500,
        title_text=f" {selected_country} vs World ({selected_year})",
        title_x=0.5,
        showlegend=False
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    
else:
    st.info("Insufficient data for box plot analysis.")


st.markdown("---")
col_trend1, col_trend2 = st.columns([2, 1], gap="large")
# -- Function to get historical trend data --
def get_country_trends(country, start_year=1980):
    """Get historical data for all metrics for a specific country from start_year onwards"""
    trend_data = {
        'years': [],
        'gdp_per_capita': [],
        'life_expectancy': [],
        'air_quality': [],
        'unemployment': []
    }
    
    # Get all available years for this country across all datasets
    all_years = set()
    
    datasets_info = [
        ("gdp_capita", "GDP Per Capita", "gdp_per_capita"),
        ("life_expectancy", "life expectancy", "life_expectancy"),
        ("air_quality", "Air Quality", "air_quality"),
        ("unemployment_rate", "Unemployment", "unemployment")
    ]
    
    # Collect all possible years from start_year onwards
    for dataset_name, _, _ in datasets_info:
        df = load_data(dataset_name)
        country_data = df[df["Entity"] == country]
        years = country_data["Year"].unique()
        # Filter years to only include start_year and onwards
        filtered_years = [year for year in years if year >= start_year]
        all_years.update(filtered_years)
    
    # Sort years
    sorted_years = sorted(all_years)
    
    # Get data for each year
    for year in sorted_years:
        year_metrics = get_country_metrics(country, year)
        
        # Only include years where we have at least some data
        if any(v is not None for v in year_metrics.values()):
            trend_data['years'].append(year)
            trend_data['gdp_per_capita'].append(year_metrics.get("GDP per Capita"))
            trend_data['life_expectancy'].append(year_metrics.get("Life Expectancy"))
            trend_data['air_quality'].append(year_metrics.get("Air Quality (PM2.5)"))
            trend_data['unemployment'].append(year_metrics.get("Unemployment Rate"))
    
    return trend_data

# Function to normalize data to 0-100 scale for trend comparison
def normalize_for_trend(values):
    clean_values = [v for v in values if v is not None]
    if len(clean_values) < 2:
        return values
    
    min_val, max_val = min(clean_values), max(clean_values)
    if min_val == max_val:
        return [50] * len(values)  # Return neutral values if no variation
    
    normalized = []
    for v in values:
        if v is None:
            normalized.append(None)
        else:
            norm_val = ((v - min_val) / (max_val - min_val)) * 100
            normalized.append(norm_val)
    return normalized

# Main trend visualization
st.markdown("**📊 Historical Trends Analysis (1980 onwards)**")

# Get historical data from 1980
trend_data = get_country_trends(selected_country, start_year=1980)

if len(trend_data['years']) > 1:  # Need at least 2 data points for trends
    
    # Create normalized comparison chart
    fig_normalized = go.Figure()
    
    # Colors for each metric
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # Normalize each metric and add to the same chart
    metrics_to_plot = [
        ('gdp_per_capita', 'GDP per Capita', colors[0]),
        ('life_expectancy', 'Life Expectancy', colors[1]),
        ('air_quality', 'Air Quality (Inverted)', colors[2]),  # Inverted because lower is better
        ('unemployment', 'Unemployment Rate (Inverted)', colors[3])  # Inverted because lower is better
    ]
    
    for metric_key, metric_name, color in metrics_to_plot:
        values = trend_data[metric_key]
        if any(v is not None for v in values):
            # For air quality and unemployment, invert the normalization (lower values = higher normalized score)
            if 'Inverted' in metric_name:
                clean_values = [v for v in values if v is not None]
                if len(clean_values) >= 2:
                    min_val, max_val = min(clean_values), max(clean_values)
                    if min_val != max_val:
                        normalized = []
                        for v in values:
                            if v is None:
                                normalized.append(None)
                            else:
                                # Invert: lower original value = higher normalized score
                                norm_val = (1 - (v - min_val) / (max_val - min_val)) * 100
                                normalized.append(norm_val)
                    else:
                        normalized = [50] * len(values)
                else:
                    normalized = values
            else:
                normalized = normalize_for_trend(values)
            
            fig_normalized.add_trace(
                go.Scatter(
                    x=trend_data['years'],
                    y=normalized,
                    mode='lines+markers',
                    name=metric_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6),
                    connectgaps=True
                )
            )
    
    # Add current year line
    fig_normalized.add_vline(
        x=selected_year, 
        line_dash="dash", 
        line_color="red", 
        opacity=0.7,
        annotation_text=f"Current Year: {selected_year}"
    )
    
    fig_normalized.update_layout(
        title=f"{selected_country} - Normalized Trends Comparison (1980-Present)",
        xaxis_title="Year",
        yaxis_title="Normalized Score (0-100)",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_normalized, use_container_width=True)
    
else:
    st.info("Insufficient historical data to show trends. Need at least 2 data points from 1980 onwards.")
import streamlit as st
from utils.utils import load_data, get_dataset_names
import plotly.express as px

st.set_page_config(
    page_title="Life Quality Dashboard",
    page_icon="🏠",
    layout="wide"
)

# Optional: Add a sidebar to all pages
with st.sidebar:
    st.header("Choose factor to visualise")
    selected_dataset = st.sidebar.selectbox(
        "Choose Dataset",
        get_dataset_names(),
        index=0
    )

# Load selected data
df = load_data(selected_dataset)

# Create a mapping dictionary for consistent column names
DATA_PLOT_MAPPING = {
    "air_quality": "FactValueNumeric",
    "crime_rate": "CrimeRate_OverallCriminalityScoreGOCI",
    "gdp_capita": "GDP Per Capita",
    "gender_equality": "EqualityIndex",  # Add actual column name
    "green_space": "GreenSpacePercentage",  # Add actual column name
    "healthy_diet": "Can't afford healthy diet (%)",
    "life_expectancy": "life expectancy",  # Add actual column name
    "literacy_rate": "LiteracyRate_TotalPopulation_Pct",
    "mental_health": "MentalHealthIndex",  # Add actual column name
    "unemployment_rate": "Unemployment"
}

# Get the correct column name for the current dataset
data_plot = DATA_PLOT_MAPPING.get(selected_dataset, df.columns[-1])  # Fallback to last column


# Custom CSS to fine-tune map spacing
st.markdown("""
    <style>
    /* Reduce padding around the main container */
    .block-container {
        padding: 1.5rem 1rem 1rem 1rem;
    }

    /* Optional: also reduce gaps for charts */
    .element-container:has(.js-plotly-plot) {
        margin-top: -1rem;
        margin-bottom: -1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Calculate bounds using full dataset for consistent scaling
lower_bound = df[data_plot].quantile(0.05)
upper_bound = df[data_plot].quantile(0.95)

col = st.columns((4,4), gap='small')

with col[0]:
    if "Year" in df.columns and len(df["Year"].unique()) > 1:
        # Sort dataframe by Year in ascending order
        df = df.sort_values("Year", ascending=True)
        # For datasets with time series data
        fig = px.choropleth(
            df,  # Use full dataset, not filtered data
            locations="Entity",
            locationmode="country names",
            color=data_plot,
            animation_frame="Year",  # This enables automatic animation
            range_color=[lower_bound, upper_bound],
            color_continuous_scale="RdBu_r",
            title=f"{selected_dataset.replace('_', ' ').title()} Over Time"
        )
        fig.update_layout(
        coloraxis_colorbar=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="center",
        x=0.5
    ))
        # Customize animation settings
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500
        fig.layout.sliders[0].currentvalue["prefix"] = "Year: "
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # For single-year or non-temporal data
        fig = px.choropleth(
            df,
            locations="Entity",
            locationmode="country names",
            color=data_plot,
            color_continuous_scale="RdGr_r",
            title=f"{selected_dataset.replace('_', ' ').title()}",
        )
        st.plotly_chart(fig, use_container_width=True)

with col[1]:
        # ----------------------------------------------------
    st.header('Trend Over Time', divider='rainbow')
    selected_countries = st.multiselect(
        'Select Countries to Compare:',
        options=sorted(df['Entity'].unique()),
        default=[df['Entity'].iloc[0]],  # Default to first country
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

col = st.columns((2,2,2,2), gap='small')

with col[0]:
    #choosing only the top 5 one.
    df_grouped = df.groupby('Entity', as_index=False)[data_plot].mean()
    df_top = df_grouped.sort_values(data_plot, ascending=False).head(5)
    fig = px.bar(df_top, x='Entity', y=data_plot,
             title='Country Scores Ranked',
             text= data_plot)

    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title='Entity', yaxis_title=data_plot, showlegend=False)

    fig.show()
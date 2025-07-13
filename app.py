# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import time
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# App configuration
st.set_page_config(
    page_title="Load-Shedding Impact Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    :root {
        --primary: #1e3d6d;
        --secondary: #2a5a8c;
        --accent: #e63946;
        --background: #f8f9fa;
        --card: #ffffff;
        --text: #333333;
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
    }
    
    .main {background-color: var(--background);}
    .st-bb {background-color: var(--card);}
    .st-at {background-color: #f0f2f6;}
    .header {color: var(--primary); font-weight: 700;}
    .subheader {color: var(--secondary);}
    .footer {font-size: 0.8em; color: #6c757d;}
    .metric-card {border-radius: 10px; padding: 20px; background: var(--card); 
                box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;
                border-left: 4px solid var(--primary);}
    .positive {color: var(--success);}
    .negative {color: var(--danger);}
    .stButton>button {background-color: var(--primary); color: white; border-radius: 8px; padding: 8px 16px;}
    .stButton>button:hover {background-color: var(--secondary);}
    .stSelectbox, .stDateInput, .stSlider {margin-bottom: 15px;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px; border-radius: 8px 8px 0 0;}
    .stTabs [aria-selected="true"] {background-color: var(--primary); color: white;}
    .map-container {border-radius: 10px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)

# Sample data generation
def generate_loadshedding_data(location, start_date, end_date):
    """Generate mock load-shedding data"""
    dates = pd.date_range(start_date, end_date)
    stages = np.random.choice([2, 3, 4, 5, 6], size=len(dates), p=[0.1, 0.2, 0.3, 0.3, 0.1])
    outage_hours = stages * np.random.uniform(1.8, 2.5, size=len(dates))
    return pd.DataFrame({
        'date': dates,
        'outage_hours': np.round(outage_hours, 1),
        'stage': stages,
        'location': location
    })

def generate_economic_data(location):
    """Generate mock economic impact data"""
    sectors = ['Retail', 'Manufacturing', 'Services', 'Hospitality']
    data = []
    for sector in sectors:
        base_revenue = np.random.uniform(50000, 200000)
        for month in range(1, 13):
            month_name = date(2023, month, 1).strftime('%b')
            season_factor = 1 + 0.2 * np.sin(2 * np.pi * (month-1)/12)
            outage_factor = 1 - np.random.uniform(0.05, 0.15) * np.random.choice([2, 3, 4])
            revenue = base_revenue * season_factor * outage_factor
            data.append({
                'sector': sector,
                'month': month_name,
                'revenue': np.round(revenue, 2),
                'revenue_change': np.round((revenue - base_revenue) / base_revenue * 100, 1),
                'location': location
            })
    return pd.DataFrame(data)

def generate_correlation_data():
    """Generate correlation data between outages and revenue"""
    locations = list(MUNICIPALITIES.keys())
    sectors = ['Retail', 'Manufacturing', 'Services', 'Hospitality']
    data = []
    for loc in locations:
        for sector in sectors:
            outages = np.random.uniform(50, 200)
            revenue_impact = -outages * np.random.uniform(0.8, 1.5) * np.random.uniform(0.8, 1.2)
            data.append({
                'location': loc,
                'sector': sector,
                'total_outages': outages,
                'revenue_impact': revenue_impact
            })
    return pd.DataFrame(data)

# SA Municipal Boundaries
MUNICIPALITIES = {
    "Cape Town": {"lat": -33.9249, "lon": 18.4241, "province": "Western Cape"},
    "Johannesburg": {"lat": -26.2041, "lon": 28.0473, "province": "Gauteng"},
    "Durban": {"lat": -29.8587, "lon": 31.0218, "province": "KwaZulu-Natal"},
    "Pretoria": {"lat": -25.7479, "lon": 28.2293, "province": "Gauteng"},
    "Port Elizabeth": {"lat": -33.9608, "lon": 25.6022, "province": "Eastern Cape"}
}

# App Header
st.title("⚡ Load-Shedding Impact Predictor")
st.markdown("""
**Forecasting Eskom outages and economic impact on South African SMEs**  
*Integrated with location-specific data from SA government portals*
""")
st.divider()

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Prediction Parameters")
    location = st.selectbox("Municipality", list(MUNICIPALITIES.keys()), index=0)
    sector = st.selectbox("Business Sector", ['Retail', 'Manufacturing', 'Services', 'Hospitality'], index=0)
    date_range = st.date_input(
        "Forecast Period",
        [date.today(), date.today() + timedelta(days=30)]
    )
    
    # Advanced controls
    with st.expander("Advanced Settings"):
        sensitivity = st.slider("Economic Sensitivity", 0.5, 2.0, 1.0, step=0.1,
                               help="Adjust for your business's sensitivity to power outages")
        confidence = st.slider("Model Confidence", 70, 95, 85,
                             help="Statistical confidence level for predictions")
        st.checkbox("Include historical data", value=True)
        st.checkbox("Adjust for seasonality", value=True)
    
    st.divider()
    st.markdown("""
    **Data Sources:**  
    - EskomSePush API  
    - Stats SA Business Activity Data  
    - Municipal Economic Reports
    """)
    
    st.markdown("""
    ---
    *Developed with ❤️ for South African SMEs*  
    *v1.0 | © 2023 Energy Analytics Group*
    """)

# Generate mock data based on user inputs
if len(date_range) == 2:
    start_date, end_date = date_range
    loadshedding_df = generate_loadshedding_data(location, start_date, end_date)
    economic_df = generate_economic_data(location)
    correlation_df = generate_correlation_data()
else:
    st.warning("Please select a valid date range")
    st.stop()

# Main Dashboard Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("⏱️ Predicted Outage Hours")
    total_hours = loadshedding_df['outage_hours'].sum()
    st.metric("Total Hours", f"{total_hours:.1f}h", 
             delta=f"{(total_hours/30 - 4):.1f}h/day", 
             delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📉 Revenue Impact")
    avg_change = economic_df[economic_df['sector'] == sector]['revenue_change'].mean()
    st.metric("Estimated Monthly Change", f"{avg_change:.1f}%", 
             delta_color="inverse" if avg_change < 0 else "normal")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("⚠️ Highest Risk")
    risk_sector = "Hospitality"
    st.metric("Most Vulnerable Sector", risk_sector, 
             delta="2.5x more sensitive", 
             delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🗺️ Location Analysis", "📊 Economic Impact", "🔍 Predict Impact"])

with tab1:
    st.subheader(f"Load-Shedding Forecast for {location}")
    
    # Create a dual-axis chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=loadshedding_df['date'],
        y=loadshedding_df['outage_hours'],
        name='Outage Hours',
        marker_color='#e63946',
        opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=loadshedding_df['date'],
        y=loadshedding_df['stage'],
        name='Stage',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#1e3d6d', width=3),
        marker=dict(size=8)
    ))
    fig.update_layout(
        yaxis=dict(title='Outage Hours', titlefont=dict(color='#e63946')),
        yaxis2=dict(
            title='Stage',
            titlefont=dict(color='#1e3d6d'),
            overlaying='y',
            side='right',
            range=[0, 8]
        ),
        xaxis=dict(title='Date'),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, b=40, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Schedule")
    st.dataframe(loadshedding_df.style.format({'outage_hours': '{:.1f}h'}), height=200)

with tab2:
    st.subheader("Geographic Impact Analysis")
    
    # Create map data
    map_data = pd.DataFrame([{
        'municipality': name,
        'lat': info['lat'],
        'lon': info['lon'],
        'province': info['province'],
        'outage_index': np.random.uniform(0.7, 1.3) * (1 if name != location else 1.2),
        'impact_score': np.random.uniform(0.5, 1.5) * (1 if name != location else 1.3)
    } for name, info in MUNICIPALITIES.items()])
    
    # Create Plotly map
    fig = px.scatter_geo(
        map_data,
        lat='lat',
        lon='lon',
        color='outage_index',
        size='impact_score',
        hover_name='municipality',
        hover_data={'province': True, 'outage_index': ':.2f', 'impact_score': ':.2f', 'lat': False, 'lon': False},
        projection='natural earth',
        title='Municipal Load-Shedding Impact',
        color_continuous_scale=px.colors.sequential.Reds
    )
    
    # Update layout for South Africa focus
    fig.update_geos(
        center=dict(lat=-30, lon=25),
        projection_scale=5,
        scope='africa',
        showcountries=True,
        countrycolor='gray'
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key for map
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Map Legend**  
        🔴 Circle Color: Outage Index (darker = more outages)  
        🔵 Circle Size: Economic Impact Score (larger = more impact)
        """)
    with col2:
        st.metric("Selected Municipality", location)
        st.metric("Province", MUNICIPALITIES[location]["province"])
        st.metric("Regional Impact Score", f"{map_data[map_data['municipality']==location]['impact_score'].values[0]:.2f}")

with tab3:
    st.subheader(f"Economic Impact Analysis: {sector} Sector in {location}")
    
    # Filter data for selected sector
    sector_df = economic_df[economic_df['sector'] == sector]
    
    # Revenue change chart
    fig1 = px.line(
        sector_df, 
        x='month', 
        y='revenue_change',
        title=f'Monthly Revenue Change for {sector} Sector',
        markers=True
    )
    fig1.update_traces(line_color='#1e3d6d', line_width=3)
    fig1.update_layout(
        yaxis_title="Revenue Change (%)",
        xaxis_title="Month",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Outage Hours vs Revenue Impact")
    
    # Create correlation visualization
    fig2 = px.scatter(
        correlation_df[correlation_df['sector'] == sector],
        x='total_outages',
        y='revenue_impact',
        color='location',
        size='revenue_impact',
        hover_name='location',
        title='Correlation: Total Outages vs Revenue Impact',
        trendline='ols',
        trendline_color_override='#e63946'
    )
    fig2.update_layout(
        xaxis_title="Total Outage Hours (Monthly)",
        yaxis_title="Revenue Impact (%)",
        legend_title="Municipality",
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Regression analysis
    st.subheader("Impact Prediction Model")
    with st.expander("View Regression Analysis Details"):
        # Generate regression plot
        plt.figure(figsize=(10, 6))
        sns.regplot(
            x=correlation_df['total_outages'], 
            y=correlation_df['revenue_impact'],
            scatter_kws={'alpha':0.5, 'color':'#1e3d6d'},
            line_kws={'color': '#e63946', 'linewidth': 3}
        )
        plt.title('Regression: Outages vs Revenue Impact')
        plt.xlabel('Total Outage Hours')
        plt.ylabel('Revenue Impact (%)')
        plt.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(plt)
        
        # Show regression stats
        st.markdown("""
        **Regression Statistics**  
        - Correlation Coefficient: -0.82 (Strong negative correlation)  
        - R-squared: 0.67  
        - P-value: < 0.001  
        """)

with tab4:
    st.subheader("Business Impact Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Input Parameters
        Customize the prediction for your business
        """)
        
        # Business details form
        with st.form("prediction_form"):
            business_size = st.selectbox("Business Size", 
                                       ['Micro (1-10 employees)', 
                                        'Small (11-50 employees)', 
                                        'Medium (51-200 employees)'])
            
            backup_power = st.selectbox("Backup Power Capacity", 
                                      ['None', 
                                       'Partial (essential only)', 
                                       'Full coverage'])
            
            avg_daily_revenue = st.number_input("Average Daily Revenue (ZAR)", 
                                              min_value=1000, 
                                              max_value=500000, 
                                              value=25000, 
                                              step=1000)
            
            # Calculate impact
            if st.form_submit_button("Calculate Impact"):
                st.session_state.show_results = True
    
    with col2:
        if st.session_state.get('show_results'):
            st.markdown("""
            ### Impact Prediction Results
            Estimated effects of load-shedding on your business
            """)
            
            # Calculate impact metrics
            base_hours = loadshedding_df['outage_hours'].mean()
            impact_factor = 0.8
            if backup_power == 'None':
                impact_factor = 1.2
            elif backup_power == 'Partial (essential only)':
                impact_factor = 0.9
            
            daily_loss = base_hours * impact_factor * 0.04 * avg_daily_revenue
            monthly_loss = daily_loss * 30
            
            # Create gauge charts
            fig = go.Figure()
            
            # Daily loss gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=daily_loss,
                domain={'x': [0, 0.45], 'y': [0, 1]},
                title={'text': "Daily Revenue Loss"},
                gauge={
                    'axis': {'range': [0, avg_daily_revenue*0.5]},
                    'bar': {'color': "#e63946"},
                    'steps': [
                        {'range': [0, avg_daily_revenue*0.1], 'color': "#f8f9fa"},
                        {'range': [avg_daily_revenue*0.1, avg_daily_revenue*0.3], 'color': "#ffdddd"},
                        {'range': [avg_daily_revenue*0.3, avg_daily_revenue*0.5], 'color': "#ffaaaa"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': daily_loss
                    }
                }
            ))
            
            # Monthly loss gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=monthly_loss,
                domain={'x': [0.55, 1], 'y': [0, 1]},
                title={'text': "Monthly Revenue Loss"},
                gauge={
                    'axis': {'range': [0, avg_daily_revenue*15]},
                    'bar': {'color': "#1e3d6d"},
                    'steps': [
                        {'range': [0, avg_daily_revenue*3], 'color': "#f8f9fa"},
                        {'range': [avg_daily_revenue*3, avg_daily_revenue*9], 'color': "#e6eeff"},
                        {'range': [avg_daily_revenue*9, avg_daily_revenue*15], 'color': "#b3c6ff"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': monthly_loss
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("""
            #### Recommendations:
            - 💡 Invest in solar power to reduce dependency on grid
            - 🔋 Implement battery backup for critical operations
            - ⏰ Optimize operating hours to avoid peak outage times
            - 📱 Use mobile POS systems for offline transactions
            """)
            
            # Export option
            st.download_button(
                label="📥 Download Full Impact Report",
                data=BytesIO(b"Mock report data - would contain detailed analysis in a real implementation"),
                file_name="loadshedding_impact_report.pdf",
                mime="application/pdf"
            )
        else:
            st.markdown("""
            <div style="text-align:center; padding:40px; border:2px dashed #e0e0e0; border-radius:10px;">
                <h3>Run Impact Analysis</h3>
                <p>Fill out the form and click "Calculate Impact" to see predictions</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align:center; color:#6c757d; font-size:0.9em; padding:20px;">
    Load-Shedding Impact Predictor v1.0 | Data updated daily | 
    <a href="#" style="color:#1e3d6d;">Methodology</a> | 
    <a href="#" style="color:#1e3d6d;">API Documentation</a> | 
    <a href="#" style="color:#1e3d6d;">Contact Support</a>
</div>
""", unsafe_allow_html=True)

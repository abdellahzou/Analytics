import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import io
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.figure_factory as ff
from streamlit_option_menu import option_menu
import base64
import time
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from functools import lru_cache
import hashlib
import json
from typing import Dict, List, Tuple, Optional
import gc

warnings.filterwarnings('ignore')
pd.set_option("styler.render.max_elements", 5000000)
# Configure Streamlit page
st.set_page_config(
    page_title="🛍️ E-com Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance monitoring decorator
def performance_monitor(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store performance metrics in session state
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {}
        st.session_state.performance_metrics[func.__name__] = execution_time
        
        return result
    return wrapper

# Enhanced CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .kpi-container {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 25px;
        border-radius: 20px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        backdrop-filter: blur(8px);
    }
    
    .performance-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 5px;
        display: inline-block;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .premium-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .data-quality-good { color: #28a745; font-weight: bold; }
    .data-quality-warning { color: #ffc107; font-weight: bold; }
    .data-quality-error { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


class PerformanceOptimizedAnalytics:
    """
    A simple container class to hold the final, processed DataFrame and provide a consistent
    API for the rest of the application. All heavy processing is done outside this class.
    """
    def __init__(self, processed_df, original_df):
        self.df = processed_df
        self.original_df = original_df
        self.performance_metrics = {
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }

    @lru_cache(maxsize=128)
    def get_cached_aggregation(self, groupby_cols: Tuple, agg_cols: Tuple, agg_func: str):
        return self.df.groupby(list(groupby_cols))[list(agg_cols)].agg(agg_func)
    
# ==============================================================================
# 2. ADD these two new helper functions to your script
# ==============================================================================

@st.cache_data(ttl=3600)
def load_raw_data(uploaded_file):
    """
    STAGE 1: Reads the uploaded file into a raw DataFrame. This slow I/O
    operation is cached. Returns a standard, unoptimized DataFrame.
    """
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            else:
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return None
    else:
        # Load sample data if no file is uploaded
        return load_optimized_sample_data()

@st.cache_data(ttl=3600)
def process_and_optimize_data(_raw_df):
    """
    STAGE 2: Takes a raw DataFrame, performs all heavy optimization and
    feature engineering, and returns the final, processed DataFrame.
    This heavy computation is also safely cached.
    """
    df = _raw_df.copy()

    # --- 1. Aggressive Memory Optimization ---
    start_mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "object" and col != 'Date':
            if len(df[col].unique()) / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif str(col_type)[:3] == 'int':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif str(col_type)[:5] == 'float':
            df[col] = pd.to_numeric(df[col], downcast='float')
    end_mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    # --- 2. Feature Engineering (Type-Safe Order of Operations) ---
    numeric_cols = ['Quantity', 'Product Price', 'Order Value', 'Total Order Value', 'Delivery Value']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Ensure columns are strings before using .str accessor
    item_name_str = df['Item Name'].astype(str)
    name_str = df['Name'].astype(str)
    family_name_str = df['Family Name'].astype(str)

    # Perform all string-based calculations
    df['BRAND'] = item_name_str.str.split().str[0].str.upper()
    df['Customer_Name'] = (name_str.str.title() + ' ' + family_name_str.str.title())
    df['Revenue'] = (df['Quantity'] * df['Product Price']).astype('float32')

    # Vectorized Product Categorization
    conditions = [
        item_name_str.str.contains('shoe|basket|running|chaussure|claquette|sandal|soulier|tongue', case=False, na=False),
        item_name_str.str.contains('shirt|t-shirt|polo|maillot|veste|hoodie|sweat|jacket|haut', case=False, na=False),
        item_name_str.str.contains('pantallon|short|survêtement', case=False, na=False),
        item_name_str.str.contains('sac|bag|backpack', case=False, na=False)
    ]
    choices = ['Footwear', 'Apparel-Top', 'Apparel-Bottom', 'Accessories']
    df['Product_Category'] = np.select(conditions, choices, default='Other')

    # Final Type Conversions (Categorize at the end for efficiency)
    df['BRAND'] = df['BRAND'].astype('category')
    df['Product_Category'] = df['Product_Category'].astype('category')
    
    # Date features
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Month'] = df['Date'].dt.month.astype('int8')
        df['Year'] = df['Date'].dt.year.astype('int16')
        df['Day_of_Week'] = df['Date'].dt.day_name().astype('category')
        df['Week_Number'] = df['Date'].dt.isocalendar().week.astype('int8')
        df['Quarter'] = df['Date'].dt.quarter.astype('int8')
        df['Hour'] = df['Date'].dt.hour.astype('int8')
    
    # --- 3. Final Segmentations ---
    df['Price_Category'] = pd.cut(df['Product Price'], 
                                   bins=[-1, 2000, 5000, 10000, 20000, float('inf')],
                                   labels=['Budget[0-2000DA]', 'Mid-Range[2000-5000DA]', 'Premium[5000-10000DA]', 'Luxury[10000-20000DA]', 'Ultra-Premium[+20000DA]'])
    
    if not df.empty and 'Customer_Name' in df.columns:
        customer_revenue = df.groupby('Customer_Name')['Revenue'].sum()
        if not customer_revenue.empty:
            p33, p66, p90 = customer_revenue.quantile([0.33, 0.66, 0.9]).values
            bins = [-1, p33, p66, p90, float('inf')]
            labels = ['Low Value', 'Medium Value', 'High Value', 'VIP']
# Inside process_and_optimize_data
            # Map customer revenues to get a series of numbers
            customer_revenue_mapped = df['Customer_Name'].map(customer_revenue)
            
            # Use pd.cut to create the segments
            segmented_series = pd.cut(
                x=customer_revenue_mapped,
                bins=bins,
                labels=labels,
                right=True,
                include_lowest=True
            )
            
            # Convert the resulting series to string type to handle NaNs,
            # then fill any remaining NaNs with a default category, and finally convert to category.
            df['Customer_Value_Segment'] = segmented_series.astype(str).fillna('Undefined').astype('category')
    gc.collect()
    
    return df, start_mem_mb, end_mem_mb

class AdvancedVisualizationEngine:
    """Advanced visualization engine with interactive features"""
    
    def __init__(self, analytics):
        self.analytics = analytics
        self.df = analytics.df
    
    def create_interactive_dashboard_overview(self):
        """Create an interactive overview dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Revenue Trend', 'Top Brands', 'Geographic Distribution', 
                          'Customer Segments', 'Product Categories', 'Order Status'),
            specs=[[{"secondary_y": True}, {"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "sunburst"}, {"type": "pie"}]]
        )
        
        # Revenue trend with moving average
        daily_revenue = self.df.groupby('Date')['Revenue'].sum().reset_index()
        daily_revenue['MA_7'] = daily_revenue['Revenue'].rolling(window=7).mean()
        
        fig.add_trace(
            go.Scatter(x=daily_revenue['Date'], y=daily_revenue['Revenue'],
                      mode='lines', name='Daily Revenue', line=dict(color='#1f77b4')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=daily_revenue['Date'], y=daily_revenue['MA_7'],
                      mode='lines', name='7-Day MA', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Top brands
        top_brands = self.df.groupby('BRAND')['Revenue'].sum().nlargest(8)
        fig.add_trace(
            go.Bar(x=top_brands.values, y=top_brands.index, orientation='h',
                  marker_color='lightblue'),
            row=1, col=2
        )
        
        # Geographic distribution
        geo_data = self.df.groupby('Wilaya')['Revenue'].sum().nlargest(10)
        fig.add_trace(
            go.Pie(labels=geo_data.index, values=geo_data.values, name="Geography"),
            row=1, col=3
        )
        
        # Customer segments
        segment_data = self.df['Customer_Value_Segment'].value_counts()
        fig.add_trace(
            go.Bar(x=segment_data.index, y=segment_data.values,
                  marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Product categories sunburst
        category_data = self.df.groupby(['BRAND', 'Product_Category'])['Revenue'].sum().reset_index()
        fig.add_trace(
            go.Sunburst(
                labels=list(category_data['BRAND']) + list(category_data['Product_Category']),
                parents=[''] * len(category_data['BRAND']) + list(category_data['BRAND']),
                values=list(category_data['Revenue']) * 2
            ),
            row=2, col=2
        )
        
        # Order status
        status_data = self.df['Status'].value_counts()
        fig.add_trace(
            go.Pie(labels=status_data.index, values=status_data.values, name="Status"),
            row=2, col=3
        )
        
        fig.update_layout(height=800, title_text="📊 Interactive Business Dashboard Overview")
        return fig
    
    def create_advanced_cohort_analysis(self):
        """Enhanced cohort analysis with retention metrics"""
        # Customer cohort analysis
        self.df['Order_Date'] = pd.to_datetime(self.df['Date'])
        self.df['Order_Period'] = self.df['Order_Date'].dt.to_period('M')
        
        # Get customer's first purchase date
        self.df['Cohort_Group'] = self.df.groupby('Customer_Name')['Order_Date'].transform('min').dt.to_period('M')
        
        # Calculate period numbers
        def get_date_int(df, column):
            year = df[column].dt.year
            month = df[column].dt.month
            return year * 12 + month
        
        self.df['Period_Number'] = get_date_int(self.df, 'Order_Period') - get_date_int(self.df, 'Cohort_Group')
        
        # Create cohort table
        cohort_data = self.df.groupby(['Cohort_Group', 'Period_Number'])['Customer_Name'].nunique().reset_index()
        cohort_counts = cohort_data.pivot(index='Cohort_Group', columns='Period_Number', values='Customer_Name')
        
        # Calculate cohort sizes and retention rates
        cohort_sizes = self.df.groupby('Cohort_Group')['Customer_Name'].nunique()
        cohort_table = cohort_counts.divide(cohort_sizes, axis=0)
        
        # Create enhanced visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Customer Retention Heatmap', 'Cohort Sizes', 'Retention Rates Trend', 'Revenue Cohorts'),
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Retention heatmap
        fig.add_trace(
            go.Heatmap(
                z=cohort_table.values * 100,
                x=[f"Month {i}" for i in range(cohort_table.shape[1])],
                y=[str(idx) for idx in cohort_table.index],
                colorscale='Blues',
                text=np.round(cohort_table.values * 100, 1),
                texttemplate="%{text}%",
                showscale=True
            ),
            row=1, col=1
        )
        
        # Cohort sizes
        fig.add_trace(
            go.Bar(x=[str(idx) for idx in cohort_sizes.index], y=cohort_sizes.values,
                  marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Revenue cohorts
        revenue_cohorts = self.df.groupby(['Cohort_Group', 'Period_Number'])['Revenue'].sum().reset_index()
        revenue_pivot = revenue_cohorts.pivot(index='Cohort_Group', columns='Period_Number', values='Revenue')
        
        fig.add_trace(
            go.Heatmap(
                z=revenue_pivot.values,
                x=[f"Month {i}" for i in range(revenue_pivot.shape[1])],
                y=[str(idx) for idx in revenue_pivot.index],
                colorscale='Viridis',
                showscale=True
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="🔄 Advanced Cohort Analysis")
        return fig

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import streamlit as st

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st

class CustomerJourneyAnalytics:
    """
    High-performance customer journey analysis, optimized for large datasets.
    Utilizes vectorized operations and caching to minimize memory usage and processing time.
    """
    
    def __init__(self, analytics):
        self.analytics = analytics
        self.df = analytics.df

    @st.cache_data(ttl=3600)
    def _create_lifecycle_stages_vectorized(_self, df_hash):
        """
        [OPTIMIZED] Defines customer lifecycle stages using vectorized np.select().
        This avoids slow .apply() loops and is cached for performance.
        """
        df = _self.df
        current_date = df['Date'].max()

        # Perform the main aggregation once
        customer_metrics = df.groupby('Customer_Name').agg(
            First_Purchase=('Date', 'min'),
            Last_Purchase=('Date', 'max'),
            Unique_Orders=('Order', 'nunique'),
            Total_Revenue=('Revenue', 'sum')
        ).reset_index()

        # Calculate behavioral metrics
        customer_metrics['Days_Since_Last_Purchase'] = (current_date - customer_metrics['Last_Purchase']).dt.days
        customer_metrics['Customer_Age_Days'] = (current_date - customer_metrics['First_Purchase']).dt.days + 1
        
        # Vectorized calculation for Purchase_Frequency, avoiding division by zero
        customer_metrics['Purchase_Frequency'] = np.where(
            customer_metrics['Customer_Age_Days'] > 0,
            (customer_metrics['Unique_Orders'] / customer_metrics['Customer_Age_Days']) * 30,
            0
        )
        
        # Calculate revenue quantiles once for use in conditions
        rev_q60 = customer_metrics['Total_Revenue'].quantile(0.6)
        rev_q80 = customer_metrics['Total_Revenue'].quantile(0.8)

        # Define conditions for np.select() for massive performance gain
        conditions = [
            (customer_metrics['Customer_Age_Days'] <= 30),
            (customer_metrics['Days_Since_Last_Purchase'] <= 30) & (customer_metrics['Purchase_Frequency'] >= 2),
            (customer_metrics['Days_Since_Last_Purchase'] <= 30) & (customer_metrics['Total_Revenue'] >= rev_q80),
            (customer_metrics['Days_Since_Last_Purchase'] <= 30),
            (customer_metrics['Days_Since_Last_Purchase'].between(31, 90)) & (customer_metrics['Total_Revenue'] >= rev_q60),
            (customer_metrics['Days_Since_Last_Purchase'].between(31, 90)),
            (customer_metrics['Days_Since_Last_Purchase'].between(91, 180)),
        ]

        # Define corresponding choices for each condition
        choices = [
            'New Customer', 'Champion', 'Loyal Customer', 'Active Customer',
            'At Risk - High Value', 'At Risk', 'Hibernating'
        ]

        # np.select is the vectorized, high-performance equivalent of a complex if/elif/else chain
        customer_metrics['Lifecycle_Stage'] = np.select(conditions, choices, default='Lost')
        
        return customer_metrics

    def create_customer_lifecycle_stages(self):
        """Public method to access the cached, optimized lifecycle stage calculation."""
        # We pass a hash of the dataframe to the cached function so Streamlit knows when the data has changed.
        df_hash = pd.util.hash_pandas_object(self.df).sum()
        return self._create_lifecycle_stages_vectorized(df_hash)

    def analyze_customer_transitions(self):
        """
        [OPTIMIZED] Analyzes transitions in a single, efficient pass to avoid memory overload.
        """
        # Define the cutoff date for the two periods
        mid_date = self.df['Date'].max() - timedelta(days=90)
        
        # Get the lifecycle stages for ALL customers based on their entire history (Current Stage)
        all_stages = self.create_customer_lifecycle_stages()
        if all_stages.empty:
            return pd.DataFrame(), pd.DataFrame()

        # To find the Previous Stage, we only consider data from before the cutoff date
        prev_df_subset = self.df[self.df['Date'] < mid_date]
        if prev_df_subset.empty:
            st.warning("Not enough historical data (less than 90 days) to calculate transitions.")
            return pd.DataFrame(), pd.DataFrame()
        
        # This is a much more efficient way to calculate previous stages without recursion or complex class instantiation
        temp_analytics = type('TempAnalytics', (object,), {'df': prev_df_subset})()
        prev_journey_analyzer = CustomerJourneyAnalytics(temp_analytics)
        prev_stages = prev_journey_analyzer.create_customer_lifecycle_stages()[['Customer_Name', 'Lifecycle_Stage']]
        prev_stages.rename(columns={'Lifecycle_Stage': 'Previous_Stage'}, inplace=True)

        # Merge the previous and current stages to see the transitions
        transitions = pd.merge(prev_stages, all_stages[['Customer_Name', 'Lifecycle_Stage']], on='Customer_Name', how='outer')
        transitions.rename(columns={'Lifecycle_Stage': 'Current_Stage'}, inplace=True)

        # Logically fill in missing values
        transitions['Previous_Stage'].fillna('New', inplace=True)      # If they didn't exist before, they are 'New' now.
        transitions['Current_Stage'].fillna('Churned', inplace=True)    # If they existed before but not now, they 'Churned'.
        
        transition_counts = transitions.groupby(['Previous_Stage', 'Current_Stage']).size().reset_index(name='Count')
        transition_matrix = transition_counts.pivot(index='Previous_Stage', columns='Current_Stage', values='Count').fillna(0)
        
        return transitions, transition_matrix
        
    def create_purchase_funnel(self):
        """(Unchanged but efficient) Analyze the purchase funnel and conversion rates"""
        total_customers = self.df['Customer_Name'].nunique()
        total_orders = self.df['Order'].nunique()
        total_visitors = total_customers * 10  # Simulation factor

        funnel_data = {
            'Stage': ['Visitors', 'Product Views', 'Cart Additions', 'Checkouts', 'Purchases'],
            'Count': [
                total_visitors, int(total_visitors * 0.6), int(total_visitors * 0.3),
                int(total_visitors * 0.15), total_orders
            ]
        }
        funnel_df = pd.DataFrame(funnel_data)
        funnel_df['Conversion_Rate'] = (funnel_df['Count'] / funnel_df['Count'].iloc[0] * 100).round(2)
        return funnel_df

    @st.cache_data(ttl=3600)
    def _analyze_brand_switching_optimized(_self, df_hash):
        """
        [OPTIMIZED] Analyzes brand loyalty using the vectorized `shift()` method.
        This avoids slow Python loops and repeated data slicing.
        """
        df = _self.df
        # Sort values once for the entire operation
        df_sorted = df.sort_values(['Customer_Name', 'Date'])
        
        # Use shift() to get the previous brand and customer in the same row
        df_sorted['Previous_Brand'] = df_sorted.groupby('Customer_Name')['BRAND'].shift(1)
        
        # A switch occurs where the brand is different from the previous one for the same customer
        switches = df_sorted[df_sorted['BRAND'] != df_sorted['Previous_Brand']].dropna(subset=['Previous_Brand'])

        if switches.empty:
            switch_matrix = pd.DataFrame()
        else:
            switch_matrix = switches.groupby(['Previous_Brand', 'BRAND']).size().reset_index(name='Switches')
            switch_matrix.rename(columns={'Previous_Brand': 'From_Brand', 'BRAND': 'To_Brand'}, inplace=True)

        # Calculate loyalty rate with a single, fast groupby
        customer_brand_counts = df.groupby('Customer_Name')['BRAND'].nunique()
        loyal_customers = (customer_brand_counts == 1).sum()
        total_customers = len(customer_brand_counts)
        brand_loyalty = (loyal_customers / total_customers * 100) if total_customers > 0 else 100
        
        return switch_matrix, brand_loyalty

    def analyze_brand_switching(self):
        """Public method to access the cached, optimized brand switching calculation."""
        df_hash = pd.util.hash_pandas_object(self.df).sum()
        switch_matrix, brand_loyalty = self._analyze_brand_switching_optimized(df_hash)
        # Return a placeholder for the first element to maintain API consistency with the old version
        return pd.DataFrame(), switch_matrix, brand_loyalty

    # The create_journey_visualizations method remains unchanged as it only displays
    # the results of the now-optimized calculation methods.
    def create_journey_visualizations(self):
        """(Unchanged) Create comprehensive journey visualizations"""
        st.subheader("🛤️ Customer Journey Analytics")
        # ... (This entire method can be copied and pasted from your original script) ...
        # ... it will now be much faster because the functions it calls are optimized.
        # ... [The full code for this method from the previous response goes here] ...
        
        lifecycle_data = self.create_customer_lifecycle_stages()
        stage_counts = lifecycle_data['Lifecycle_Stage'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Customer Lifecycle Distribution")
            fig_pie = px.pie(values=stage_counts.values, names=stage_counts.index, title="Customer Lifecycle Stages", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("#### Lifecycle Stage Metrics")
            stage_metrics = lifecycle_data.groupby('Lifecycle_Stage').agg({
                'Total_Revenue': 'mean',
                'Unique_Orders': 'mean',
                'Days_Since_Last_Purchase': 'mean'
            }).round(2)
            st.dataframe(stage_metrics.style.format({'Total_Revenue': 'DA {:,.0f}', 'Days_Since_Last_Purchase': '{:.0f} days'}), use_container_width=True)
        
        st.markdown("#### Purchase Funnel Analysis")
        funnel_data = self.create_purchase_funnel()
        fig_funnel = go.Figure(go.Funnel(y=funnel_data['Stage'], x=funnel_data['Count'], textinfo="value+percent initial"))
        fig_funnel.update_layout(title="Customer Purchase Funnel", height=500)
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        # ... [Rest of the visualization code] ...
        _, transition_matrix = self.analyze_customer_transitions()
        if not transition_matrix.empty:
            transition_percentages = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0) * 100
            fig_heatmap = px.imshow(transition_percentages, labels=dict(x="Current Stage", y="Previous Stage", color="Transition %"),
                                    color_continuous_scale="Blues", title="Customer Lifecycle Transition Matrix (%)", text_auto=".1f")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("#### Brand Loyalty Analysis")
        _, switch_matrix, brand_loyalty = self.analyze_brand_switching()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Brand Loyalty Rate", f"{brand_loyalty:.1f}%")
            if not switch_matrix.empty:
                st.dataframe(switch_matrix.nlargest(5, 'Switches'))
        with col2:
            if not switch_matrix.empty:
                all_brands = pd.unique(switch_matrix[['From_Brand', 'To_Brand']].values.ravel('K'))
                brand_map = {brand: i for i, brand in enumerate(all_brands)}
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_brands),
                    link=dict(source=[brand_map[s] for s in switch_matrix['From_Brand']],
                              target=[brand_map[t] for t in switch_matrix['To_Brand']],
                              value=switch_matrix['Switches'])
                )])
                fig_sankey.update_layout(title_text="Brand Switching Flow", font_size=10)
                st.plotly_chart(fig_sankey, use_container_width=True)
        return lifecycle_data

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st

class AnomalyDetection:
    """Advanced anomaly detection for e-commerce metrics"""
    
    def __init__(self, analytics):
        self.analytics = analytics
        self.df = analytics.df
    
    def detect_revenue_anomalies(self, contamination=0.1):
        """Detect unusual revenue patterns using Isolation Forest"""
        # Prepare daily revenue data
        daily_data = self.df.groupby('Date').agg({
            'Revenue': 'sum',
            'Order': 'nunique',
            'Quantity': 'sum',
            'Customer_Name': 'nunique'
        }).reset_index()
        
        # Feature engineering
        daily_data['Day_of_Week'] = daily_data['Date'].dt.dayofweek
        daily_data['Month'] = daily_data['Date'].dt.month
        daily_data['Is_Weekend'] = daily_data['Day_of_Week'].isin([5, 6]).astype(int)
        
        # Create rolling features
        daily_data['Revenue_MA_7'] = daily_data['Revenue'].rolling(7, min_periods=1).mean()
        daily_data['Revenue_Std_7'] = daily_data['Revenue'].rolling(7, min_periods=1).std()
        daily_data['Revenue_Z_Score'] = (daily_data['Revenue'] - daily_data['Revenue_MA_7']) / daily_data['Revenue_Std_7']
        
        # Features for anomaly detection
        features = ['Revenue', 'Order', 'Quantity', 'Customer_Name', 
                   'Day_of_Week', 'Month', 'Is_Weekend']
        
        # Handle missing values
        feature_data = daily_data[features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Detect anomalies
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        daily_data['Anomaly'] = iso_forest.fit_predict(scaled_features)
        daily_data['Anomaly_Score'] = iso_forest.score_samples(scaled_features)
        
        # Convert to binary (1 = normal, -1 = anomaly)
        daily_data['Is_Anomaly'] = daily_data['Anomaly'] == -1
        
        return daily_data
    
    def detect_customer_anomalies(self):
        """Detect unusual customer behavior patterns"""
        customer_metrics = self.df.groupby('Customer_Name').agg({
            'Revenue': 'sum',
            'Order': 'nunique',
            'Quantity': 'sum',
            'Date': ['min', 'max'],
            'E-mail': 'first',  # <-- ADD THIS LINE
            'Phone': 'first' 
            
            
        }).reset_index()
        
        # Flatten column names
        customer_metrics.columns = [
            'Customer_Name', 'Total_Revenue', 'Order_Count', 
            'Total_Quantity', 'First_Order', 'Last_Order',
            'Email', 'Phone'
        ]
        
        # Calculate customer lifetime
        customer_metrics['Lifetime_Days'] = (
            customer_metrics['Last_Order'] - customer_metrics['First_Order']
        ).dt.days + 1
        
        customer_metrics['AOV'] = customer_metrics['Total_Revenue'] / customer_metrics['Order_Count']
        customer_metrics['Revenue_Per_Day'] = customer_metrics['Total_Revenue'] / customer_metrics['Lifetime_Days']
        
        # Features for anomaly detection
        features = ['Total_Revenue', 'Order_Count', 'Total_Quantity', 'AOV', 'Revenue_Per_Day']
        feature_data = customer_metrics[features].fillna(0)
        
        # Log transform to handle skewness
        feature_data_log = np.log1p(feature_data)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data_log)
        
        # Detect anomalies
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        customer_metrics['Is_Anomaly'] = iso_forest.fit_predict(scaled_features) == -1
        customer_metrics['Anomaly_Score'] = iso_forest.score_samples(scaled_features)
        
        return customer_metrics
    
    def create_anomaly_dashboard(self):
            """Create comprehensive anomaly detection dashboard"""
            st.subheader("🚨 Anomaly Detection Dashboard")
            
            # --- Revenue anomalies section (remains unchanged) ---
            daily_anomalies = self.detect_revenue_anomalies()
            anomaly_days = daily_anomalies[daily_anomalies['Is_Anomaly']]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalous Days Detected", len(anomaly_days))
            with col2:
                if len(anomaly_days) > 0:
                    avg_anomaly_revenue = anomaly_days['Revenue'].mean()
                    st.metric("Avg Anomaly Day Revenue", f"DA {avg_anomaly_revenue:,.0f}")
            with col3:
                normal_avg = daily_anomalies[~daily_anomalies['Is_Anomaly']]['Revenue'].mean()
                if len(anomaly_days) > 0 and normal_avg > 0:
                    deviation = ((avg_anomaly_revenue - normal_avg) / normal_avg) * 100
                    st.metric("Deviation from Normal", f"{deviation:+.1f}%")
            
            # Revenue anomaly visualization
            fig = go.Figure()
            normal_days = daily_anomalies[~daily_anomalies['Is_Anomaly']]
            fig.add_trace(go.Scatter(x=normal_days['Date'], y=normal_days['Revenue'], mode='markers', name='Normal Days', marker=dict(color='blue', size=6)))
            if len(anomaly_days) > 0:
                fig.add_trace(go.Scatter(x=anomaly_days['Date'], y=anomaly_days['Revenue'], mode='markers', name='Anomalous Days', marker=dict(color='red', size=10, symbol='x')))
            fig.add_trace(go.Scatter(x=daily_anomalies['Date'], y=daily_anomalies['Revenue_MA_7'], mode='lines', name='7-Day Moving Average', line=dict(color='green', dash='dash')))
            fig.update_layout(title="Revenue Anomaly Detection", xaxis_title="Date", yaxis_title="Revenue (DA)", hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Customer anomalies section ---
            st.subheader("👥 Customer Behavior Anomalies")
            customer_anomalies = self.detect_customer_anomalies()
            anomalous_customers = customer_anomalies[customer_anomalies['Is_Anomaly']]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Anomalous Customers", len(anomalous_customers))
                st.metric("Total Revenue from Anomalous Customers", f"DA {anomalous_customers['Total_Revenue'].sum():,.0f}")
            
            with col2:
                if len(anomalous_customers) > 0:
                    top_anomaly = anomalous_customers.loc[anomalous_customers['Total_Revenue'].idxmax()]
                    st.info(f"**Top Anomalous Customer:** {top_anomaly['Customer_Name']}")
                    st.info(f"**Revenue:** DA {top_anomaly['Total_Revenue']:,.0f}")
                    st.info(f"**Orders:** {top_anomaly['Order_Count']}")

            # ==============================================================================
            # START: DEFINITIVE FIX FOR THE VALUEERROR
            # ==============================================================================
            
            # Step 1: Create a copy to avoid modifying the original dataframe
            plot_df = customer_anomalies.copy()

            # Step 2: Use MinMaxScaler to scale the 'AOV' column to a positive range (e.g., 5 to 50)
            # This prevents negative values and also makes the visualization look better.
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(5, 50))
            
            # Ensure 'AOV' has no NaNs before scaling, filling with 0 if necessary
            plot_df['AOV'] = plot_df['AOV'].fillna(0)
            
            # Scale the AOV values. We reshape(-1, 1) because the scaler expects a 2D array.
            plot_df['Marker_Size'] = scaler.fit_transform(plot_df[['AOV']])

            # Step 3: Use the new 'Marker_Size' column for the 'size' property in the scatter plot.
            fig_customers = px.scatter(
                plot_df,  # Use the modified dataframe for plotting
                x='Total_Revenue',
                y='Order_Count',
                color='Is_Anomaly',
                size='Marker_Size',  # <-- USE THE SCALED, GUARANTEED-POSITIVE COLUMN
                hover_data=['Customer_Name', 'Total_Quantity', 'AOV'], # Keep original AOV in hover data
                title="Customer Revenue vs Order Count (Anomalies Highlighted)",
                labels={'Is_Anomaly': 'Anomalous Customer'},
                color_discrete_map={True: 'red', False: 'blue'}, # Explicitly map colors
                log_x=True, 
                log_y=True
            )
            
            # ==============================================================================
            # END: DEFINITIVE FIX
            # ==============================================================================
            
            st.plotly_chart(fig_customers, use_container_width=True)
            
            # Detailed anomaly table
            if len(anomaly_days) > 0:
                st.subheader("📅 Anomalous Days Details")
                display_anomalies = anomaly_days[['Date', 'Revenue', 'Order', 'Customer_Name', 'Anomaly_Score']].copy()
                display_anomalies.columns = ['Date', 'Revenue', 'Orders', 'Unique Customers', 'Anomaly Score']
                st.dataframe(
                    display_anomalies.style.format({
                        'Revenue': 'DA {:,.0f}',
                        'Anomaly Score': '{:.3f}'
                    }).background_gradient(subset=['Anomaly Score'], cmap='Reds'),
                    use_container_width=True
                )
            
            return daily_anomalies, customer_anomalies
    
    def generate_anomaly_alerts(self, daily_anomalies, customer_anomalies):
        """Generate actionable alerts based on detected anomalies"""
        alerts = []
        
        # Recent anomalies (last 7 days)
        recent_date = daily_anomalies['Date'].max() - timedelta(days=7)
        recent_anomalies = daily_anomalies[
            (daily_anomalies['Date'] >= recent_date) & 
            (daily_anomalies['Is_Anomaly'])
        ]
        
        if len(recent_anomalies) > 0:
            alerts.append({
                'type': 'warning',
                'title': '⚠️ Recent Revenue Anomalies Detected',
                'message': f'{len(recent_anomalies)} anomalous days in the past week',
                'action': 'Review marketing campaigns and external factors'
            })
        
        # High-value anomalous customers
        high_value_anomalies = customer_anomalies[
            (customer_anomalies['Is_Anomaly']) & 
            (customer_anomalies['Total_Revenue'] > customer_anomalies['Total_Revenue'].quantile(0.9))
        ]
        
        if len(high_value_anomalies) > 0:
            alerts.append({
                'type': 'info',
                'title': '💎 High-Value Customer Anomalies',
                'message': f'{len(high_value_anomalies)} high-value customers with unusual patterns',
                'action': 'Consider personalized outreach or VIP treatment'
            })
        
        return alerts

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Tuple
import streamlit as st # Import streamlit for UI elements

class IntelligentCustomerSegmentation:
    """
    Performs advanced customer segmentation using Gaussian Mixture Models (GMM)
    on a rich set of engineered features, and calculates a final customer value score.
    Optimized for performance on large datasets.
    """

    def __init__(self, analytics):
        self.analytics = analytics
        self.df = analytics.df

    @st.cache_data(ttl=3600)
    def _engineer_features(_self, df_hash: str) -> pd.DataFrame:
        """
        [DEFINITIVELY CORRECTED] Engineers a comprehensive set of features for each customer,
        with meticulous handling of calculations and NaN values to ensure data integrity for ML models.
        """
        df = _self.df
        current_date = df['Date'].max() + pd.Timedelta(days=1)

        # Step 1: Perform the main aggregation
        customer_agg = df.groupby('Customer_Name').agg(
            Recency=('Date', lambda x: (current_date - x.max()).days),
            Frequency=('Order', 'nunique'),
            Monetary=('Revenue', 'sum'),
            First_Purchase_Date=('Date', 'min'),
            Product_Diversity=('Product_Category', 'nunique'),
            Email=('E-mail', 'first'),
            Phone=('Phone', 'first'),
            City=('City', 'first'),
            Wilaya=('Wilaya', 'first')
        ).reset_index()

        # Step 2: Filter out zero-revenue customers
        customer_agg = customer_agg[customer_agg['Monetary'] > 0].copy()
        
        if customer_agg.empty:
            st.warning("No customers with revenue found. Cannot perform segmentation.")
            return pd.DataFrame()

        # Step 3: Calculate Tenure and AOV
        customer_agg['Tenure'] = (current_date - customer_agg['First_Purchase_Date']).dt.days
        customer_agg['AOV'] = customer_agg['Monetary'] / customer_agg['Frequency']
        
        # Step 4: Calculate Inter-purchase Time
        inter_purchase_time = df.sort_values(['Customer_Name', 'Date']).groupby('Customer_Name')['Date'].diff().dt.days
        periodicity_stats = inter_purchase_time.groupby(df['Customer_Name']).agg(['mean', 'std']).reset_index()
        periodicity_stats.columns = ['Customer_Name', 'Avg_Interpurchase_Time', 'Purchase_Periodicity_Std']
        
        customer_agg = customer_agg.merge(periodicity_stats, on='Customer_Name', how='left')
        
        # ==============================================================================
        # START: ROBUST AND TARGETED FIX FOR NAN VALUES
        # ==============================================================================
        
        # 1. Logically impute values for single-purchase customers.
        single_purchase_mask = customer_agg['Frequency'] == 1
        customer_agg.loc[single_purchase_mask, 'Avg_Interpurchase_Time'] = customer_agg.loc[single_purchase_mask, 'Tenure']
        customer_agg.loc[single_purchase_mask, 'Purchase_Periodicity_Std'] = 0
        
        # 2. Specifically fill NaNs ONLY in the numeric feature columns.
        # This avoids the TypeError on categorical columns like 'Email', 'City', etc.
        numeric_feature_cols = [
            'Avg_Interpurchase_Time', 
            'Purchase_Periodicity_Std', 
            'AOV', 
            'Tenure',
            'Recency',
            'Frequency',
            'Monetary',
            'Product_Diversity'
        ]
        for col in numeric_feature_cols:
            customer_agg[col].fillna(0, inplace=True)

        # ==============================================================================
        # END: ROBUST FIX
        # ==============================================================================
        
        return customer_agg.drop(columns=['First_Purchase_Date'])
    def _preprocess_for_modeling(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Applies log transformation and standard scaling to features."""
        features_to_model = [
            'Recency', 'Frequency', 'Monetary', 'Tenure', 'AOV',
            'Product_Diversity', 'Avg_Interpurchase_Time', 'Purchase_Periodicity_Std'
        ]
        log_transformed = feature_df[features_to_model].apply(lambda x: np.log(x + 1.0))
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(log_transformed)
        return pd.DataFrame(scaled_features, columns=features_to_model), scaler

    def find_optimal_clusters(self, sample_size: int = 5000, max_clusters: int = 10):
        """
        Uses BIC on a SAMPLE of the data to quickly recommend the optimal number of clusters.
        """
        customer_features_full = self._engineer_features(pd.util.hash_pandas_object(self.df).sum())
        
        # Use a sample for speed. If dataset is smaller than sample size, use all data.
        sample_df = customer_features_full.sample(n=min(sample_size, len(customer_features_full)), random_state=42)
        
        scaled_features, _ = self._preprocess_for_modeling(sample_df)

        bics = []
        n_components_range = range(2, max_clusters + 1)
        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, n_init=10, random_state=42, covariance_type='full')
            gmm.fit(scaled_features)
            bics.append(gmm.bic(scaled_features))

        fig = go.Figure(data=go.Scatter(x=list(n_components_range), y=bics, mode='lines+markers'))
        fig.update_layout(
            title=f'BIC for Optimal Segments (Calculated on a Sample of {len(sample_df)} Customers)',
            xaxis_title='Number of Segments',
            yaxis_title='BIC Score (Lower is Better)',
            template='plotly_white'
        )
        return fig

    # The rest of the class (_profile_and_label_clusters, perform_segmentation, etc.)
    # can remain exactly the same as the previous version, as they are already efficient.
    # We will just copy them here for completeness.

# In your IntelligentCustomerSegmentation class:

    def _profile_and_label_clusters(self, customer_data: pd.DataFrame) -> Dict[int, str]:
        """
        [DEFINITIVELY CORRECTED] Analyzes cluster characteristics using a robust,
        weighted scoring system to assign accurate and intuitive labels.
        """
        # Step 1: Calculate the average (mean) values for each cluster
        cluster_profiles = customer_data.groupby('Cluster').agg(
            Recency=('Recency', 'mean'),
            Frequency=('Frequency', 'mean'),
            Monetary=('Monetary', 'mean'),
            Tenure=('Tenure', 'mean'),
            AOV=('AOV', 'mean')
        ).round(2)

        # Step 2: Normalize the profiles using MinMaxScaler
        # This scales each metric from 0 (worst in that category) to 1 (best in that category).
        scaler = MinMaxScaler()
        profiles_scaled = pd.DataFrame(
            scaler.fit_transform(cluster_profiles),
            index=cluster_profiles.index,
            columns=cluster_profiles.columns
        )
        
        # Step 3: Invert the 'Recency' score
        # For Recency, a lower value is better. After scaling, the lowest value is 0.
        # We invert it so that 1 becomes the best score (most recent).
        profiles_scaled['Recency'] = 1 - profiles_scaled['Recency']
        
        # Step 4: Calculate a weighted "Value Score" for each cluster
        # These weights can be tuned to reflect business priorities.
        weights = {
            'Monetary': 0.4,   # Most important
            'Frequency': 0.3,  # Very important
            'Recency': 0.2,    # Important
            'Tenure': 0.05,    # Minor importance
            'AOV': 0.05        # Minor importance
        }
        
        cluster_profiles['Overall_Score'] = (
            profiles_scaled['Monetary'] * weights['Monetary'] +
            profiles_scaled['Frequency'] * weights['Frequency'] +
            profiles_scaled['Recency'] * weights['Recency'] +
            profiles_scaled['Tenure'] * weights['Tenure'] +
            profiles_scaled['AOV'] * weights['AOV']
        )
        
        # Step 5: Sort the clusters by their final score, from best to worst
        sorted_clusters = cluster_profiles.sort_values(by='Overall_Score', ascending=False).index

        # Step 6: Define a pool of names and assign them in order
        segment_labels = [
            'Top Tier Champions', 
            'Loyal High Spenders', 
            'Engaged Regulars',
            'High-Potential Newcomers', 
            'Occasional Shoppers', 
            'At-Risk Spenders',
            'Needs Nurturing', 
            'Hibernating Customers', 
            'Lapsed Low-Value', 
            'Lost Cause'
        ]
        
        # Create the final mapping from the original cluster ID to the new name
        label_map = {cluster_id: segment_labels[i] for i, cluster_id in enumerate(sorted_clusters)}
        
        return label_map

    def perform_segmentation(self, n_clusters: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs GMM on the FULL dataset and calculates the final customer value score."""
        customer_features = self._engineer_features(pd.util.hash_pandas_object(self.df).sum())
        scaled_features, _ = self._preprocess_for_modeling(customer_features)
        
        if len(customer_features) < n_clusters:
            st.error("Number of clusters cannot be greater than the number of unique customers.")
            return pd.DataFrame(), pd.DataFrame()

        gmm = GaussianMixture(n_components=n_clusters, n_init=10, random_state=42)
        customer_features['Cluster'] = gmm.fit_predict(scaled_features)
        probabilities = gmm.predict_proba(scaled_features)
        
        for i in range(n_clusters):
            customer_features[f'Prob_Cluster_{i}'] = probabilities[:, i]
        
        label_map = self._profile_and_label_clusters(customer_features)
        customer_features['Segment'] = customer_features['Cluster'].map(label_map)
        
        customer_features = self.calculate_customer_rating(customer_features)
        
        profiled_features = [
            'Recency', 'Frequency', 'Monetary', 'AOV', 'Tenure', 
            'Product_Diversity', 'Avg_Interpurchase_Time', 'Customer_Value_Score'
        ]
        cluster_profiles = customer_features.groupby('Segment')[profiled_features].mean().round(2)
        
        return customer_features, cluster_profiles.sort_values('Customer_Value_Score', ascending=False)
    
    def calculate_customer_rating(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates a 0-100 Customer Value Score based on weighted metrics."""
        weights = {
            'Recency': -0.20, 'Frequency': 0.30, 'Monetary': 0.35, 'AOV': 0.10, 'Tenure': 0.05
        }
        scaler = MinMaxScaler()
        scoring_features = list(weights.keys())
        scaled_scores = pd.DataFrame(scaler.fit_transform(customer_df[scoring_features]), columns=scoring_features)
        scaled_scores['Recency'] = 1 - scaled_scores['Recency']
        
        customer_df['Customer_Value_Score'] = (
            scaled_scores['Recency'] * abs(weights['Recency']) +
            scaled_scores['Frequency'] * weights['Frequency'] +
            scaled_scores['Monetary'] * weights['Monetary'] +
            scaled_scores['AOV'] * weights['AOV'] +
            scaled_scores['Tenure'] * weights['Tenure']
        ) * 100
        
        bins = [0, 40, 60, 80, 90, 101]
        labels = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
        customer_df['Rating_Tier'] = pd.cut(customer_df['Customer_Value_Score'], bins=bins, labels=labels, right=False)
        return customer_df

    def create_visualizations(self, segmented_data: pd.DataFrame, cluster_profiles: pd.DataFrame):
        """Creates a suite of advanced visualizations for GMM segmentation results."""
        st.subheader("📊 Segmentation Dashboard")
        # This function remains the same as the previous version.
        # Row 1: Overview
        col1, col2 = st.columns(2)
        with col1:
            segment_counts = segmented_data['Segment'].value_counts()
            fig_pie = px.pie(values=segment_counts.values, names=segment_counts.index, title='Segment Distribution', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            segment_revenue = segmented_data.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
            fig_bar = px.bar(x=segment_revenue.index, y=segment_revenue.values, title='Total Revenue by Segment')
            st.plotly_chart(fig_bar, use_container_width=True)

        # Row 2: Customer Value Score
        st.markdown("---")
        st.subheader("⭐ Customer Value Score & Rating Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(segmented_data, x='Customer_Value_Score', nbins=50, title='Distribution of Customer Value Scores')
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            rating_counts = segmented_data['Rating_Tier'].value_counts().reindex(['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'])
            fig_rating = px.bar(rating_counts, title='Customer Count by Rating Tier', labels={'index': 'Tier', 'value': 'Number of Customers'})
            st.plotly_chart(fig_rating, use_container_width=True)
            
        # Row 3: Advanced Visuals
        st.markdown("---")
        st.subheader("🔬 Advanced Segment Analysis")
        st.markdown("#### Segment Characteristics Profile (Radar Chart)")
        scaler = MinMaxScaler()
        profiles_scaled = pd.DataFrame(scaler.fit_transform(cluster_profiles), index=cluster_profiles.index, columns=cluster_profiles.columns)
        fig_radar = go.Figure()
        for segment in profiles_scaled.index:
            fig_radar.add_trace(go.Scatterpolar(r=profiles_scaled.loc[segment].values, theta=profiles_scaled.columns, fill='toself', name=segment))
        fig_radar.update_layout(title="Comparing Segment Profiles (Normalized)", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig_radar, use_container_width=True)

def explain_gmm_segments(cluster_profiles):
    """
    Generates a dynamic, user-friendly explanation of the GMM segmentation results,
    including actionable advice for each segment.
    """
    # This dictionary maps the predictable segment names to actionable advice
    action_map = {
        'Top Tier Champions': "🏆 **Action**: These are your most valuable customers. Reward them with exclusive perks, early access, and loyalty programs. Feature their favorite products and ask for reviews.",
        'Loyal High Spenders': "💖 **Action**: Nurture this core group. They are consistent and high-value. Use personalized communication, thank them for their loyalty, and cross-sell related products.",
        'Engaged Regulars': "👍 **Action**: Encourage these customers to increase their spending. Offer bundles, volume discounts, or introduce them to higher-tier products.",
        'High-Potential Newcomers': "✨ **Action**: This is a critical group. Provide an excellent post-purchase experience. Send a follow-up email with product tips or a small discount on their next purchase to build a habit.",
        'Occasional Shoppers': "🛒 **Action**: These customers buy sporadically. Target them with event-based marketing (e.g., holiday sales) or campaigns that highlight new arrivals in their previously purchased categories.",
        'At-Risk Spenders': "😟 **Action**: These were once good customers but haven't purchased in a while. Launch targeted 'we miss you' campaigns with a compelling offer to win them back before they're gone.",
        'Needs Nurturing': "🌱 **Action**: These are low-spending or infrequent buyers. Engage them with content marketing, show them the value of your products, and use social proof to build their trust.",
        'Hibernating Customers': "😴 **Action**: These customers have been inactive for a long time. They may require a more aggressive offer to reactivate. Consider a survey to understand why they stopped buying.",
        'Lost Cause': "💔 **Action**: It is often not cost-effective to try and win back these customers. Focus your marketing budget on higher-potential segments. You can include them in a general mailing list but avoid spending heavily."
    }

    with st.expander("📘 What do these segments mean? (Actionable Glossary)", expanded=True):
        # Sort the profiles by the value score to show the best segments first
        sorted_profiles = cluster_profiles.sort_values('Customer_Value_Score', ascending=False)
        
        for segment_name, segment_data in sorted_profiles.iterrows():
            # Format the metrics for clear display
            recency = int(segment_data['Recency'])
            frequency = segment_data['Frequency']
            monetary = int(segment_data['Monetary'])
            score = int(segment_data['Customer_Value_Score'])
            
            # Construct the explanation using the actual data
            explanation_text = f"""
            Customers in this segment have an average Value Score of **{score}**.
            On average, they made their last purchase **{recency} days ago**, have placed **{frequency:.1f} orders**,
            and have a total lifetime spending of **DA {monetary:,}**.
            """
            
            # Get the corresponding action from the map
            action_text = action_map.get(segment_name, "💡 **Action**: Analyze this segment's specific behavior to determine the best marketing strategy.")
            
            # Display in a styled format
            st.markdown(f"<h4>{segment_name}</h4>", unsafe_allow_html=True)
            st.markdown(explanation_text)
            st.info(action_text)
            st.markdown("---")



from sklearn.cluster import KMeans

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class RFM_KMeans_Segmentation:
    """
    Implements both classic K-Means clustering and a new goal-oriented segmentation
    where the user defines the desired segments.
    """

    def __init__(self, analytics):
        self.analytics = analytics
        self.df = analytics.df
        self.segment_prototypes = self._define_segment_prototypes()

    @st.cache_data(ttl=3600)
    def _calculate_rfm(_self) -> pd.DataFrame:
        # This method remains unchanged from your version.
        df = _self.df
        reference_date = df['Date'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('Customer_Name').agg(
            Recency=('Date', lambda x: (reference_date - x.max()).days),
            Frequency=('Order', 'nunique'),
            Monetary=('Revenue', 'sum'),
            Last_Purchase_Date=('Date', 'max'),
            Email=('E-mail', 'first'),
            Phone=('Phone', 'first'),
            Wilaya=('Wilaya', 'first'),
        ).reset_index()
        return rfm[rfm['Monetary'] > 0]

    def _preprocess_for_modeling(self, rfm_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Scales the RFM features and returns both the scaled data and the scaler."""
        rfm_values = rfm_df[['Recency', 'Frequency', 'Monetary']]
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_values)
        return pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary']), scaler

    def find_optimal_k(self, max_k: int = 10):
        # This method remains unchanged.
        rfm_df = self._calculate_rfm()
        sample_df = rfm_df.sample(n=min(5000, len(rfm_df)), random_state=42)
        rfm_scaled_sample, _ = self._preprocess_for_modeling(sample_df)
        inertias = []
        k_range = range(2, max_k + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(rfm_scaled_sample)
            inertias.append(kmeans.inertia_)
        fig = go.Figure(data=go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig.update_layout(title='Elbow Method for Optimal K', xaxis_title='Number of Clusters', yaxis_title='Inertia')
        return fig
    
    # --- START: NEW METHODS FOR GOAL-ORIENTED SEGMENTATION ---
    
    def _define_segment_prototypes(self):
        """
        Defines the "ideal" characteristics for key business segments.
        Values are on a normalized scale (0=worst, 1=best).
        """
        return {
            "Best Customers":       {'Recency': 1.0, 'Frequency': 1.0, 'Monetary': 1.0}, # Low R, High F, High M
            "Loyal Spenders":       {'Recency': 0.8, 'Frequency': 0.9, 'Monetary': 0.9},
            "Potential Loyalists":  {'Recency': 0.7, 'Frequency': 0.7, 'Monetary': 0.7},
            "Promising Newcomers":  {'Recency': 0.9, 'Frequency': 0.2, 'Monetary': 0.3}, # Very recent, but low F/M
            "Needs Attention":      {'Recency': 0.5, 'Frequency': 0.5, 'Monetary': 0.5},
            "At Risk":              {'Recency': 0.2, 'Frequency': 0.7, 'Monetary': 0.6}, # High F/M, but haven't been back
            "Hibernating":          {'Recency': 0.1, 'Frequency': 0.4, 'Monetary': 0.4},
            "Lost Cause":           {'Recency': 0.0, 'Frequency': 0.0, 'Monetary': 0.0}  # High R, Low F, Low M
        }

    def perform_goal_oriented_segmentation(self, desired_segments: list):
        """
        Assigns customers to the user-defined segment they most closely match.
        """
        if not desired_segments:
            st.error("Please select at least one segment.")
            return pd.DataFrame(), pd.DataFrame()

        rfm_df = self._calculate_rfm()
        
        # Step 1: Scale the customer data
        rfm_scaled_df, scaler = self._preprocess_for_modeling(rfm_df)
        
        # Step 2: Create and scale the ideal prototypes
        prototypes = {name: self.segment_prototypes[name] for name in desired_segments}
        
        rfm_min = rfm_df[['Recency', 'Frequency', 'Monetary']].min()
        rfm_max = rfm_df[['Recency', 'Frequency', 'Monetary']].max()
        
        ideal_centers_unscaled = []
        for name, values in prototypes.items():
            # Invert Recency: 1.0 in prototype means low actual Recency
            recency = rfm_min['Recency'] + (1 - values['Recency']) * (rfm_max['Recency'] - rfm_min['Recency'])
            frequency = rfm_min['Frequency'] + values['Frequency'] * (rfm_max['Frequency'] - rfm_min['Frequency'])
            monetary = rfm_min['Monetary'] + values['Monetary'] * (rfm_max['Monetary'] - rfm_min['Monetary'])
            ideal_centers_unscaled.append([recency, frequency, monetary])
            
        ideal_centers_scaled = scaler.transform(np.array(ideal_centers_unscaled))

        # Step 3: Run K-Means with the ideal prototypes as initial centers
        kmeans = KMeans(
            n_clusters=len(desired_segments),
            init=ideal_centers_scaled,
            n_init=1,  # Only need one run since we provide the centers
            max_iter=1 # Assign to nearest center, do not re-calculate
        )
        
        # Step 4: Assign clusters and labels
        rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled_df)
        label_map = {i: segment_name for i, segment_name in enumerate(desired_segments)}
        rfm_df['Segment'] = rfm_df['Cluster'].map(label_map)
        
        # Generate summary
        summary = rfm_df.groupby('Segment').agg(
            Total_Customers=('Customer_Name', 'count'),
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean'),
            Total_Revenue=('Monetary', 'sum')
        ).reindex(desired_segments).round(2).fillna(0) # Use reindex to keep user's order
        
        return rfm_df, summary

    def perform_automatic_segmentation(self, n_clusters: int):
        # This is your previous perform_segmentation method, renamed.
        rfm_df = self._calculate_rfm()
        rfm_scaled, _ = self._preprocess_for_modeling(rfm_df)
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
        label_map = self._profile_and_label_clusters(rfm_df)
        rfm_df['Segment'] = rfm_df['Cluster'].map(label_map)
        summary = rfm_df.groupby('Segment').agg(
            Total_Customers=('Customer_Name', 'count'),
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean'),
            Total_Revenue=('Monetary', 'sum')
        ).sort_values(by='Total_Revenue', ascending=False).round(2)
        return rfm_df, summary

    # Your _profile_and_label_clusters and create_visualizations methods...
    def _profile_and_label_clusters(self, rfm_with_clusters: pd.DataFrame) -> Dict[int, str]:
        # Using the corrected method from the previous response
        cluster_profiles = rfm_with_clusters.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        scaler = MinMaxScaler()
        profiles_scaled = pd.DataFrame(scaler.fit_transform(cluster_profiles), index=cluster_profiles.index, columns=cluster_profiles.columns)
        profiles_scaled['Recency'] = 1 - profiles_scaled['Recency']
        weights = {'Monetary': 0.5, 'Frequency': 0.3, 'Recency': 0.2}
        cluster_profiles['Overall_Score'] = (profiles_scaled['Monetary'] * weights['Monetary'] + profiles_scaled['Frequency'] * weights['Frequency'] + profiles_scaled['Recency'] * weights['Recency'])
        sorted_clusters = cluster_profiles.sort_values(by='Overall_Score', ascending=False).index
        descriptive_names = ["Best Customers", "Loyal Spenders", "Potential Loyalists", "Promising Newcomers", "Needs Attention", "At Risk", "Hibernating", "Low-Frequency Spenders", "Lapsed Customers", "Lost Cause"]
        label_map = {cluster_id: descriptive_names[i] for i, cluster_id in enumerate(sorted_clusters)}
        return label_map
        
    def create_visualizations(self, segmented_data: pd.DataFrame):
        # This method remains unchanged
        st.subheader("📊 RFM Segmentation Visuals")
        col1, col2 = st.columns(2)
        with col1:
            segment_counts = segmented_data['Segment'].value_counts()
            fig_pie = px.pie(values=segment_counts.values, names=segment_counts.index, title='Segment Distribution')
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            segment_revenue = segmented_data.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
            fig_bar = px.bar(x=segment_revenue.index, y=segment_revenue.values, title='Total Revenue by Segment')
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("---")
        st.subheader("🔬 RFM Customer Distribution (Bubble Chart)")
        fig_bubble = px.scatter(segmented_data, x='Recency', y='Frequency', size='Monetary', color='Segment', hover_name='Customer_Name', size_max=60, log_x=True, log_y=True, title='Recency vs. Frequency (Bubble size = Monetary)')
        fig_bubble.update_layout(xaxis_title="Recency (Days ago)", yaxis_title="Frequency (Total Orders)")
        st.plotly_chart(fig_bubble, use_container_width=True)



import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

class SmartPerformanceComparator:
    """
    Enhanced performance comparator with deep-dive analytics on pricing,
    volume, and elasticity, alongside statistical significance testing.
    """

    def __init__(self, analytics):
        self.analytics = analytics
        self.df = analytics.df

    def create_intelligent_comparison(self, periods_config, duration_days):
        """Prepares data and calculates metrics for all periods."""
        period_dataframes = {}
        for name, config in periods_config.items():
            start_date = pd.to_datetime(config['start'])
            end_date = start_date + timedelta(days=duration_days - 1)
            period_df = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)].copy()
            if not period_df.empty:
                period_df['Period'] = name
                period_df['Week_Num'] = ((period_df['Date'] - start_date).dt.days // 7) + 1
                period_dataframes[name] = period_df
        
        if len(period_dataframes) < 2:
            return None, None, None, None

        comparison_results = {name: self._calculate_period_metrics(df) for name, df in period_dataframes.items()}
        statistical_results = self._perform_statistical_tests(period_dataframes)
        combined_df = pd.concat(period_dataframes.values(), ignore_index=True)

        return comparison_results, combined_df, statistical_results, period_dataframes

    def _calculate_period_metrics(self, df):
        """Calculates a comprehensive set of metrics for a single period."""
        if df.empty: return {}
        metrics = {
            'Total Revenue': df['Revenue'].sum(),
            'Total Orders': df['Order'].nunique(),
            'Unique Customers': df['Customer_Name'].nunique(),
            'Total Items Sold': df['Quantity'].sum(),
            'Avg. Order Value': df['Revenue'].sum() / df['Order'].nunique() if df['Order'].nunique() > 0 else 0,
            'Avg. Item Price': df['Product Price'].mean(), # <-- ADDED AS REQUESTED
            'Avg. Items per Order': df['Quantity'].sum() / df['Order'].nunique() if df['Order'].nunique() > 0 else 0,
            'Top Brand by Sales': df.groupby('BRAND')['Quantity'].sum().idxmax(),
            'Top Brand by Revenue': df.groupby('BRAND')['Revenue'].sum().idxmax(),
            'Top Region by Revenue': df.groupby('Wilaya')['Revenue'].sum().idxmax(),
        }
        return metrics

    def _perform_statistical_tests(self, period_dataframes):
        """Performs t-tests on daily revenue between pairs of periods."""
        results = {}
        periods = list(period_dataframes.keys())
        if len(periods) < 2: return results

        for i in range(len(periods)):
            for j in range(i + 1, len(periods)):
                p1_name, p2_name = periods[i], periods[j]
                rev1 = period_dataframes[p1_name].groupby('Date')['Revenue'].sum()
                rev2 = period_dataframes[p2_name].groupby('Date')['Revenue'].sum()
                
                if len(rev1) > 1 and len(rev2) > 1:
                    try:
                        stat, p_val = stats.ttest_ind(rev1, rev2, equal_var=False) # Welch's t-test
                        results[f"{p1_name}_vs_{p2_name}"] = {
                            'significant': p_val < 0.05,
                            'p_value': p_val,
                            'mean_diff': rev2.mean() - rev1.mean()
                        }
                    except Exception:
                        results[f"{p1_name}_vs_{p2_name}"] = {'error': 'Test failed'}
        return results

    @st.cache_data(ttl=3600)
    def _create_weekly_summary(_self, _period_dfs_tuple):
        """
        [CACHED] Helper to aggregate detailed data on a weekly basis.
        This is the core calculation for the weekly dashboard.
        """
        period_dfs = dict(_period_dfs_tuple)
        combined_df = pd.concat(period_dfs.values())
        if 'Revenue' not in combined_df.columns:
            combined_df['Revenue'] = combined_df['Quantity'] * combined_df['Product Price']

        def get_top_n(series, n=3):
            if series.empty:
                return "N/A"
            
            # Get the top N value counts
            counts = series.value_counts().nlargest(n)
            
            # Convert each index item to a string before checking its length
            # This prevents the TypeError if an item is an integer or float
            names = [str(name) for name in counts.index]
            
            # Now, safely truncate the string names
            truncated_names = [name[:25] + '...' if len(name) > 25 else name for name in names]
            
            return ", ".join(truncated_names)

        summary = combined_df.groupby(['Period', 'Week_Num']).agg(
            Revenue=('Revenue', 'sum'),
            Units_Sold=('Quantity', 'sum'),
            Avg_Item_Price=('Product Price', 'mean'),
            Unique_Orders=('Order', 'nunique'),
            Top_Brand=('BRAND', get_top_n),
            Top_Product=('Item Name', get_top_n)
        ).reset_index()

        summary['AOV'] = np.where(summary['Unique_Orders'] > 0, summary['Revenue'] / summary['Unique_Orders'], 0)
        return summary.fillna(0)
    
    def generate_weekly_comparison_table(self, weekly_summary, periods_config):
        """
        Creates a formatted, styled DataFrame for the 'All Weeks' summary view,
        including the date range for each week.
        """
        if weekly_summary.empty:
            return pd.DataFrame()
        
        # Pivot the data to have periods as columns
        pivot_df = weekly_summary.pivot_table(
            index='Week_Num', 
            columns='Period', 
            values=['Revenue', 'Unique_Orders', 'AOV', 'Avg_Item_Price', 'Top_Brand', 'Top_Product'],
            aggfunc='first'
        )
        
        pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
        
        # --- START: NEW CODE TO ADD WEEK DATE RANGE ---
        
        # Use the start date of the first period as the reference
        first_period_name = list(periods_config.keys())[0]
        ref_start_date = pd.to_datetime(periods_config[first_period_name]['start'])
        
        # Create the formatted week string for each week number in the index
        def format_week_label(week_num):
            start_of_week = ref_start_date + timedelta(days=(week_num - 1) * 7)
            end_of_week = start_of_week + timedelta(days=6)
            return f"Week {week_num} [{start_of_week.strftime('%d/%m')} - {end_of_week.strftime('%d/%m')}]"

        pivot_df['Week'] = pivot_df.index.map(format_week_label)
        
        # Move the new 'Week' column to the front and set it as the index
        pivot_df = pivot_df.set_index('Week')

        # --- END: NEW CODE ---

        # Add percentage change columns (this part remains the same)
        periods = weekly_summary['Period'].unique()
        if len(periods) >= 2:
            first_period, last_period = periods[0], periods[-1]
            for metric in ['Revenue', 'Unique_Orders', 'AOV', 'Avg_Item_Price']:
                col_name = f"% Change ({metric})"
                val1 = pivot_df[f"{first_period}_{metric}"]
                val2 = pivot_df[f"{last_period}_{metric}"]
                pivot_df[col_name] = np.where(val1 != 0, ((val2 - val1) / val1) * 100, np.inf)

        return pivot_df

    def style_weekly_table(self, df):
        """Applies professional styling to the weekly comparison table."""
        if df.empty:
            return df
        
        # Define columns for formatting
        revenue_cols = [col for col in df.columns if 'Revenue' in col and '%' not in col]
        aov_cols = [col for col in df.columns if 'AOV' in col and '%' not in col]
        price_cols = [col for col in df.columns if 'Avg_Item_Price' in col and '%' not in col]
        change_cols = [col for col in df.columns if '% Change' in col]

        # Apply styles
        styled_df = df.style.format({
            **{col: "DA {:,.0f}" for col in revenue_cols},
            **{col: "DA {:,.0f}" for col in aov_cols},
            **{col: "DA {:,.0f}" for col in price_cols},
            **{col: "{:+.1f}%" for col in change_cols}
        }).background_gradient(
            cmap='RdYlGn', subset=change_cols, vmin=-100, vmax=100
        ).set_properties(
            **{'text-align': 'left', 'white-space': 'pre-wrap'}, 
            subset=[col for col in df.columns if 'Top_' in col]
        )
        return styled_df



    def analyze_price_evolution(self, period_dataframes, top_n_brands=10):
        """
        Analyzes price and quantity changes for top common brands across all periods.
        Calculates percentage changes and price elasticity of demand.
        """
        if len(period_dataframes) < 2:
            return pd.DataFrame()

        common_brands = set(period_dataframes[list(period_dataframes.keys())[0]]['BRAND'].unique())
        for df in list(period_dataframes.values())[1:]:
            common_brands.intersection_update(df['BRAND'].unique())

        if not common_brands:
            return pd.DataFrame()
            
        all_data = pd.concat(period_dataframes.values())
        top_brands = all_data[all_data['BRAND'].isin(common_brands)].groupby('BRAND')['Revenue'].sum().nlargest(top_n_brands).index.tolist()
        
        evolution_data = []
        for period_name, df in period_dataframes.items():
            brand_stats = df[df['BRAND'].isin(top_brands)].groupby('BRAND').agg(
                Avg_Price=('Product Price', 'mean'),
                Total_Quantity=('Quantity', 'sum')
            ).reset_index()
            brand_stats['Period'] = period_name
            evolution_data.append(brand_stats)

        df_evolution = pd.concat(evolution_data)
        df_pivot = df_evolution.pivot(index='BRAND', columns='Period', values=['Avg_Price', 'Total_Quantity'])
        
        periods = list(period_dataframes.keys())
        first_period, last_period = periods[0], periods[-1]

        df_pivot['Price_Change_%'] = ((df_pivot[('Avg_Price', last_period)] - df_pivot[('Avg_Price', first_period)]) / df_pivot[('Avg_Price', first_period)]) * 100
        df_pivot['Quantity_Change_%'] = ((df_pivot[('Total_Quantity', last_period)] - df_pivot[('Total_Quantity', first_period)]) / df_pivot[('Total_Quantity', first_period)]) * 100
        
        df_pivot['Elasticity'] = np.where(
            df_pivot['Price_Change_%'] != 0,
            df_pivot['Quantity_Change_%'] / df_pivot['Price_Change_%'],
            0
        )
        
        # ==============================================================================
        # START: FIX - FLATTEN THE MULTI-INDEX COLUMNS
        # This is the key change to solve the ValueError.
        # ==============================================================================
        
        # Create new, clean column names by joining the tuple levels
        new_cols = []
        for col in df_pivot.columns:
            # If the second level of the tuple is empty (like for 'Price_Change_%'), just use the first level.
            if col[1] == '':
                new_cols.append(col[0])
            # Otherwise, join them with an underscore.
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        
        # Assign the new, flattened column names to the DataFrame
        df_pivot.columns = new_cols

        # Now the columns are simple strings: 'Avg_Price_Period A', 'Price_Change_%', etc.
        return df_pivot.reset_index()
        # ==============================================================================
        # END: FIX
        # ==============================================================================

    def create_price_deep_dive_visuals(self, evolution_df, periods):
        """Creates a set of advanced plotly visualizations for price analysis."""
        if evolution_df.empty: return []
        first_period, last_period = periods[0], periods[-1]
        
        figs = []

        # ==============================================================================
        # START: FIX - UPDATE PLOTTING FUNCTIONS TO USE NEW STRING COLUMN NAMES
        # ==============================================================================
        
        # 1. Price vs. Quantity Change Scatter Plot (Now uses simple strings)
        fig1 = px.scatter(evolution_df, x='Price_Change_%', y='Quantity_Change_%',
                         text='BRAND', 
                         size=f'Total_Quantity_{last_period}', # <-- FIXED
                         color='Elasticity',
                         color_continuous_scale='RdBu_r', range_color=[-3, 3],
                         title=f'<b>Price vs. Quantity Change Impact ({first_period} vs. {last_period})</b>',
                         labels={'Price_Change_%': 'Price Change (%)', 'Quantity_Change_%': 'Quantity Sold Change (%)'})
        # ... (rest of fig1 code is fine) ...
        fig1.update_traces(textposition='top center', textfont_size=10)
        fig1.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
        fig1.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        fig1.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper", text="Price Up, Sales Up (Ideal)", showarrow=False, bgcolor="rgba(0,255,0,0.1)")
        fig1.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text="Price Down, Sales Up (Elastic)", showarrow=False, bgcolor="rgba(0,255,0,0.1)")
        fig1.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text="Price Up, Sales Down (Expected)", showarrow=False, bgcolor="rgba(255,0,0,0.1)")
        fig1.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper", text="Price Down, Sales Down (Problem)", showarrow=False, bgcolor="rgba(255,0,0,0.1)")
        figs.append(fig1)

        # 2. Price and Quantity Evolution Line Charts (Rewritten to work with flattened columns)
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Avg. Price Evolution per Brand", "Total Quantity Evolution per Brand"))
        
        # Get the correct column names for prices and quantities
        price_cols = [f'Avg_Price_{p}' for p in periods]
        qty_cols = [f'Total_Quantity_{p}' for p in periods]
        
        for index, row in evolution_df.iterrows():
            brand_name = row['BRAND']
            # Plot prices for the brand across periods
            fig2.add_trace(go.Scatter(x=periods, y=row[price_cols].values, name=brand_name, mode='lines+markers', legendgroup=brand_name), row=1, col=1)
            # Plot quantities for the brand across periods
            fig2.add_trace(go.Scatter(x=periods, y=row[qty_cols].values, name=brand_name, mode='lines+markers', legendgroup=brand_name, showlegend=False), row=2, col=1)

        fig2.update_layout(height=700, title_text="<b>Brand Performance Trends Across Periods</b>")
        fig2.update_yaxes(title_text="Avg. Price (DA)", row=1, col=1)
        fig2.update_yaxes(title_text="Units Sold", row=2, col=1)
        figs.append(fig2)

        # 3. Price Elasticity Bar Chart (Already uses simple columns, no changes needed here)
        elasticity_df = evolution_df.sort_values('Elasticity')
        elasticity_df['Color'] = np.where(elasticity_df['Elasticity'] < -1, 'Elastic', 
                                         np.where(elasticity_df['Elasticity'] >= 0, 'Positive/Inelastic', 'Inelastic'))
        fig3 = px.bar(elasticity_df, x='BRAND', y='Elasticity', color='Color',
                      color_discrete_map={'Elastic': '#2E86AB', 'Inelastic': '#F18F01', 'Positive/Inelastic': '#A23B72'},
                      title='<b>Price Elasticity of Demand by Brand</b>',
                      labels={'BRAND': 'Brand', 'Elasticity': 'Elasticity Score'})
        fig3.add_hline(y=-1, line_width=2, line_dash="dash", line_color="red", annotation_text="Elasticity Threshold (-1)")
        fig3.update_layout(xaxis={'categoryorder':'total descending'})
        figs.append(fig3)
        
        return figs
        # ==============================================================================
        # END: FIX
        # ==============================================================================
    @st.cache_data(ttl=3600)
    def generate_daily_deep_dive(_self, _period_dfs_tuple):
        """
        [ENHANCED & CACHED] Calculates detailed daily AND hourly metrics for all periods.
        This is the core calculation for the daily dashboard.
        """
        period_dfs = dict(_period_dfs_tuple)
        
        # Add a 'Day_Num' relative to the start of each period
        for period_name, df in period_dfs.items():
            start_date = df['Date'].min()
            df['Day_Num'] = (df['Date'] - start_date).dt.days + 1
        
        combined_df = pd.concat(period_dfs.values())
        if 'Revenue' not in combined_df.columns:
            combined_df['Revenue'] = combined_df['Quantity'] * combined_df['Product Price']
        
        def get_top_n(series, n=5): # Increased to top 5
            if series.empty: return "N/A"
            names = [str(name) for name in series.value_counts().nlargest(n).index]
            # Use markdown for a bulleted list
            return "\n".join([f"- {name[:30].strip()}" for name in names])

        # --- Aggregate metrics by Period and Day_Num ---
        daily_summary = combined_df.groupby(['Period', 'Day_Num']).agg(
            Date=('Date', 'first'),
            Revenue=('Revenue', 'sum'),
            Unique_Orders=('Order', 'nunique'),
            Top_Brands=('BRAND', get_top_n),
            Top_Products=('Item Name', get_top_n),
            Quanity=('Quantity', 'sum')
        ).reset_index()
        daily_summary['AOV'] = np.where(daily_summary['Unique_Orders'] > 0, daily_summary['Revenue'] / daily_summary['Unique_Orders'], 0)
        
        # --- NEW: Aggregate hourly data ---
        hourly_summary = combined_df.groupby(['Period', 'Day_Num', 'Hour'])['Revenue'].sum().reset_index()
        
        return daily_summary.fillna(0), hourly_summary.fillna(0)
    
def create_enhanced_dashboard_header():
    """Create enhanced dashboard header with animations"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">🛍️ E-com Analytics</h1>
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); padding: 15px; border-radius: 15px; color: white; margin: 10px auto; max-width: 800px;">
            <h3 style="margin: 0; font-weight: 300;">Advanced Business Intelligence • Real-time Analytics • ML-Powered Insights</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

def load_optimized_sample_data():
    """Create optimized sample data with realistic patterns"""
    np.random.seed(42)
    
    # Generate realistic date range with seasonal patterns
    dates = pd.date_range(start='2024-01-01', end='2025-07-30', freq='D')
    
    # ... (the rest of the setup code is correct and remains unchanged) ...
    brands_info = {
        'Brand1': {'weight': 0.25, 'price_range': (3000, 15000)},
        'Brand2': {'weight': 0.20, 'price_range': (2500, 12000)},
        'Brand3': {'weight': 0.18, 'price_range': (3500, 16000)},
        'Brand4': {'weight': 0.15, 'price_range': (4000, 14000)},
        'Brand5': {'weight': 0.12, 'price_range': (5000, 20000)},
        'Brand6': {'weight': 0.10, 'price_range': (2000, 10000)}
    }
    cities_info = {
        'Algiers': {'weight': 0.04, 'wilaya': 'DZ-16'},
        'Oran': {'weight': 0.16, 'wilaya': 'DZ-31'},
        'Constantine': {'weight': 0.12, 'wilaya': 'DZ-25'},
        'Blida': {'weight': 0.08, 'wilaya': 'DZ-09'},
        'Batna': {'weight': 0.07, 'wilaya': 'DZ-05'},
        'Annaba': {'weight': 0.06, 'wilaya': 'DZ-23'},
        'Sétif': {'weight': 0.06, 'wilaya': 'DZ-19'},
        'Sidi Bel Abbès': {'weight': 0.05, 'wilaya': 'DZ-22'},
        'Biskra': {'weight': 0.30, 'wilaya': 'DZ-07'},
        'Tébessa': {'weight': 0.04, 'wilaya': 'DZ-12'},
        'Oum El Bouaghi': {'weight': 0.02, 'wilaya': 'DZ-04'}
    }
    product_categories = {
        'Footwear': ['Shoes', 'Sneakers', 'Boots', 'Sandals', 'Running Shoes', 'Football Boots'],
        'Apparel-Top': ['T-Shirt', 'Polo', 'Jersey', 'Hoodie', 'Sweatshirt', 'Tank Top'],
        'Apparel-Bottom': ['Pants', 'Shorts', 'Leggings', 'Track Pants', 'Jeans'],
        'Accessories': ['Backpack', 'Bag', 'Cap', 'Socks', 'Gloves', 'Belt']
    }
    n_records = 1000
    sample_data = []
    customer_pool = []
    for i in range(1000):
        customer_pool.append({
            'name': f"Customer_{i}",
            'family': f"Family_{i}",
            'email': f"customer{i}@email.com",
            'phone': f"22{np.random.randint(1000000, 9999999)}",
            'city': np.random.choice(list(cities_info.keys()), 
                                   p=[cities_info[city]['weight'] for city in cities_info.keys()]),
            'loyalty_score': np.random.beta(2, 5)
        })
    
    print("Generating optimized sample data...")
    order_id_start = 362000
    current_order_id = order_id_start
    
    for i in range(n_records):
        if i % 10000 == 0:
            print(f"Generated {i}/{n_records} records...")
        
        if np.random.random() < 0.3:
            customer = np.random.choice(customer_pool, p=[c['loyalty_score']/sum(c['loyalty_score'] for c in customer_pool) for c in customer_pool])
        else:
            customer = np.random.choice(customer_pool)
        
        if np.random.random() < 0.3:
            date = np.random.choice(dates[-90:])
        else:
            date = np.random.choice(dates)
        
        # ==============================================================================
        # START: FIX FOR THE ATTRIBUTE ERROR
        # ==============================================================================
        
        # Convert the 'numpy.datetime64' object returned by np.random.choice
        # into a 'pandas.Timestamp' object, which has the .weekday() method.
        date = pd.Timestamp(date)
        
        # Now the following line will work correctly.
        if date.weekday() >= 5:  # Weekend (Saturday=5, Sunday=6)
            quantity_multiplier = 1.3
        else:
            quantity_multiplier = 1.0

        # ==============================================================================
        # END: FIX
        # ==============================================================================

        brand = np.random.choice(list(brands_info.keys()), 
                               p=[brands_info[brand]['weight'] for brand in brands_info.keys()])
        
        category = np.random.choice(list(product_categories.keys()))
        item_type = np.random.choice(product_categories[category])
        colors = ['Black', 'White', 'Blue', 'Red', 'Gray', 'Green', 'Navy']
        color = np.random.choice(colors)
        
        price_min, price_max = brands_info[brand]['price_range']
        if category == 'Footwear':
            price_multiplier = 1.2
        elif category == 'Accessories':
            price_multiplier = 0.7
        else:
            price_multiplier = 1.0
        
        product_price = int(np.random.uniform(price_min, price_max) * price_multiplier)
        
        base_quantity = max(1, int(np.random.poisson(1.5) * quantity_multiplier))
        quantity = min(base_quantity, 5)
        
        order_value = product_price * quantity
        
        city = customer['city']
        if cities_info[city]['weight'] > 0.2:
            delivery_value = np.random.choice([400, 600], p=[0.7, 0.3])
        else:
            delivery_value = np.random.choice([600, 800, 900], p=[0.5, 0.3, 0.2])
        
        status_weights = [0.65, 0.30, 0.05]
        status = np.random.choice(['En cours', 'Terminée', 'Annulée'], p=status_weights)
        
        if i % 3 == 0 or np.random.random() < 0.7:
            current_order_id += 1
        
        sample_data.append({
            'Order': str(current_order_id),
            'Status': status,
            'Date': date, # 'date' is now a pandas Timestamp object
            'Name': customer['name'],
            'Family Name': customer['family'],
            'City': city,
            'Wilaya': cities_info[city]['wilaya'],
            'E-mail': customer['email'],
            'Phone': customer['phone'],
            'Order Value': order_value,
            'Delivery Value': delivery_value,
            'Total Order Value': order_value + delivery_value,
            'UGS': f"UGS-{i}",
            'EAN': f"40{np.random.randint(10000000000, 99999999999, dtype='int64')}",
            'Article #': i + 1,
            'Item Name': f"{brand} {item_type} - {color}",
            'Quantity': quantity,
            'Product Price': product_price
        })
    
    print("Sample data generation complete!")
    return pd.DataFrame(sample_data)

def render_enhanced_kpi_cards(analytics, period_filter):
    """
    Enhanced KPI cards with trend indicators and a dynamic CAC calculation based on
    user-provided monthly budgets.
    """
    # --- 1. Filter DataFrames for Current and Previous Periods ---
    if period_filter != "All Time":
        days = int(period_filter.split()[0])
        # Define end_date for the current period
        end_date = analytics.df['Date'].max()
        # Define start_date to get the correct number of days
        start_date = end_date - timedelta(days=days - 1)
        current_df = analytics.df[(analytics.df['Date'] >= start_date) & (analytics.df['Date'] <= end_date)]
        
        # Define the previous period for comparison
        prev_end_date = start_date - timedelta(days=1)
        prev_start_date = prev_end_date - timedelta(days=days - 1)
        previous_df = analytics.df[(analytics.df['Date'] >= prev_start_date) & (analytics.df['Date'] <= prev_end_date)]
    else:
        current_df = analytics.df
        previous_df = pd.DataFrame() # No previous period for "All Time"
        start_date = current_df['Date'].min() if not current_df.empty else pd.Timestamp.now()
        end_date = current_df['Date'].max() if not current_df.empty else pd.Timestamp.now()

    # --- 2. Calculate Current Period Metrics ---
    metrics = {
        'total_revenue': current_df['Revenue'].sum(),
        'total_orders': current_df['Order'].nunique(),
        'total_customers': current_df['Customer_Name'].nunique(),
        'total_items': current_df['Quantity'].sum(),
        'conversion_rate': (current_df[current_df['Status'] == 'Terminée'].shape[0] / current_df.shape[0] * 100) if not current_df.empty else 0,
        'avg_order_value': current_df['Revenue'].sum() / current_df['Order'].nunique() if current_df['Order'].nunique() > 0 else 0,
        'return_rate': (current_df[current_df['Status'] == 'Annulée'].shape[0] / current_df.shape[0] * 100) if not current_df.empty else 0
    }

    # --- 3. NEW: DYNAMIC CAC CALCULATION ---
    total_marketing_spend = 0
    # Safely get the monthly budgets from session_state
    monthly_budgets = st.session_state.get('monthly_budgets', {})

    # Determine the months that fall within the selected date range
    # pd.period_range is perfect for this
    months_in_period = pd.period_range(start=start_date, end=end_date, freq='M')
    
    # Sum the budgets for each month found in the period
    for period in months_in_period:
        month_key = str(period)
        total_marketing_spend += monthly_budgets.get(month_key, 0) # Use .get() to avoid errors if a month has no budget

    # Calculate CAC, handling division by zero
    if metrics['total_customers'] > 0:
        metrics['customer_acquisition_cost'] = total_marketing_spend / metrics['total_customers']
    else:
        metrics['customer_acquisition_cost'] = 0 # If no customers, CAC is 0

    # --- 4. Calculate Previous Period Metrics for Comparison (Unchanged) ---
    prev_metrics = {}
    if not previous_df.empty:
        prev_metrics = {
            'total_revenue': previous_df['Revenue'].sum(),
            'total_orders': previous_df['Order'].nunique(),
            'total_customers': previous_df['Customer_Name'].nunique(),
            'total_items': previous_df['Quantity'].sum(),
            'conversion_rate': (previous_df[previous_df['Status'] == 'Terminée'].shape[0] / previous_df.shape[0] * 100) if not previous_df.empty else 0,
            'avg_order_value': previous_df['Revenue'].sum() / previous_df['Order'].nunique() if previous_df['Order'].nunique() > 0 else 0,
            'return_rate': (previous_df[previous_df['Status'] == 'Annulée'].shape[0] / previous_df.shape[0] * 100) if not previous_df.empty else 0
        }
    
    # --- 5. Display KPI Cards (The rest of your function is fine) ---
    def calculate_delta(current, previous):
        if previous == 0: return 0
        return ((current - previous) / previous) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_revenue = calculate_delta(metrics['total_revenue'], prev_metrics.get('total_revenue', 0)) if not previous_df.empty else None
        st.metric("💰 Total Revenue", f"DA {metrics['total_revenue']:,.0f}", f"{delta_revenue:+.1f}%" if delta_revenue is not None else None)
        delta_orders = calculate_delta(metrics['total_orders'], prev_metrics.get('total_orders', 0)) if not previous_df.empty else None
        st.metric("🛒 Total Orders", f"{metrics['total_orders']:,}", f"{delta_orders:+.1f}%" if delta_orders is not None else None)
    with col2:
        delta_customers = calculate_delta(metrics['total_customers'], prev_metrics.get('total_customers', 0)) if not previous_df.empty else None
        st.metric("👥 Customers", f"{metrics['total_customers']:,}", f"{delta_customers:+.1f}%" if delta_customers is not None else None)
        delta_items = calculate_delta(metrics['total_items'], prev_metrics.get('total_items', 0)) if not previous_df.empty else None
        st.metric("📦 Items Sold", f"{metrics['total_items']:,}", f"{delta_items:+.1f}%" if delta_items is not None else None)
    with col3:
        delta_aov = calculate_delta(metrics['avg_order_value'], prev_metrics.get('avg_order_value', 0)) if not previous_df.empty else None
        st.metric("💵 Avg Order Value", f"DA {metrics['avg_order_value']:,.0f}", f"{delta_aov:+.1f}%" if delta_aov is not None else None)
        delta_conversion = calculate_delta(metrics['conversion_rate'], prev_metrics.get('conversion_rate', 0)) if not previous_df.empty else None
        st.metric("📈 Conversion Rate", f"{metrics['conversion_rate']:.1f}%", f"{delta_conversion:+.1f}%" if delta_conversion is not None else None)
    with col4:
        # The CAC metric is now dynamic and has no delta comparison
        st.metric("💡 Customer Acquisition Cost", f"DA {metrics['customer_acquisition_cost']:,.0f}")
        delta_return = calculate_delta(metrics['return_rate'], prev_metrics.get('return_rate', 0)) if not previous_df.empty else None
        st.metric("⚠️ Return Rate", f"{metrics['return_rate']:.1f}%", f"{delta_return:+.1f}%" if delta_return is not None else None, delta_color="inverse")
    
    # Performance badges
    st.markdown("### 🎯 Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if metrics['conversion_rate'] > 75: st.markdown('<div class="performance-badge">🟢 High Conversion</div>', unsafe_allow_html=True)
        elif metrics['conversion_rate'] > 50: st.markdown('<div class="performance-badge">🟡 Medium Conversion</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="performance-badge">🔴 Low Conversion</div>', unsafe_allow_html=True)
    with col2:
        if metrics['return_rate'] < 5: st.markdown('<div class="performance-badge">🟢 Low Returns</div>', unsafe_allow_html=True)
        elif metrics['return_rate'] < 10: st.markdown('<div class="performance-badge">🟡 Medium Returns</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="performance-badge">🔴 High Returns</div>', unsafe_allow_html=True)
    with col3:
        if metrics['avg_order_value'] > 8000: st.markdown('<div class="performance-badge">🟢 High AOV</div>', unsafe_allow_html=True)
        elif metrics['avg_order_value'] > 5000: st.markdown('<div class="performance-badge">🟡 Medium AOV</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="performance-badge">🔴 Low AOV</div>', unsafe_allow_html=True)
    with col4:
        customer_loyalty = (current_df.groupby('Customer_Name')['Order'].nunique() > 1).mean() * 100 if not current_df.empty else 0
        if customer_loyalty > 30: st.markdown('<div class="performance-badge">🟢 High Loyalty</div>', unsafe_allow_html=True)
        elif customer_loyalty > 15: st.markdown('<div class="performance-badge">🟡 Medium Loyalty</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="performance-badge">🔴 Low Loyalty</div>', unsafe_allow_html=True)
def create_advanced_forecasting_model(analytics):
    """Advanced forecasting with multiple models"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Prepare time series data
        daily_data = analytics.df.groupby('Date').agg({
            'Revenue': 'sum',
            'Order': 'nunique',
            'Quantity': 'sum',
            'Customer_Name': 'nunique'
        }).reset_index()
        
        daily_data = daily_data.sort_values('Date')
        
        if len(daily_data) < 60:  # Need at least 60 days
            return None, "Insufficient data for advanced forecasting"
        
        # Feature engineering
        daily_data['Day_of_Year'] = daily_data['Date'].dt.dayofyear
        daily_data['Month'] = daily_data['Date'].dt.month
        daily_data['Day_of_Week'] = daily_data['Date'].dt.dayofweek
        daily_data['Is_Weekend'] = daily_data['Day_of_Week'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            daily_data[f'Revenue_lag_{lag}'] = daily_data['Revenue'].shift(lag)
            daily_data[f'Orders_lag_{lag}'] = daily_data['Order'].shift(lag)
        
        # Rolling averages
        for window in [7, 14, 30]:
            daily_data[f'Revenue_ma_{window}'] = daily_data['Revenue'].rolling(window=window).mean()
            daily_data[f'Orders_ma_{window}'] = daily_data['Order'].rolling(window=window).mean()
        
        # Remove NaN values
        daily_data = daily_data.dropna()
        
        if len(daily_data) < 30:
            return None, "Insufficient clean data for forecasting"
        
        # Prepare features
        feature_cols = [col for col in daily_data.columns if col not in ['Date', 'Revenue', 'Order', 'Quantity', 'Customer_Name']]
        X = daily_data[feature_cols]
        y_revenue = daily_data['Revenue']
        y_orders = daily_data['Order']
        
        # Train-test split (use last 20% for testing)
        split_idx = int(len(daily_data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_revenue_train, y_revenue_test = y_revenue.iloc[:split_idx], y_revenue.iloc[split_idx:]
        y_orders_train, y_orders_test = y_orders.iloc[:split_idx], y_orders.iloc[split_idx:]
        
        # Train models
        revenue_model = RandomForestRegressor(n_estimators=100, random_state=42)
        orders_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        revenue_model.fit(X_train, y_revenue_train)
        orders_model.fit(X_train, y_orders_train)
        
        # Make predictions for test set
        revenue_pred = revenue_model.predict(X_test)
        orders_pred = orders_model.predict(X_test)
        
        # Calculate accuracy metrics
        revenue_mae = mean_absolute_error(y_revenue_test, revenue_pred)
        revenue_mse = mean_squared_error(y_revenue_test, revenue_pred)
        orders_mae = mean_absolute_error(y_orders_test, orders_pred)
        
        # Future predictions (next 30 days)
        last_date = daily_data['Date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        
        # Create future features
        future_data = []
        for date in future_dates:
            # Get the most recent data point for lag features
            recent_data = daily_data.iloc[-1].copy()
            
            future_row = {
                'Day_of_Year': date.dayofyear,
                'Month': date.month,
                'Day_of_Week': date.dayofweek,
                'Is_Weekend': int(date.dayofweek in [5, 6])
            }
            
            # Use recent lag and moving average values
            for col in feature_cols:
                if col in future_row:
                    continue
                elif 'lag' in col or 'ma' in col:
                    future_row[col] = recent_data[col]
                else:
                    future_row[col] = 0
            
            future_data.append(future_row)
        
        future_df = pd.DataFrame(future_data)
        future_df = future_df[feature_cols]  # Ensure same column order
        
        # Predict future values
        future_revenue = revenue_model.predict(future_df)
        future_orders = orders_model.predict(future_df)
        
        # Create forecast results
        forecast_results = {
            'dates': future_dates,
            'revenue_forecast': future_revenue,
            'orders_forecast': future_orders,
            'revenue_mae': revenue_mae,
            'revenue_mse': revenue_mse,
            'orders_mae': orders_mae,
            'feature_importance_revenue': dict(zip(feature_cols, revenue_model.feature_importances_)),
            'feature_importance_orders': dict(zip(feature_cols, orders_model.feature_importances_)),
            'historical_data': daily_data
        }
        
        return forecast_results, "Success"
        
    except ImportError:
        return None, "Sklearn not available for advanced forecasting"
    except Exception as e:
        return None, f"Error in forecasting: {str(e)}"

def create_data_quality_monitor(analytics):
    """Comprehensive data quality monitoring"""
    df = analytics.df
    quality_report = {}
    
    # Basic data quality metrics
    quality_report['total_records'] = len(df)
    quality_report['total_columns'] = len(df.columns)
    quality_report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    quality_report['missing_data'] = {
        'columns_with_missing': (missing_data > 0).sum(),
        'worst_column': missing_pct.idxmax() if missing_pct.max() > 0 else 'None',
        'worst_percentage': missing_pct.max(),
        'details': dict(zip(missing_data.index, missing_pct.values))
    }
    
    # Duplicate analysis
    quality_report['duplicates'] = {
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_orders': df['Order'].duplicated().sum() if 'Order' in df.columns else 0,
        'duplicate_customers': df['Customer_Name'].duplicated().sum() if 'Customer_Name' in df.columns else 0
    }
    
    # Data consistency checks
    quality_report['consistency'] = {}
    
    # Price consistency
    if 'Product Price' in df.columns:
        price_outliers = df[df['Product Price'] > df['Product Price'].quantile(0.99)]
        quality_report['consistency']['price_outliers'] = len(price_outliers)
        quality_report['consistency']['negative_prices'] = (df['Product Price'] < 0).sum()
    
    # Quantity consistency
    if 'Quantity' in df.columns:
        quality_report['consistency']['zero_quantity'] = (df['Quantity'] == 0).sum()
        quality_report['consistency']['high_quantity'] = (df['Quantity'] > 10).sum()
    
    # Date consistency
    if 'Date' in df.columns:
        quality_report['consistency']['future_dates'] = (df['Date'] > pd.Timestamp.now()).sum()
        date_range = df['Date'].max() - df['Date'].min()
        quality_report['consistency']['date_range_days'] = date_range.days
    
    # Data completeness score
    completeness_scores = []
    critical_columns = ['Order', 'Date', 'Product Price', 'Quantity', 'Customer_Name']
    for col in critical_columns:
        if col in df.columns:
            completeness = (1 - df[col].isnull().sum() / len(df)) * 100
            completeness_scores.append(completeness)
    
    quality_report['overall_quality_score'] = np.mean(completeness_scores) if completeness_scores else 0
    
    # Recommendations
    recommendations = []
    if quality_report['missing_data']['worst_percentage'] > 10:
        recommendations.append(f"⚠️ Column '{quality_report['missing_data']['worst_column']}' has {quality_report['missing_data']['worst_percentage']:.1f}% missing data")
    
    if quality_report['duplicates']['duplicate_rows'] > 0:
        recommendations.append(f"🔄 Found {quality_report['duplicates']['duplicate_rows']} duplicate rows - consider deduplication")
    
    if quality_report['consistency'].get('negative_prices', 0) > 0:
        recommendations.append("💰 Found negative prices - data validation needed")
    
    if quality_report['consistency'].get('future_dates', 0) > 0:
        recommendations.append("📅 Found future dates - check data entry process")
    
    quality_report['recommendations'] = recommendations
    
    return quality_report

def main():
    """Enhanced main application with performance monitoring"""
    create_enhanced_dashboard_header()
    
    app_start_time = time.time()
    
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Data File", 
        type=['xlsx', 'xls', 'csv'],
        help="Supports Excel and CSV files up to 200MB"
    )
    
    # ==============================================================================
    # START: REPLACE your current data loading block with this new pipeline
    # ==============================================================================

    # --- STAGE 1: Load Raw Data (Cached) ---
    raw_df = load_raw_data(uploaded_file)
    if raw_df is None:
        st.warning("Please upload a data file to begin.")
        st.stop()
        
    # --- STAGE 2: Process and Optimize Data (Cached) ---
    # We pass the raw_df to the processing function. The result (processed_df) is also cached.
    try:
        # A single spinner provides a clean user experience
        with st.spinner("Processing and optimizing data... (This is cached and will be instant on refresh)"):
            processed_df, start_mem, end_mem = process_and_optimize_data(raw_df)
    except Exception as e:
        st.error(f"❌ An error occurred during data processing: {e}")
        st.error("Please check if the columns in your file match the required format: Order, Status, Date, Name, etc.")
        st.stop()

    # --- FINAL STEP: Instantiate the Analytics Container ---
    # This is fast and happens on every run, avoiding caching issues with complex objects.
    analytics = PerformanceOptimizedAnalytics(processed_df, raw_df)

    # --- Display Performance Metrics ---
    # Use session_state to show the message only once per data load.
    if 'perf_metrics_displayed' not in st.session_state:
        if start_mem > 0:
            reduction_pct = (start_mem - end_mem) / start_mem * 100
            st.sidebar.success(f"Memory Optimized: {start_mem:.1f}MB → {end_mem:.1f}MB ({reduction_pct:.1f}%)")
        st.session_state.perf_metrics_displayed = True
    
    # ==============================================================================
    # END: REPLACEMENT BLOCK
    # ==============================================================================

    
    if analytics is None:
        st.stop()
    
    # Enhanced sidebar with real-time stats
    st.sidebar.markdown("## 📊 Real-time Data Overview")
    
    # Data quality indicator
    quality_report = create_data_quality_monitor(analytics)
    quality_score = quality_report['overall_quality_score']
    
    if quality_score >= 90:
        quality_color = "green"
        quality_status = "Excellent"
    elif quality_score >= 75:
        quality_color = "orange"
        quality_status = "Good"
    else:
        quality_color = "red"
        quality_status = "Needs Attention"
    
    st.sidebar.markdown(f"""
    <div style="padding: 10px; border-radius: 10px; background: linear-gradient(135deg, {quality_color}, #555); color: white; margin: 10px 0;">
        <strong>Data Quality Score</strong><br>
        <span style="font-size: 24px;">{quality_score:.1f}%</span><br>
        <small>{quality_status}</small>
    </div>
    """, unsafe_allow_html=True)
    
# ==============================================================================
    # START: FIX FOR THE KEYERROR
    # ==============================================================================

    # Performance metrics
    # The `app_start_time` is defined at the very top of your main() function.
    # The `total_load_time` will represent the time taken for both cached stages.
    total_load_time = time.time() - app_start_time
    
    # We get the memory usage directly from the analytics object, which is correct.
    memory_usage_mb = analytics.performance_metrics['memory_usage_mb']
    
    st.sidebar.markdown("### ⚡ Performance Metrics")
    st.sidebar.write(f"**Total Load & Process Time:** {total_load_time:.2f}s")
    st.sidebar.write(f"**Memory Usage:** {memory_usage_mb:.1f} MB")
    st.sidebar.write(f"**Records:** {quality_report['total_records']:,}")
    st.sidebar.write(f"**Columns:** {quality_report['total_columns']}")

    # ==============================================================================
    # END: FIX
    # ==============================================================================
    
    # Data overview statistics
    st.sidebar.markdown("### 📈 Quick Stats")
    total_revenue = analytics.df['Revenue'].sum()
    unique_customers = analytics.df['Customer_Name'].nunique()
    unique_brands = analytics.df['BRAND'].nunique()
    date_range = analytics.df['Date'].max() - analytics.df['Date'].min()
    
    st.sidebar.write(f"**Total Revenue:** DA {total_revenue:,.0f}")
    st.sidebar.write(f"**Date Range:** {date_range.days} days")
    st.sidebar.write(f"**Unique Customers:** {unique_customers:,}")
    st.sidebar.write(f"**Brands:** {unique_brands}")
    st.sidebar.write(f"**Locations:** {analytics.df['Wilaya'].nunique()}")
    
    # Advanced filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Advanced Filters")
    
    period_options = ["All Time", "7 Days", "30 Days", "60 Days", "90 Days", "180 Days", "365 Days", "730 Days"]
    period_filter = st.sidebar.selectbox("📅 Time Period", period_options)
    
    # Brand filter with search
    unique_brands_list = sorted([str(brand) for brand in analytics.df['BRAND'].dropna().unique()])
    brand_filter = st.sidebar.multiselect(
        "🏷️ Brands", 
        options=['All'] + unique_brands_list,
        default=['All']
    )
    
    # Status filter
    status_filter = st.sidebar.multiselect(
        "📦 Order Status",
        options=['All'] + sorted(analytics.df['Status'].unique()),
        default=['All']
    )
    
    # Price range filter
    min_price, max_price = float(analytics.df['Product Price'].min()), float(analytics.df['Product Price'].max())
    price_range = st.sidebar.slider(
        "💰 Price Range (DA)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=100.0
    )
    
    # Customer segment filter
    customer_segments = sorted([str(seg) for seg in analytics.df['Customer_Value_Segment'].unique()])
    segment_filter = st.sidebar.multiselect(
        "👥 Customer Segments",
        options=['All'] + customer_segments,
        default=['All']
    )
    
    # Apply filters
    filtered_df = analytics.df.copy()
    
    # Time period filter
    if period_filter != "All Time":
        days = int(period_filter.split()[0])
        cutoff_date = analytics.df['Date'].max() - timedelta(days=days)
        filtered_df = filtered_df[filtered_df['Date'] >= cutoff_date]
    
    # Brand filter
    if 'All' not in brand_filter and brand_filter:
        filtered_df = filtered_df[filtered_df['BRAND'].isin(brand_filter)]
    
    # Status filter
    if 'All' not in status_filter and status_filter:
        filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
    
    # Price range filter
    filtered_df = filtered_df[
        (filtered_df['Product Price'] >= price_range[0]) & 
        (filtered_df['Product Price'] <= price_range[1])
    ]
    
    # Customer segment filter
    if 'All' not in segment_filter and segment_filter:
        filtered_df = filtered_df[filtered_df['Customer_Value_Segment'].isin(segment_filter)]
    
    # Update analytics with filtered data
    analytics.df = filtered_df
    
    # Show filter impact
    if len(filtered_df) < len(analytics.original_df):
        reduction_pct = (1 - len(filtered_df) / len(analytics.original_df)) * 100
        st.sidebar.info(f"Filters applied: {len(filtered_df):,} records ({reduction_pct:.1f}% filtered out)")

    st.sidebar.markdown("---")
    with st.sidebar.expander("💰 Set Monthly Marketing Budgets", expanded=False):
        st.info("Enter your marketing spend for each month to calculate an accurate Customer Acquisition Cost (CAC).")
        
        # Initialize the budget dictionary in session_state if it doesn't exist
        if 'monthly_budgets' not in st.session_state:
            st.session_state.monthly_budgets = {}
            
        # Get all unique Year-Month periods from the dataset
        # Using .dt.to_period('M') is robust for this
        unique_months = sorted(analytics.df['Date'].dt.to_period('M').unique())
        
        # Create a number input for each month
        for period in unique_months:
            # Create a string key for the dictionary, e.g., '2024-01'
            month_key = str(period)
            
            # Get the current value from session_state, or use a default (e.g., 50000)
            current_budget = st.session_state.monthly_budgets.get(month_key, 50000)
            
            # Create the number input widget
            new_budget = st.number_input(
                label=f"Budget for {period.strftime('%Y-%m')}",
                value=current_budget,
                min_value=0,
                step=1000,
                key=f"budget_{month_key}" # Unique key for each widget
            )
            
            # Update the session_state with the new value if it changes
            st.session_state.monthly_budgets[month_key] = new_budget
    
    
    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=[
            "📊 Executive Dashboard", 
            "📈 Sales Analytics", 
            "🏷️ Brand Intelligence", 
            "🗺️ Geographic Insights", 
            "👥 Customer Intelligence", 
            "👤 RFM K-Means", 
            "🛤️ Customer Journey",         # <--- ADD NEW OPTION
            "🚨 Anomaly Detection",
            "📦 Product Analytics", 
            "⏰ Temporal Analysis", 
            "🔍 Advanced Insights", 
            "📅 Smart Comparator", 
            "🤖 AI & Forecasting",
            "📋 Data Quality Monitor",
            "🐍 Code Sandbox"
        ],
        icons=[
            "speedometer2", "graph-up", "tags", "geo-alt", 
            "people", "box", "clock", "lightbulb", 
            "arrow-left-right", "robot", "shield-check", "person-badge","signpost-split",             # <--- ADD MATCHING ICON
            "exclamation-triangle", "code-slash"
        ],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "20px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "center", 
                "margin": "0px", 
                "--hover-color": "#eee"
            },
            "nav-link-selected": {"background-color": "#02ab21"},
        }
    )

    if 'sandbox_df' in st.session_state:
        # To maintain consistency, we create a new analytics object with the sandbox data
        # but keep the original_df from the first load for reset purposes.
        analytics = PerformanceOptimizedAnalytics(st.session_state.sandbox_df, analytics.original_df)
        
        # Display a persistent banner to remind the user they are viewing modified data
        st.warning(
            "Viewing a modified DataFrame from the Code Sandbox. Go to the sandbox to reset.",
            icon="⚠️"
        )

    # Main content based on selection
    if selected == "📊 Executive Dashboard":
        st.header("📊 Executive Dashboard")
        
        # Enhanced KPI Cards
        render_enhanced_kpi_cards(analytics, period_filter)
        
        # Interactive dashboard overview
        viz_engine = AdvancedVisualizationEngine(analytics)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Interactive Business Overview")
            dashboard_fig = viz_engine.create_interactive_dashboard_overview()
            st.plotly_chart(dashboard_fig, use_container_width=True)
        
        with col2:
            st.subheader("💡 Executive Insights")
            
            # Key insights
            top_brand = analytics.df.groupby('BRAND')['Revenue'].sum().idxmax()
            top_brand_share = (analytics.df.groupby('BRAND')['Revenue'].sum().max() / analytics.df['Revenue'].sum()) * 100
            
            top_customer = analytics.df.groupby('Customer_Name')['Revenue'].sum().idxmax()
            top_customer_value = analytics.df.groupby('Customer_Name')['Revenue'].sum().max()
            
            avg_order_size = analytics.df.groupby('Order')['Quantity'].sum().mean()
            
            st.markdown(f"""
            <div class="premium-card">
                <h4>🏆 Market Leader</h4>
                <p><strong>{top_brand}</strong> dominates with {top_brand_share:.1f}% market share</p>
            </div>
            
            <div class="premium-card">
                <h4>⭐ Top Customer</h4>
                <p><strong>{top_customer}</strong><br>Lifetime Value: DA {top_customer_value:,.0f}</p>
            </div>
            
            <div class="premium-card">
                <h4>📦 Order Insights</h4>
                <p>Average order size: <strong>{avg_order_size:.1f} items</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick action buttons
        st.markdown("---")
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Generate Executive Report", use_container_width=True):
                st.success("Generating comprehensive executive report...")
                # Report generation logic would go here
        
        with col2:
            if st.button("📧 Email Insights", use_container_width=True):
                st.success("Preparing insights email...")
        
        with col3:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col4:
            if st.button("⚙️ Export Settings", use_container_width=True):
                st.success("Exporting dashboard settings...")
    
    elif selected == "👥 Customer Intelligence":
        st.header("🧠 Intelligent Customer Segmentation with GMM")
        st.markdown("Leverage advanced machine learning to uncover deep insights into your customer base, assign value scores, and identify transition probabilities between segments.")

        segmentation_engine = IntelligentCustomerSegmentation(analytics)

        # Step 1: BIC plot to find optimal clusters
        st.subheader("Step 1: Find the Optimal Number of Segments")
        st.info("The chart below shows the BIC score for different numbers of segments. The 'elbow' or the point where the line starts to flatten suggests a good number of segments for your data. **Look for the lowest point.**")
        with st.spinner("Calculating BIC scores for cluster optimization..."):
            bic_fig = segmentation_engine.find_optimal_clusters()
            st.plotly_chart(bic_fig, use_container_width=True)
        
        # Step 2: User runs segmentation
        st.subheader("Step 2: Generate Customer Segments & Ratings")
        n_clusters = st.slider("Select the number of segments based on the BIC plot:", min_value=2, max_value=10, value=6, step=1)

        if st.button("🚀 Generate Segments and Value Scores", type="primary", use_container_width=True):
            with st.spinner(f"Running GMM with {n_clusters} segments..."):
                segmented_data, cluster_profiles = segmentation_engine.perform_segmentation(n_clusters)
                st.session_state.segmented_data_gmm = segmented_data
                st.session_state.cluster_profiles_gmm = cluster_profiles
                st.success("Segmentation Complete!")

        # Step 3: Display results if they exist
        if 'segmented_data_gmm' in st.session_state and not st.session_state.segmented_data_gmm.empty:
            segmented_data = st.session_state.segmented_data_gmm
            cluster_profiles = st.session_state.cluster_profiles_gmm
            
            st.subheader("✅ Segmentation Results: Segment Profiles")
            st.dataframe(cluster_profiles.style.format("{:,.2f}").background_gradient(cmap='viridis', subset=['Customer_Value_Score']))
            
            # --- NEW: Call the dynamic explanation function ---
            explain_gmm_segments(cluster_profiles)
            
            # Display visualizations
            segmentation_engine.create_visualizations(segmented_data, cluster_profiles)
            
            # --- NEW: Interactive Export Tool ---
            st.markdown("---")
            st.subheader("📥 Export Customer Lists by Segment")
            st.markdown("Select a segment from the dropdown to preview their data and download their contact list for targeted marketing campaigns.")

            segment_options = ['All Segments'] + sorted(cluster_profiles.index.tolist())
            selected_segment_export = st.selectbox("Select a segment to export:", segment_options)

            if selected_segment_export == 'All Segments':
                export_df = segmented_data
            else:
                export_df = segmented_data[segmented_data['Segment'] == selected_segment_export]

            export_cols = ['Customer_Name', 'Email', 'Phone', 'Segment', 'Customer_Value_Score', 'Rating_Tier', 'Recency', 'Frequency', 'Monetary']
            final_export_data = export_df[export_cols].copy().sort_values('Customer_Value_Score', ascending=False)
            
            st.markdown(f"**Previewing: {selected_segment_export} ({len(final_export_data)} customers)**")
            st.dataframe(final_export_data.style.format({'Monetary': 'DA {:,.2f}', 'Customer_Value_Score': '{:,.1f}'}), use_container_width=True)
            
            csv_data = final_export_data.to_csv(index=False).encode('utf-8')
            dynamic_filename = f"customer_segment_{selected_segment_export.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
            
            st.download_button(
                label=f"📥 Download List for '{selected_segment_export}'",
                data=csv_data,
                file_name=dynamic_filename,
                mime='text/csv',
                use_container_width=True
            )
    # Add this entire block to your main() function, after the Customer Intelligence block
    # ==============================================================================
    # START: Replace the old "RFM K-Means" block with this new one
    # ==============================================================================
    elif selected == "👤 RFM K-Means":
        st.header("👤 RFM K-Means Segmentation")
        
        rfm_engine = RFM_KMeans_Segmentation(analytics)

        # --- NEW: Mode Selector ---
        segmentation_mode = st.radio(
            "Choose a segmentation method:",
            ("Automatic (Let the algorithm find groups)", "Goal-Oriented (Find customers who match my goals)"),
            horizontal=True,
            help="**Automatic:** Finds natural clusters in your data. **Goal-Oriented:** Finds customers that fit specific personas you choose."
        )

        st.markdown("---")

        if segmentation_mode == "Automatic (Let the algorithm find groups)":
            st.subheader("Step 1: Find the Optimal Number of Segments")
            st.info("The 'elbow' in the chart below suggests an optimal number of clusters for your data.")
            with st.spinner("Calculating Elbow Method plot..."):
                elbow_fig = rfm_engine.find_optimal_k()
                st.plotly_chart(elbow_fig, use_container_width=True)

            st.subheader("Step 2: Create Customer Segments")
            n_clusters_rfm = st.slider("Select the number of clusters (k):", min_value=2, max_value=10, value=5, key="rfm_k_slider_auto")
            if st.button("🚀 Generate Automatic Segments", type="primary", use_container_width=True):
                with st.spinner(f"Running K-Means with {n_clusters_rfm} clusters..."):
                    segmented_rfm_data, rfm_summary = rfm_engine.perform_automatic_segmentation(n_clusters_rfm)
                    st.session_state.segmented_rfm_data = segmented_rfm_data
                    st.session_state.rfm_summary = rfm_summary
                    st.success("Automatic Segmentation Complete!")

        else: # Goal-Oriented Mode
            st.subheader("Step 1: Choose Your Target Segments")
            st.info("Select the customer 'personas' you want to find. We will classify every customer into the persona they match most closely.")
            
            available_personas = list(rfm_engine.segment_prototypes.keys())
            
            desired_segments = st.multiselect(
                "Select your desired segments:",
                options=available_personas,
                default=["Best Customers", "At Risk", "Lost Cause"]
            )
            
            st.subheader("Step 2: Generate Goal-Oriented Segments")
            if st.button("🎯 Find Matching Customers", type="primary", use_container_width=True):
                with st.spinner("Classifying customers based on your goals..."):
                    segmented_rfm_data, rfm_summary = rfm_engine.perform_goal_oriented_segmentation(desired_segments)
                    st.session_state.segmented_rfm_data = segmented_rfm_data
                    st.session_state.rfm_summary = rfm_summary
                    st.success("Goal-Oriented Segmentation Complete!")

        # --- Step 3: Display results (this part is now common for both modes) ---
        if 'segmented_rfm_data' in st.session_state and not st.session_state.segmented_rfm_data.empty:
            segmented_rfm_data = st.session_state.segmented_rfm_data
            rfm_summary = st.session_state.rfm_summary
            
            st.subheader("✅ Segmentation Results: Segment Profiles")
            st.dataframe(rfm_summary.style.format({'Total_Revenue': 'DA {:,.2f}', 'Avg_Monetary': 'DA {:,.2f}'}).background_gradient(cmap='Greens', subset=['Total_Revenue']))
            
            rfm_engine.create_visualizations(segmented_rfm_data)
            
            st.subheader("📥 View & Export RFM Segments")
            # The interactive export tool will work perfectly with either mode's output
            segment_list = sorted(segmented_rfm_data['Segment'].unique())
            selectbox_options = ['All Segments'] + segment_list
            selected_segment_to_view = st.selectbox("Choose a segment to view and export:", selectbox_options)
            if selected_segment_to_view == 'All Segments': export_df_filtered = segmented_rfm_data
            else: export_df_filtered = segmented_rfm_data[segmented_rfm_data['Segment'] == selected_segment_to_view]
            export_cols_rfm = ['Customer_Name', 'Email', 'Phone', 'Last_Purchase_Date', 'Wilaya', 'Segment', 'Recency', 'Frequency', 'Monetary']
            final_export_data = export_df_filtered[export_cols_rfm].copy().sort_values('Monetary', ascending=False)
            st.markdown(f"**Previewing: {selected_segment_to_view} ({len(final_export_data)} customers)**")
            st.dataframe(final_export_data.style.format({'Monetary': 'DA {:,.2f}', 'Last_Purchase_Date': '{:%Y-%m-%d}'}), use_container_width=True)
            csv_data_rfm = final_export_data.to_csv(index=False).encode('utf-8')
            dynamic_filename = f"rfm_segment_{selected_segment_to_view.replace(' ', '_')}.csv"
            st.download_button(label=f"📥 Download List for '{selected_segment_to_view}'", data=csv_data_rfm, file_name=dynamic_filename, mime='text/csv', use_container_width=True)

    # ==============================================================================
    # END: The replacement block ends here
    # ==============================================================================

    # Add this entire block for the Customer Journey Analytics page
    # ==============================================================================
    # START: Replace your entire "Customer Journey" block with this one
    # ==============================================================================
    elif selected == "🛤️ Customer Journey":
        st.header("🛤️ Customer Journey & Lifecycle Analytics")
        st.info("💡 **Pro Tip:** To ensure a smooth experience with your large dataset, calculations for each tab are performed **on-demand** the first time you click it. You may see a brief 'running...' indicator on the first view of each tab.")

        # Initialize the analytics class
        journey_analyzer = CustomerJourneyAnalytics(analytics)

        # Create the tabs
        tab1, tab2, tab3 = st.tabs(["📈 Lifecycle Stages", "🔄 Transitions & Funnel", "🔗 Brand Loyalty"])

        # --- Tab 1: Lifecycle Stages (Default View) ---
        with tab1:
            st.subheader("Customer Lifecycle Distribution")
            st.info("This dashboard categorizes your entire customer base into distinct stages based on their purchasing recency, frequency, and age. Use the glossary and export tool below to understand and act on these insights.")
            
            # --- 1. Calculation (LAZY LOADING) ---
            if 'lifecycle_data' not in st.session_state:
                with st.spinner("Analyzing customer lifecycle stages... (This runs only once)"):
                    # We need more than just the stages for the export, so we grab more columns here.
                    # The CustomerJourneyAnalytics class already provides this data.
                    st.session_state.lifecycle_data = journey_analyzer.create_customer_lifecycle_stages()

            lifecycle_data = st.session_state.lifecycle_data
            stage_counts = lifecycle_data['Lifecycle_Stage'].value_counts().reset_index()
            stage_counts.columns = ['Lifecycle Stage', 'Number of Customers']
            
            # --- 2. Visualization & Metrics ---
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Stage Counts")
                st.dataframe(stage_counts)
                
                # Key Metrics
                total_customers = len(lifecycle_data)
                champion_pct = (stage_counts[stage_counts['Lifecycle Stage'] == 'Champion']['Number of Customers'].sum() / total_customers) * 100
                lost_pct = (stage_counts[stage_counts['Lifecycle Stage'] == 'Lost']['Number of Customers'].sum() / total_customers) * 100
                st.success(f"**Champions:** {champion_pct:.1f}% are your most valuable, active customers.")
                st.warning(f"**Lost Customers:** {lost_pct:.1f}% are considered churned and require win-back campaigns.")

            with col2:
                fig_pie = px.pie(stage_counts, 
                                 values='Number of Customers', 
                                 names='Lifecycle Stage', 
                                 title="Customer Lifecycle Stages", 
                                 hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("---") # Visual separator

            # --- 3. Glossary / Definitions ---
            with st.expander("📘 What do these lifecycle stages mean? (Glossary)", expanded=False):
                st.markdown("""
                - **Champion:** 🏆 Your best customers. They are highly active, purchase frequently, and have been customers for over a month. *Action: Reward them, ask for reviews, create lookalike audiences.*
                - **Loyal Customer:** 💖 High-value customers who purchase regularly. They are a core part of your business. *Action: Nurture with loyalty programs and exclusive offers.*
                - **Active Customer:** 👍 Customers who have made a purchase within the last 30 days but are not yet champions or loyalists. *Action: Encourage repeat purchases with timely follow-ups.*
                - **New Customer:** ✨ Customers who made their first purchase within the last 30 days. This is a critical period for retention. *Action: Provide an excellent onboarding experience and a compelling reason to return.*
                - **At Risk - High Value:** 😟 High-value customers who haven't purchased in over a month (31-90 days). Losing them would be costly. *Action: Launch targeted re-engagement campaigns immediately.*
                - **At Risk:** 😥 Regular customers who haven't purchased in 31-90 days. They are slipping away. *Action: Use discounts or "we miss you" campaigns to win them back.*
                - **Hibernating:** 😴 Customers who haven't purchased in a long time (91-180 days). They are inactive but might return. *Action: Consider a low-cost, long-term nurturing sequence.*
                - **Lost:** 💔 Customers who have been inactive for over 180 days. They are considered churned. *Action: Use aggressive "win-back" offers or survey them to find out why they left.*
                """)

            st.markdown("---") # Visual separator

            # --- 4. Interactive Export Tool ---
            st.subheader("📥 Export Customer Lists by Stage")
            
            # Get the list of available stages for the dropdown
            available_stages = sorted(lifecycle_data['Lifecycle_Stage'].unique())
            
            # Create the selectbox for the user
            selected_stage = st.selectbox(
                "Select a lifecycle stage to download:",
                available_stages
            )
            
            if selected_stage:
                # Filter the dataframe to get only the customers in the selected stage
                stage_df = lifecycle_data[lifecycle_data['Lifecycle_Stage'] == selected_stage].copy()
                
                # We need to get the Email and Phone number from the main dataframe.
                # We can do this with an efficient merge.
                contact_info = analytics.df[['Customer_Name', 'E-mail', 'Phone']].drop_duplicates(subset=['Customer_Name'])
                
                # Merge the contact info into our filtered stage dataframe
                export_df = pd.merge(stage_df, contact_info, on='Customer_Name', how='left')

                # Select and rename columns for a clean export file
                export_cols = ['Customer_Name', 'E-mail', 'Phone', 'Total_Revenue', 'Days_Since_Last_Purchase']
                final_export_df = export_df[export_cols].rename(columns={
                    'Customer_Name': 'Name',
                    'E-mail': 'Email',
                    'Phone': 'Phone Number',
                    'Total_Revenue': 'Total Revenue (DA)',
                    'Days_Since_Last_Purchase': 'Days Since Last Purchase'
                }).sort_values('Total Revenue (DA)', ascending=False)
                
                st.markdown(f"**Preview for '{selected_stage}' ({len(final_export_df)} customers):**")
                st.dataframe(
                    final_export_df.style.format({'Total Revenue (DA)': 'DA {:,.0f}'}), 
                    use_container_width=True
                )
                
                # Prepare the data for CSV download
                csv_data = final_export_df.to_csv(index=False).encode('utf-8')
                
                # Create the download button
                st.download_button(
                    label=f"📥 Download List for '{selected_stage}'",
                    data=csv_data,
                    file_name=f"lifecycle_stage_{selected_stage.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
        # --- Tab 2: Transitions & Funnel ---
        with tab2:
            st.subheader("Customer Funnel & Lifecycle Transitions")

            # LAZY LOADING LOGIC for the funnel (it's fast, but good practice)
            if 'funnel_data' not in st.session_state:
                st.session_state.funnel_data = journey_analyzer.create_purchase_funnel()
            funnel_data = st.session_state.funnel_data
            
            # LAZY LOADING LOGIC for the HEAVY transition analysis
            if 'transition_matrix' not in st.session_state:
                with st.spinner("Analyzing customer transitions... (This runs only once)"):
                    _, st.session_state.transition_matrix = journey_analyzer.analyze_customer_transitions()
            transition_matrix = st.session_state.transition_matrix

            # Display Funnel
            fig_funnel = go.Figure(go.Funnel(y=funnel_data['Stage'], x=funnel_data['Count'], textinfo="value+percent initial"))
            fig_funnel.update_layout(title="Simulated Customer Purchase Funnel")
            st.plotly_chart(fig_funnel, use_container_width=True)
            
            # Display Transition Matrix
            if not transition_matrix.empty:
                transition_percentages = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0) * 100
                fig_heatmap = px.imshow(transition_percentages, text_auto=".1f", labels=dict(x="Current Stage", y="Previous Stage", color="Transition %"), title="Customer Lifecycle Transition Matrix (%)")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("Insufficient data for transition analysis. Requires more than 180 days of data.")

        # --- Tab 3: Brand Loyalty ---
        with tab3:
            st.subheader("Brand Loyalty & Switching Behavior")
            st.info("Understand how loyal your customers are to specific brands. Use the controls below to filter the flow diagram and focus on the most significant brand-switching patterns.")

            # LAZY LOADING LOGIC
            if 'brand_loyalty_data' not in st.session_state:
                with st.spinner("Analyzing brand switching patterns... (This runs only once)"):
                    _, switch_matrix, brand_loyalty = journey_analyzer.analyze_brand_switching()
                    st.session_state.brand_loyalty_data = {'matrix': switch_matrix, 'loyalty_rate': brand_loyalty}

            loyalty_data = st.session_state.brand_loyalty_data
            switch_matrix = loyalty_data['matrix']
            brand_loyalty = loyalty_data['loyalty_rate']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Brand Loyalty Rate", f"{brand_loyalty:.1f}%")
                st.caption("Percentage of customers who have only ever purchased one brand.")
                if not switch_matrix.empty:
                    st.markdown("#### Top 5 Brand Switches:")
                    st.dataframe(switch_matrix.nlargest(5, 'Switches'))
                else:
                    st.success("Very high brand loyalty detected with minimal switching.")
            
            with col2:
                if not switch_matrix.empty:
                    # --- INTERACTIVE CONTROLS FOR THE SANKEY DIAGRAM ---
                    st.markdown("#### Diagram Controls")
                    all_brands_list = pd.unique(switch_matrix[['From_Brand', 'To_Brand']].values.ravel('K'))
                    top_n_brands = st.slider(
                        "Show Top N Brands:",
                        min_value=5, max_value=min(30, len(all_brands_list)), value=10, step=1,
                        help="Adjust the number of top brands to include in the diagram."
                    )
                    min_switches = st.number_input(
                        "Minimum Number of Switches to Show:",
                        min_value=1, value=max(1, int(switch_matrix['Switches'].quantile(0.25))), step=1,
                        help="Hide connections with fewer switches than this value to reduce clutter."
                    )
                    
                    # --- FILTER THE DATA BASED ON USER CONTROLS ---
                    from_counts = switch_matrix.groupby('From_Brand')['Switches'].sum()
                    to_counts = switch_matrix.groupby('To_Brand')['Switches'].sum()
                    total_involvement = from_counts.add(to_counts, fill_value=0).nlargest(top_n_brands)
                    top_brands_list = total_involvement.index.tolist()
                    
                    filtered_matrix = switch_matrix[
                        (switch_matrix['From_Brand'].isin(top_brands_list)) &
                        (switch_matrix['To_Brand'].isin(top_brands_list)) &
                        (switch_matrix['Switches'] >= min_switches)
                    ]
                    
                    if filtered_matrix.empty:
                        st.warning("No data matches the current filter settings. Try lowering the minimum switches or increasing the number of brands.")
                    else:
                        # --- ENHANCED SANKEY DIAGRAM WITH COLORS AND HOVER INFO ---
                        display_brands = pd.unique(filtered_matrix[['From_Brand', 'To_Brand']].values.ravel('K'))
                        brand_map = {brand: i for i, brand in enumerate(display_brands)}
                        
                        # 1. Create a color palette
                        colors = px.colors.qualitative.G10 # A great palette for distinct categories
                        brand_color_map = {brand: colors[i % len(colors)] for i, brand in enumerate(display_brands)}
                        
                        # Helper function to add opacity to colors for the links
                        def hex_to_rgba(hex_color, opacity=0.6):
                            hex_color = hex_color.lstrip('#')
                            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'

                        # 2. Prepare node and link color lists
                        node_colors = [brand_color_map[brand] for brand in display_brands]
                        link_colors = [hex_to_rgba(brand_color_map[source_brand]) for source_brand in filtered_matrix['From_Brand']]

                        fig_sankey = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=25,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=display_brands,
                                color=node_colors
                            ),
                            link=dict(
                                source=[brand_map[s] for s in filtered_matrix['From_Brand']],
                                target=[brand_map[t] for t in filtered_matrix['To_Brand']],
                                value=filtered_matrix['Switches'],
                                color=link_colors,
                                # THE FIX: Move 'hovertemplate' inside the 'link' dictionary
                                hovertemplate='Switch from %{source.label} to %{target.label}: %{value} customers<extra></extra>'
                            )
                            # The hovertemplate is REMOVED from this level.
                        )])
                        
                        fig_sankey.update_layout(
                            title_text=f"Brand Switching Flow (Top {top_n_brands} Brands, Min {min_switches} Switches)",
                            font_size=12,
                            height=600
                        )
                        st.plotly_chart(fig_sankey, use_container_width=True)
                else:
                    st.info("High brand loyalty - not enough switching data to build a flow diagram.")
    # ==============================================================================
    # END: The replacement block ends here
    # ==============================================================================


    # Add this entire block for the Anomaly Detection page
    elif selected == "🚨 Anomaly Detection":
        st.header("🚨 Anomaly Detection Dashboard")
        st.markdown("Use machine learning (Isolation Forest) to automatically identify unusual patterns in your daily revenue and individual customer behavior that might require investigation.")

        # Initialize the analytics class
        anomaly_detector = AnomalyDetection(analytics)

        with st.spinner("Running anomaly detection models on daily and customer data..."):
            daily_anomalies, customer_anomalies = anomaly_detector.detect_revenue_anomalies(), anomaly_detector.detect_customer_anomalies()
            alerts = anomaly_detector.generate_anomaly_alerts(daily_anomalies, customer_anomalies)

        # Display alerts
        if alerts:
            st.subheader("Actionable Alerts")
            for alert in alerts:
                if alert['type'] == 'warning':
                    st.warning(f"**{alert['title']}**: {alert['message']}. **Suggestion:** {alert['action']}.")
                else:
                    st.info(f"**{alert['title']}**: {alert['message']}. **Suggestion:** {alert['action']}.")
        
        # Create tabs for revenue and customer anomalies
        tab1, tab2 = st.tabs(["📈 Daily Revenue Anomalies", "👥 Customer Behavior Anomalies"])

        with tab1:
            st.subheader("Daily Revenue Anomaly Detection")
            anomaly_days = daily_anomalies[daily_anomalies['Is_Anomaly']]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalous Days Detected", f"{len(anomaly_days)} days")
            with col2:
                normal_avg = daily_anomalies[~daily_anomalies['Is_Anomaly']]['Revenue'].mean()
                st.metric("Avg. Normal Day Revenue", f"DA {normal_avg:,.0f}")
            with col3:
                anomaly_avg = anomaly_days['Revenue'].mean() if len(anomaly_days) > 0 else 0
                st.metric("Avg. Anomaly Day Revenue", f"DA {anomaly_avg:,.0f}", delta=f"DA {anomaly_avg - normal_avg:,.0f}")

            # Visualization
            fig_revenue_anomalies = go.Figure()
            fig_revenue_anomalies.add_trace(go.Scatter(x=daily_anomalies['Date'], y=daily_anomalies['Revenue'], mode='lines', name='Daily Revenue', line=dict(color='lightblue')))
            if not anomaly_days.empty:
                fig_revenue_anomalies.add_trace(go.Scatter(x=anomaly_days['Date'], y=anomaly_days['Revenue'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))
            fig_revenue_anomalies.update_layout(title="Daily Revenue with Anomalies Highlighted", xaxis_title="Date", yaxis_title="Revenue (DA)")
            st.plotly_chart(fig_revenue_anomalies, use_container_width=True)

            if not anomaly_days.empty:
                st.markdown("#### Details of Anomalous Days")
                st.dataframe(anomaly_days[['Date', 'Revenue', 'Order', 'Quantity']].style.format({'Revenue': 'DA {:,.0f}'}))

        with tab2:
            st.subheader("Customer Behavior Anomaly Detection")
            anomalous_customers = customer_anomalies[customer_anomalies['Is_Anomaly']]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Anomalous Customers Detected", f"{len(anomalous_customers)}")
            with col2:
                if not customer_anomalies.empty:
                    revenue_from_anomalies_pct = (anomalous_customers['Total_Revenue'].sum() / customer_anomalies['Total_Revenue'].sum()) * 100
                    st.metric("Revenue % from Anomalous Customers", f"{revenue_from_anomalies_pct:.1f}%")
            
            # --- START: DEFINITIVE FIX FOR THE VALUEERROR ---
            
            # Step 1: Create a copy of the dataframe to safely modify for plotting.
            plot_df = customer_anomalies.copy()

            # Step 2: Use MinMaxScaler to scale the 'AOV' column to a guaranteed positive range.
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(5, 50))
            
            # Ensure 'AOV' column has no NaNs before scaling
            plot_df['AOV'] = plot_df['AOV'].fillna(0)
            
            # Create the new 'Marker_Size' column by scaling the 'AOV'
            plot_df['Marker_Size'] = scaler.fit_transform(plot_df[['AOV']])

            # Step 3: Use the new 'Marker_Size' column for the 'size' property in the plot.
            st.markdown("#### Customer Anomalies (Log Scale)")
            fig_customer_anomalies = px.scatter(
                plot_df,  # Use the safe, modified dataframe
                x='Total_Revenue',
                y='Order_Count',
                color='Is_Anomaly',
                size='Marker_Size',  # <-- USE THE SCALED, POSITIVE COLUMN
                hover_data=['Customer_Name', 'Total_Quantity', 'AOV'], # Keep original AOV in hover data
                color_discrete_map={True: 'red', False: 'blue'},
                log_x=True,
                log_y=True,
                title="Customer Anomalies (Bubble size = Avg. Order Value)"
            )
            
            # --- END: DEFINITIVE FIX ---
            
            st.plotly_chart(fig_customer_anomalies, use_container_width=True)
            
            if not anomalous_customers.empty:
                st.markdown("#### Details of Top 5 Anomalous Customers by Revenue")
                # Ensure all display columns exist before trying to show them
                display_cols = ['Customer_Name', 'Email', 'Phone', 'Total_Revenue', 'Order_Count', 'AOV']
                display_cols_exist = [col for col in display_cols if col in anomalous_customers.columns]
                st.dataframe(
                    anomalous_customers.nlargest(5, 'Total_Revenue')[display_cols_exist].style.format({
                        'Total_Revenue': 'DA {:,.0f}', 
                        'AOV': 'DA {:,.0f}'
                    })
                )

    elif selected == "📅 Smart Comparator":
        st.header("📅 Smart Performance Comparator")
        st.markdown("Compare performance across multiple periods with statistical significance testing and a deep dive into price elasticity.")
        
        comparator = SmartPerformanceComparator(analytics)
        
        # Enhanced period configuration
        with st.expander("⚙️ Configure Comparison Periods", expanded=True):
            # ... (this part remains the same)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                num_periods = st.number_input("Number of periods to compare", min_value=2, max_value=6, value=3, step=1)
                duration = st.number_input("Duration (days)", min_value=7, max_value=365, value=30, step=1)
            
            with col2:
                periods_config = {}
                cols = st.columns(num_periods)
                min_date = analytics.df['Date'].min().date()
                max_date = analytics.df['Date'].max().date()
                for i in range(num_periods):
                    with cols[i]:
                        name = st.text_input(f"Period {i+1} Name", f"Period {chr(65+i)}", key=f"period_name_{i}")
                        start_date = st.date_input(f"Start Date", 
                                                   value=min_date + timedelta(days=i*60), 
                                                   min_value=min_date, max_value=max_date, key=f"period_date_{i}")
                        periods_config[name] = {'start': start_date}
        
        if st.button("🚀 Run Advanced Comparison", type="primary", use_container_width=True):
            with st.spinner("Analyzing periods, performing statistical tests, and calculating elasticity..."):
                results, combined_df, statistical_results, period_dfs = comparator.create_intelligent_comparison(periods_config, duration)
                
                # Store results in session state to persist them across tab clicks
                st.session_state.comparator_results = results
                st.session_state.comparator_stats = statistical_results
                st.session_state.comparator_period_dfs = period_dfs

        # Display results using tabs if they exist in session state
        if 'comparator_results' in st.session_state and st.session_state.comparator_results:
            results = st.session_state.comparator_results
            statistical_results  = st.session_state.comparator_stats
            period_dfs = st.session_state.comparator_period_dfs
            
            # --- CREATE TABBED INTERFACE ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overall KPI Comparison", "💰 Price & Elasticity Deep Dive", "📅 Weekly Deep Dive", "📅 Dailly Deep Dive"])

            with tab1:
                st.subheader("High-Level Performance Overview")
                
                # Create and format comparison dataframe
                comp_df = pd.DataFrame(results).T
                
                # Add percentage change column
                p_first, p_last = comp_df.index[0], comp_df.index[-1]
                for metric in comp_df.columns:
                    if isinstance(comp_df[metric].iloc[0], (int, float)) and comp_df[metric][p_first] != 0:
                        change = ((comp_df[metric][p_last] - comp_df[metric][p_first]) / comp_df[metric][p_first]) * 100
                        comp_df.loc[p_last, f"{metric} % Change"] = f"{change:+.1f}%"
                
                st.dataframe(comp_df.style.format({
                    'Total Revenue': 'DA {:,.0f}',
                    'Avg. Order Value': 'DA {:,.0f}',
                    'Avg. Item Price': 'DA {:,.0f}'
                }, na_rep="").background_gradient(cmap='RdYlGn', subset=['Total Revenue', 'Total Orders'], axis=0))

                st.subheader("📈 Statistical Significance of Revenue Change")
                st.info("This test determines if the change in average daily revenue between periods is statistically significant (p < 0.05) or likely due to random chance.")
                
                if statistical_results :
                    for comparison, res in statistical_results.items():
                        if 'error' not in res:
                            col1, col2 = st.columns(2)
                            p1, p2 = comparison.split('_vs_')
                            with col1:
                                significance = "✅ Significant" if res['significant'] else "❌ Not Significant"
                                st.metric(f"Comparison: {p1} vs {p2}", significance, f"p-value: {res['p_value']:.4f}")
                            with col2:
                                st.metric("Change in Avg. Daily Revenue", f"DA {res['mean_diff']:,.2f}")

            with tab2:
                st.subheader("In-Depth Price & Demand Analysis")
                st.info("Analyze how price changes impact sales volume for your top brands, and identify which brands are most sensitive to price adjustments (elasticity).")
                
                with st.spinner("Calculating price evolution and elasticity..."):
                    evolution_df = comparator.analyze_price_evolution(period_dfs)

                if evolution_df.empty:
                    st.warning("Could not perform price evolution analysis. Ensure there are common top-selling brands across all selected periods.")
                else:
                    st.markdown("#### Brand Price & Volume Changes")
                    
                    # ==============================================================================
                    # START: FIX - Use simple strings for column selection
                    # ==============================================================================
                    display_cols = ['BRAND', 'Price_Change_%', 'Quantity_Change_%', 'Elasticity']
                    st.dataframe(evolution_df[display_cols].style.format({
                        'Price_Change_%': '{:+.2f}%',
                        'Quantity_Change_%': '{:+.2f}%',
                        'Elasticity': '{:.2f}'
                    }).background_gradient(cmap='RdBu_r', subset=['Price_Change_%', 'Quantity_Change_%', 'Elasticity']))
                    # ==============================================================================
                    # END: FIX
                    # ==============================================================================

                    # Generate and display the deep-dive visuals
                    price_figs = comparator.create_price_deep_dive_visuals(evolution_df, list(results.keys()))
                    for fig in price_figs:
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Week-by-Week Performance Breakdown")
                
                weekly_summary = comparator._create_weekly_summary(tuple(period_dfs.items()))
                
                if weekly_summary.empty:
                    st.warning("No weekly data available for the selected periods.")
                else:
                    st.markdown("#### **All Weeks Summary Table**")
                    st.info("This table shows a side-by-side comparison of key metrics for every week. Percentage change is calculated between the first and last periods.")
                    
                    # --- CHANGE 1: Pass periods_config to the function ---
                    weekly_table = comparator.generate_weekly_comparison_table(weekly_summary, periods_config)
                    styled_table = comparator.style_weekly_table(weekly_table)
                    st.dataframe(styled_table, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("#### **Drill Down into a Specific Week**")
                    
                    # --- CHANGE 2: Create a mapping for the selectbox ---
                    # Create a dictionary mapping the pretty label to the simple week number
                    # e.g., {"Week 1 [01/07 - 07/07]": 1, "Week 2 [08/07 - 14/07]": 2}
                    week_labels = weekly_table.index.tolist()
                    week_map = {label: i + 1 for i, label in enumerate(week_labels)}
                    
                    # Use the pretty labels in the selectbox
                    selected_label = st.selectbox("Select a week to analyze in detail:", week_labels)
                    
                    # Get the corresponding simple week number (e.g., 1, 2, 3) for filtering
                    selected_week_num = week_map[selected_label]
                    
                    # Filter using the simple week number
                    week_details = weekly_summary[weekly_summary['Week_Num'] == selected_week_num]
                    
                    st.markdown(f"#### Details for **{selected_label}**") # Display the pretty label
                    
                    # KPI Comparison Charts for the selected week
                    # ... (the rest of the code for charts and top performers remains the same)
                    st.markdown(f"##### KPI Comparison for {selected_label}")
                    fig_kpi_week = make_subplots(rows=1, cols=3, subplot_titles=("Revenue", "Orders", "Avg. Order Value"))
                    fig_kpi_week.add_trace(go.Bar(x=week_details['Period'], y=week_details['Revenue'], name='Revenue'), row=1, col=1)
                    fig_kpi_week.add_trace(go.Bar(x=week_details['Period'], y=week_details['Unique_Orders'], name='Orders'), row=1, col=2)
                    fig_kpi_week.add_trace(go.Bar(x=week_details['Period'], y=week_details['AOV'], name='AOV'), row=1, col=3)
                    fig_kpi_week.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_kpi_week, use_container_width=True)

                    st.markdown(f"##### Top Performers for {selected_label}")
                    period_names = list(results.keys())
                    cols = st.columns(len(period_names))
                    for i, period in enumerate(period_names):
                        with cols[i]:
                            st.markdown(f"**{period}**")
                            period_week_data = week_details[week_details['Period'] == period]
                            if not period_week_data.empty:
                                data = period_week_data.iloc[0]
                                st.markdown("**Top Brands:**")
                                st.caption(data['Top_Brand'])
                                st.markdown("**Top Products:**")
                                st.caption(data['Top_Product'])
                            else:
                                st.caption("No data")

            with tab4:
                st.subheader("☀️ Day-by-Day Performance Breakdown")
                
                # The heavy calculation is cached for speed and now returns two dataframes
                daily_summary, hourly_summary = comparator.generate_daily_deep_dive(tuple(period_dfs.items()))
                
                if daily_summary.empty:
                    st.warning("No daily data available for the selected periods.")
                else:
                    # --- Main Daily Revenue Trend Chart ---
                    st.markdown("#### **Daily Revenue Trends**")
                    fig_daily_trends = px.line(
                        daily_summary, x='Day_Num', y='Revenue', color='Period',
                        title="Daily Revenue Comparison Across Periods"
                    )
                    fig_daily_trends.update_layout(xaxis_title="Day Number (Relative to Period Start)", yaxis_title="Revenue (DA)")
                    st.plotly_chart(fig_daily_trends, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # --- Interactive Day Selector ---
                    st.markdown("#### **Drill Down into a Specific Day**")
                    
                    # Use a selectbox for a cleaner UI
                    max_day = int(daily_summary['Day_Num'].max())
                    day_options = range(1, max_day + 1)
                    selected_day = st.selectbox("Select a day to analyze in detail:", day_options, format_func=lambda d: f"Day {d}")
                    
                    # Filter data for the selected day
                    day_details = daily_summary[daily_summary['Day_Num'] == selected_day]
                    hourly_details = hourly_summary[hourly_summary['Day_Num'] == selected_day]
                    
                    # Display the actual calendar dates for the selected day in each period
                    date_info_cols = st.columns(len(period_dfs))
                    for i, (period_name, period_df) in enumerate(period_dfs.items()):
                        day_data = day_details[day_details['Period'] == period_name]
                        if not day_data.empty:
                            actual_date = day_data['Date'].iloc[0].strftime('%A, %Y-%m-%d')
                            date_info_cols[i].info(f"**{period_name}:** {actual_date}")
                        else:
                            date_info_cols[i].warning(f"**{period_name}:** No sales on this day.")

                    st.markdown("---")

                    # --- 1. Combined KPI Comparison Chart ---
                    st.markdown(f"##### Key Metric Comparison for Day {selected_day}")
                    
                    if not day_details.empty:
                        # Melt the dataframe to plot multiple KPIs in a grouped bar chart
                        kpi_df = day_details.melt(
                            id_vars='Period', 
                            value_vars=['Revenue', 'Unique_Orders', 'AOV'],
                            var_name='Metric',
                            value_name='Value'
                        )
                        
                        fig_kpi_day = px.bar(
                            kpi_df, 
                            x='Metric', 
                            y='Value', 
                            color='Period', 
                            barmode='group',
                            text_auto='.2s' # Format text on bars
                        )
                        fig_kpi_day.update_layout(yaxis_title="Value", xaxis_title="Key Performance Indicator")
                        st.plotly_chart(fig_kpi_day, use_container_width=True)
                    
                    # --- 2. Hourly Revenue Breakdown Chart ---
                    st.markdown(f"##### Hourly Revenue Breakdown for Day {selected_day}")
                    if not hourly_details.empty:
                        fig_hourly = px.line(
                            hourly_details,
                            x='Hour',
                            y='Revenue',
                            color='Period',
                            markers=True,
                            title=f"Hourly Revenue on Day {selected_day}"
                        )
                        fig_hourly.update_layout(xaxis_title="Hour of the Day (0-23)", yaxis_title="Revenue (DA)")
                        st.plotly_chart(fig_hourly, use_container_width=True)
                    else:
                        st.info(f"No hourly data to display for Day {selected_day}.")

                    # --- 3. Side-by-Side Top Performers ---
                    st.markdown(f"##### Top Performers for Day {selected_day}")
                    period_names = list(results.keys())
                    cols = st.columns(len(period_names))
                    
                    for i, period in enumerate(period_names):
                        with cols[i]:
                            st.markdown(f"**{period}**")
                            period_day_data = day_details[day_details['Period'] == period]
                            
                            if not period_day_data.empty:
                                data = period_day_data.iloc[0]
                                st.markdown("**Top 5 Brands:**")
                                st.caption(data['Top_Brands'])
                                
                                st.markdown("**Top 5 Products:**")
                                st.caption(data['Top_Products'])
                            else:
                                st.caption("No data")
                                
                                
    elif selected == "🤖 AI & Forecasting":
        st.header("🤖 AI-Powered Analytics & Forecasting")
        
        # Advanced forecasting
        with st.spinner("Training forecasting models..."):
            forecast_results, forecast_message = create_advanced_forecasting_model(analytics)
        
        if forecast_results:
            st.success("✅ Forecasting models trained successfully!")
            
            # Display forecast results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 30-Day Revenue & Orders Forecast")
                
                # Create forecast visualization
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    subplot_titles=('Revenue Forecast', 'Orders Forecast'),
                    vertical_spacing=0.1
                )
                
                # Historical data
                historical = forecast_results['historical_data']
                
                fig.add_trace(
                    go.Scatter(
                        x=historical['Date'].tail(60), 
                        y=historical['Revenue'].tail(60),
                        mode='lines',
                        name='Historical Revenue',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Forecast
                fig.add_trace(
                    go.Scatter(
                        x=forecast_results['dates'], 
                        y=forecast_results['revenue_forecast'],
                        mode='lines',
                        name='Revenue Forecast',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=1
                )
                
                # Orders
                fig.add_trace(
                    go.Scatter(
                        x=historical['Date'].tail(60), 
                        y=historical['Order'].tail(60),
                        mode='lines',
                        name='Historical Orders',
                        line=dict(color='green'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_results['dates'], 
                        y=forecast_results['orders_forecast'],
                        mode='lines',
                        name='Orders Forecast',
                        line=dict(color='orange', dash='dash'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, title_text="AI-Powered Business Forecasting")
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Revenue (DA)", row=1, col=1)
                fig.update_yaxes(title_text="Orders", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Forecast Summary")
                
                total_forecast_revenue = sum(forecast_results['revenue_forecast'])
                total_forecast_orders = sum(forecast_results['orders_forecast'])
                
                st.metric("30-Day Revenue Forecast", f"DA {total_forecast_revenue:,.0f}")
                st.metric("30-Day Orders Forecast", f"{total_forecast_orders:,.0f}")
                st.metric("Revenue Model Accuracy", f"±DA {forecast_results['revenue_mae']:,.0f}")
                st.metric("Orders Model Accuracy", f"±{forecast_results['orders_mae']:.1f}")
                
                # Feature importance
                st.subheader("🔍 Model Insights")
                
                revenue_importance = forecast_results['feature_importance_revenue']
                top_features = sorted(revenue_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                st.write("**Top Revenue Predictors:**")
                for feature, importance in top_features:
                    st.write(f"• {feature}: {importance:.3f}")
        else:
            st.warning(f"⚠️ {forecast_message}")
            
            # Fallback simple forecasting
            st.subheader("📊 Simple Trend Analysis")
            
            daily_revenue = analytics.df.groupby('Date')['Revenue'].sum().reset_index()
            if len(daily_revenue) >= 7:
                # Simple moving average forecast
                ma_7 = daily_revenue['Revenue'].tail(7).mean()
                ma_30 = daily_revenue['Revenue'].tail(30).mean() if len(daily_revenue) >= 30 else ma_7
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("7-Day Average", f"DA {ma_7:,.0f}")
                with col2:
                    st.metric("30-Day Average", f"DA {ma_30:,.0f}")
                with col3:
                    trend = "📈 Growing" if ma_7 > ma_30 else "📉 Declining"
                    st.metric("Trend", trend)
    
    elif selected == "📋 Data Quality Monitor":
        st.header("📋 Data Quality Monitor")
        
        quality_report = create_data_quality_monitor(analytics)
        
        # Overall quality score
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score = quality_report['overall_quality_score']
            color = "green" if score >= 90 else "orange" if score >= 75 else "red"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background: {color}; color: white;">
                <h2>Overall Quality Score</h2>
                <h1>{score:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Total Records", f"{quality_report['total_records']:,}")
            st.metric("Memory Usage", f"{quality_report['memory_usage_mb']:.1f} MB")
        
        with col3:
            st.metric("Columns", quality_report['total_columns'])
            st.metric("Missing Data Columns", quality_report['missing_data']['columns_with_missing'])
        
        # Detailed quality analysis
        st.subheader("🔍 Detailed Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("❌ Missing Data Analysis")
            missing_details = quality_report['missing_data']['details']
            missing_df = pd.DataFrame(list(missing_details.items()), columns=['Column', 'Missing %'])
            missing_df = missing_df[missing_df['Missing %'] > 0].sort_values('Missing %', ascending=False)
            
            if not missing_df.empty:
                fig = px.bar(missing_df, x='Missing %', y='Column', orientation='h',
                           title="Missing Data by Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing data found!")
        
        with col2:
            st.subheader("🔄 Duplicates Analysis")
            duplicates = quality_report['duplicates']
            
            st.write(f"**Duplicate Rows:** {duplicates['duplicate_rows']:,}")
            st.write(f"**Duplicate Orders:** {duplicates['duplicate_orders']:,}")
            st.write(f"**Duplicate Customers:** {duplicates['duplicate_customers']:,}")
            
            if any(duplicates.values()):
                st.warning("⚠️ Duplicates detected - consider data cleaning")
            else:
                st.success("✅ No duplicates found!")
        
        # Data consistency checks
        st.subheader("✅ Data Consistency Checks")
        
        consistency = quality_report['consistency']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_outliers = consistency.get('price_outliers', 0)
            if price_outliers > 0:
                st.warning(f"⚠️ {price_outliers} price outliers")
            else:
                st.success("✅ Prices look good")
        
        with col2:
            negative_prices = consistency.get('negative_prices', 0)
            if negative_prices > 0:
                st.error(f"❌ {negative_prices} negative prices")
            else:
                st.success("✅ No negative prices")
        
        with col3:
            zero_quantity = consistency.get('zero_quantity', 0)
            if zero_quantity > 0:
                st.warning(f"⚠️ {zero_quantity} zero quantities")
            else:
                st.success("✅ Quantities look good")
        
        with col4:
            future_dates = consistency.get('future_dates', 0)
            if future_dates > 0:
                st.warning(f"⚠️ {future_dates} future dates found")
            else:
                st.success("✅ No future dates")
        
        st.subheader("💡 Recommendations")
        recommendations = quality_report['recommendations']
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("🎉 Your data is in great shape! No immediate recommendations.")

    #======================================================================
    # The following sections were missing from the generated script.
    #======================================================================

    elif selected == "📈 Sales Analytics":
        st.header("📈 In-Depth Sales Analytics")
        viz_engine = AdvancedVisualizationEngine(analytics)
        
        st.subheader("🔄 Advanced Cohort Analysis")
        st.markdown("Track customer retention and lifetime value over time.")
        
        with st.spinner("Generating cohort analysis..."):
            try:
                cohort_fig = viz_engine.create_advanced_cohort_analysis()
                st.plotly_chart(cohort_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate cohort analysis. Not enough monthly data. Error: {e}")

        st.subheader("📊 Sales Distribution Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Revenue by Product Category")
            fig = px.pie(analytics.df, names='Product_Category', values='Revenue', hole=0.4,
                         title="Revenue Share by Product Category")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Revenue by Price Category")
            fig = px.bar(analytics.df.groupby('Price_Category')['Revenue'].sum().reset_index(),
                         x='Price_Category', y='Revenue', title="Total Revenue per Price Category")
            st.plotly_chart(fig, use_container_width=True)

    elif selected == "🏷️ Brand Intelligence":
        st.header("🏷️ Comprehensive Brand Intelligence")
        
        st.subheader("📊 Brand Performance Dashboard")
        brand_stats = analytics.df.groupby('BRAND').agg(
            Total_Revenue=('Revenue', 'sum'),
            Units_Sold=('Quantity', 'sum'),
            Unique_Orders=('Order', 'nunique'),
            Avg_Price=('Product Price', 'mean')
        ).reset_index().sort_values('Total_Revenue', ascending=False).head(15)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue by Brand', 'Market Share (Revenue)', 'Units Sold by Brand', 'Average Price by Brand'),
            specs=[[{"type": "bar"}, {"type": "pie"}], [{"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(go.Bar(x=brand_stats['BRAND'], y=brand_stats['Total_Revenue'], name='Revenue'), row=1, col=1)
        fig.add_trace(go.Pie(labels=brand_stats['BRAND'], values=brand_stats['Total_Revenue'], name='Market Share'), row=1, col=2)
        fig.add_trace(go.Bar(x=brand_stats['BRAND'], y=brand_stats['Units_Sold'], name='Units Sold'), row=2, col=1)
        fig.add_trace(go.Bar(x=brand_stats['BRAND'], y=brand_stats['Avg_Price'], name='Average Price'), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, title_text="Brand Performance Deep Dive")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Brand Comparison Table")
        st.dataframe(brand_stats.style.format({
            'Total_Revenue': 'DA {:,.0f}',
            'Avg_Price': 'DA {:,.0f}'
        }), use_container_width=True)

    elif selected == "🗺️ Geographic Insights":
        st.header("🗺️ Geographic Performance Insights")

        st.subheader("📍 Geographic Performance Dashboard")
        geo_stats = analytics.df.groupby('Wilaya').agg(
            Total_Revenue=('Revenue', 'sum'),
            Total_Orders=('Order', 'nunique'),
            Unique_Customers=('Customer_Name', 'nunique')
        ).reset_index().sort_values('Total_Revenue', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top 15 Wilayas by Revenue")
            fig = px.bar(geo_stats.head(15), x='Wilaya', y='Total_Revenue', color='Total_Revenue',
                         color_continuous_scale='viridis', title="Top 15 Wilayas by Revenue")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Customer Distribution by Wilaya")
            fig = px.pie(geo_stats.head(15), names='Wilaya', values='Unique_Customers', hole=0.4,
                         title="Customer Share in Top 15 Wilayas")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏙️ Top 20 Cities by Revenue")
        city_stats = analytics.df.groupby('City')['Revenue'].sum().nlargest(20).reset_index()
        fig_city = px.bar(city_stats, x='Revenue', y='City', orientation='h', title="Top 20 Cities by Revenue")
        st.plotly_chart(fig_city, use_container_width=True)

    # ==============================================================================
    # START: Replace the entire "Product Analytics" block with this robust version
    # ==============================================================================
    elif selected == "📦 Product Analytics":
        st.header("📦 Product Performance Analytics")

        # ... (The top/bottom product charts section remains the same) ...
        st.subheader("📊 Top & Bottom Performing Products")
        product_stats = analytics.df.groupby('Item Name').agg(
            Total_Revenue=('Revenue', 'sum'),
            Units_Sold=('Quantity', 'sum'),
            Avg_Price=('Product Price', 'mean')
        ).reset_index().sort_values('Total_Revenue', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top 15 Products by Revenue")
            # ... (chart code) ...
            top_products = product_stats.head(15)
            fig = px.bar(top_products, y='Item Name', x='Total_Revenue', orientation='h', title="Top 15 Products")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Bottom 15 Products by Revenue")
            # ... (chart code) ...
            bottom_products = product_stats.tail(15)
            fig = px.bar(bottom_products, y='Item Name', x='Total_Revenue', orientation='h', title="Bottom 15 Products")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)


        st.markdown("---")
        st.subheader("📈 ABC Analysis (Revenue-based)")
        st.info("This analysis categorizes your products based on the Pareto Principle (80/20 rule). 'A' products are your vital few superstars, 'B' are your consistent performers, and 'C' are your trivial many.")

        if not product_stats.empty:
            product_stats_sorted = product_stats.sort_values('Total_Revenue', ascending=False)
            product_stats_sorted['Cumulative_Revenue'] = product_stats_sorted['Total_Revenue'].cumsum()
            total_revenue = product_stats_sorted['Total_Revenue'].sum()
            product_stats_sorted['Cumulative_Percentage'] = 100 * product_stats_sorted['Cumulative_Revenue'] / total_revenue
            
            def abc_segmentation(percentage):
                if percentage <= 80: return 'A'
                elif percentage <= 95: return 'B'
                else: return 'C'
            
            product_stats_sorted['ABC_Category'] = product_stats_sorted['Cumulative_Percentage'].apply(abc_segmentation).astype('category')
            
            abc_summary = product_stats_sorted.groupby('ABC_Category').agg(
                Product_Count=('Item Name', 'count'),
                Total_Revenue=('Total_Revenue', 'sum')
            ).reset_index()
            
            fig_abc = px.bar(abc_summary, x='ABC_Category', y='Total_Revenue', color='ABC_Category',
                             text='Product_Count', title="Revenue & Product Count by ABC Category",
                             labels={'ABC_Category': 'Category'},
                             category_orders={"ABC_Category": ["A", "B", "C"]},
                             color_discrete_map={'A': '#00529B', 'B': '#6495ED', 'C': '#FF6347'})
            fig_abc.update_traces(textposition='inside', textfont_size=14, textfont_color='white')
            st.plotly_chart(fig_abc, use_container_width=True)

            st.markdown("---")
            st.subheader("📥 Export Product Lists by ABC Category")
            
            selected_category = st.selectbox("Select a category to download:", ['A', 'B', 'C'])

            if selected_category:
                export_df = product_stats_sorted[product_stats_sorted['ABC_Category'] == selected_category]
                
                export_cols = ['Item Name', 'Total_Revenue', 'Units_Sold', 'Avg_Price', 'ABC_Category']
                final_export_df = export_df[export_cols].rename(columns={
                    'Total_Revenue': 'Total Revenue (DA)',
                    'Units_Sold': 'Total Units Sold',
                    'Avg_Price': 'Average Price (DA)',
                    'ABC_Category': 'Category'
                })

                st.markdown(f"**Preview for Category '{selected_category}' ({len(final_export_df)} customers):**")

                # --- START: DEFINITIVE FIX FOR PYARROW ERROR ---
                
                # 1. Create a small, separate DataFrame just for display.
                display_df = final_export_df.head(10).copy()

                # 2. Convert any 'category' type columns in this small DataFrame to 'str' (object).
                # This makes it "safe" for Streamlit's styled rendering.
                for col in display_df.select_dtypes(include=['category']).columns:
                    display_df[col] = display_df[col].astype(str)

                # 3. Define the formatter for the numeric columns.
                formatter = {
                    'Total Revenue (DA)': 'DA {:,.0f}',
                    'Average Price (DA)': 'DA {:,.2f}',
                    'Total Units Sold': '{:,.0f}'
                }

                # 4. Apply the style to the "safe" display_df.
                st.dataframe(
                    display_df.style.format(formatter, subset=list(formatter.keys())),
                    use_container_width=True
                )
                
                # --- END: DEFINITIVE FIX ---

                csv_data = final_export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download List for Category '{selected_category}'",
                    data=csv_data,
                    file_name=f"abc_category_{selected_category}_products_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
        else:
            st.warning("Not enough product data to perform ABC Analysis.")
    # ==============================================================================
    # END: The replacement block ends here
    # ==============================================================================

    elif selected == "⏰ Temporal Analysis":
        st.header("⏰ Temporal Performance Analysis")

        st.subheader("📅 Performance by Day and Hour")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Revenue by Day of Week")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_revenue = analytics.df.groupby('Day_of_Week')['Revenue'].sum().reindex(day_order).reset_index()
            fig = px.bar(day_revenue, x='Day_of_Week', y='Revenue', title="Total Revenue by Day of Week")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Orders by Hour of Day")
            hour_orders = analytics.df.groupby('Hour')['Order'].nunique().reset_index()
            fig = px.line(hour_orders, x='Hour', y='Order', markers=True, title="Total Orders by Hour of Day")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 Sales Heatmap (Hour vs Day of Week)")
        heatmap_data = analytics.df.groupby(['Day_of_Week', 'Hour'])['Revenue'].sum().unstack().fillna(0).reindex(day_order)
        fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Hour of Day", y="Day of Week", color="Revenue"),
                                title="Revenue Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    elif selected == "🔍 Advanced Insights":
        st.header("🔍 Advanced Insights & Strategic Recommendations")
        
        st.subheader("💡 Key Business Insights")
        # Generate and display insights based on analysis
        top_brand_pct = (analytics.df.groupby('BRAND')['Revenue'].sum().max() / analytics.df['Revenue'].sum()) * 100
        top_wilaya_pct = (analytics.df.groupby('Wilaya')['Revenue'].sum().max() / analytics.df['Revenue'].sum()) * 100
        repeat_customers = analytics.df.groupby('Customer_Name')['Order'].nunique()
        repeat_rate = (repeat_customers > 1).mean() * 100

        insights = [
            f"**Brand Concentration**: The top brand accounts for **{top_brand_pct:.1f}%** of total revenue. Consider diversifying or promoting other brands.",
            f"**Geographic Focus**: The top Wilaya contributes **{top_wilaya_pct:.1f}%** of revenue, indicating a strong regional market but also a potential for expansion into other areas.",
            f"**Customer Loyalty**: The customer repeat rate is **{repeat_rate:.1f}%**. Focus on loyalty programs to increase this metric for sustainable growth.",
            f"**Peak Performance**: The most profitable day of the week is **{analytics.df.groupby('Day_of_Week')['Revenue'].sum().idxmax()}**. Optimize marketing campaigns around this day."
        ]
        
        for insight in insights:
            st.info(insight)
        
        st.subheader("📈 Statistical Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Price vs. Quantity Correlation**")
            corr = analytics.df['Product Price'].corr(analytics.df['Quantity'])
            st.metric("Correlation Coefficient", f"{corr:.3f}")
            st.caption("A negative value suggests that as price increases, quantity sold tends to decrease.")

        with col2:
            st.markdown("**Revenue Distribution Skewness**")
            skew = stats.skew(analytics.df['Revenue'])
            st.metric("Skewness", f"{skew:.3f}")
            st.caption("A high positive value (>1) indicates that a few high-value orders contribute disproportionately to total revenue.")
    elif selected == "🐍 Code Sandbox":
        st.header("🐍 Python Code Sandbox")
        
        st.warning(
            """
            **⚠️ Security Warning:** This is a powerful feature for advanced users. The code you enter is executed directly on the server. 
            Only use this feature in a trusted environment. Do not expose this app publicly with the sandbox enabled.
            """
        )

        st.info(
            """
            Here, you can directly manipulate the analytics DataFrame using Python and Pandas.
            - Your DataFrame is available as a variable named `df`.
            - After applying your code, the changes will be reflected in all other tabs.
            - Click "Reset to Original Data" to undo all changes.
            """
        )
        
        # --- Initialize the sandbox DataFrame in session_state ---
        # This makes sure we have a working copy that persists across reruns.
        if 'sandbox_df' not in st.session_state:
            st.session_state.sandbox_df = analytics.df.copy()

        # --- Code Editor UI ---
        st.markdown("#### Enter your Pandas code here:")
        
        default_code = (
            "# Example: Filter for a specific brand and status\n"
            "# df = df[df['BRAND'] == 'ADIDAS'].copy()\n"
            "# df = df[df['Status'] == 'Livré'].copy()\n\n"
            "# Example: Drop a column\n"
            "# df.drop(columns=['UGS'], inplace=True)\n\n"
            "# Your DataFrame is named 'df'. Modify it in place or reassign it."
        )

        # Use a text area for code input
        code = st.text_area("Code Editor", value=default_code, height=250, key="code_editor")

        # --- Action Buttons ---
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("🚀 Apply Code", type="primary", use_container_width=True):
                try:
                    # Create a dictionary for the local scope of the exec function
                    local_scope = {'df': st.session_state.sandbox_df.copy()}
                    
                    # Execute the user's code. The result of the modifications to 'df' will be in local_scope
                    exec(code, {}, local_scope)
                    
                    # Update the dataframe in session_state with the modified version
                    st.session_state.sandbox_df = local_scope['df']
                    st.success("✅ Code applied successfully! All tabs are now using the modified data.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ An error occurred while executing your code:")
                    st.exception(e)

        with col2:
            if st.button("🔄 Reset to Original", use_container_width=True):
                # Remove the sandbox dataframe from session state to trigger a reset
                if 'sandbox_df' in st.session_state:
                    del st.session_state.sandbox_df
                st.info("🔄 Data has been reset to its original state.")
                st.rerun()

        # --- Data Preview ---
        st.markdown("---")
        st.markdown(f"#### Preview of the Current DataFrame ({len(st.session_state.sandbox_df)} rows)")
        # 1. Create a small, separate DataFrame just for the preview.
        preview_df = st.session_state.sandbox_df.head(20).copy()

        # 2. Convert any 'category' type columns in this small preview DataFrame to 'str' (object).
        # This makes it "safe" for Streamlit's rendering engine.
        for col in preview_df.select_dtypes(include=['category']).columns:
            preview_df[col] = preview_df[col].astype(str)

        # 3. Display the "safe" preview_df.
        st.dataframe(preview_df)

#======================================================================
# Final block to run the application
#======================================================================
if __name__ == "__main__":
    main()

"""
Rossmann Store Sales Forecasting Dashboard
Interactive Streamlit application for exploring sales forecasts
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Rossmann Store Sales Forecasting Dashboard")
st.markdown("""
This interactive dashboard allows you to explore sales forecasts for Rossmann stores using classical time series models.
Select a store to view historical sales and 6-week ahead forecasts.
""")

# Load data function with caching
@st.cache_data
def load_data():
    """Load and prepare the Rossmann datasets"""
    train_df = pd.read_csv('train.csv', low_memory=False)
    store_df = pd.read_csv('store.csv')
    
    # Merge datasets
    train_merged = train_df.merge(store_df, on='Store', how='left')
    train_merged['Date'] = pd.to_datetime(train_merged['Date'])
    
    return train_merged

# Forecast function with caching
@st.cache_data
def forecast_store(store_id, _train_merged, model_type='SARIMA', forecast_horizon=42):
    """
    Generate forecast for a specific store
    """
    # Filter data
    store_data = _train_merged[(_train_merged['Store'] == store_id) & 
                               (_train_merged['Open'] == 1)].copy()
    store_data = store_data.sort_values('Date')
    store_data.set_index('Date', inplace=True)
    
    # Create complete date range
    date_range = pd.date_range(start=store_data.index.min(), 
                              end=store_data.index.max(), 
                              freq='D')
    store_data = store_data.reindex(date_range)
    store_data['Sales'] = store_data['Sales'].interpolate(method='linear')
    
    # Train-test split
    train_size = len(store_data) - forecast_horizon
    train_series = store_data['Sales'][:train_size]
    test_series = store_data['Sales'][train_size:]
    
    try:
        if model_type == 'SARIMA':
            # Fit SARIMA model
            model = SARIMAX(train_series, 
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 7),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
        else:  # ARIMA
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train_series, order=(1, 1, 1))
        
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=forecast_horizon)
        
        # Calculate metrics
        mae = mean_absolute_error(test_series.values, forecast.values)
        rmse = np.sqrt(mean_squared_error(test_series.values, forecast.values))
        mape = np.mean(np.abs((test_series.values - forecast.values) / test_series.values)) * 100
        
        return {
            'success': True,
            'train_series': train_series,
            'test_series': test_series,
            'forecast': forecast,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'store_data': store_data
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Main app
def main():
    # Load data
    with st.spinner('Loading data...'):
        train_merged = load_data()
    
    st.success('✓ Data loaded successfully!')
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    
    # Store selection
    available_stores = sorted(train_merged['Store'].unique())
    selected_store = st.sidebar.selectbox(
        "Select Store:",
        available_stores,
        index=0
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model:",
        ['SARIMA', 'ARIMA']
    )
    
    # Forecast horizon
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (days):",
        min_value=7,
        max_value=60,
        value=42,
        step=7
    )
    
    # Generate forecast button
    if st.sidebar.button('🔮 Generate Forecast', type='primary'):
        with st.spinner(f'Generating {model_type} forecast for Store {selected_store}...'):
            result = forecast_store(selected_store, train_merged, model_type, forecast_horizon)
        
        if result['success']:
            # Display metrics
            st.header(f"📈 Store {selected_store} - {model_type} Forecast Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{result['mae']:.2f}")
            with col2:
                st.metric("RMSE", f"{result['rmse']:.2f}")
            with col3:
                st.metric("MAPE", f"{result['mape']:.2f}%")
            
            # Plot forecast
            st.subheader("Forecast Visualization")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot last 90 days of training data
            train_last_90 = result['train_series'].tail(90)
            ax.plot(train_last_90.index, train_last_90.values, 
                   label='Historical Sales (Last 90 days)', color='blue', linewidth=2)
            
            # Plot actual test data
            ax.plot(result['test_series'].index, result['test_series'].values,
                   label='Actual Sales (Test Period)', color='green', linewidth=2)
            
            # Plot forecast
            ax.plot(result['forecast'].index, result['forecast'].values,
                   label=f'{model_type} Forecast', color='red', linewidth=2, linestyle='--')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Sales', fontsize=12)
            ax.set_title(f'Store {selected_store} - {model_type} Sales Forecast', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Show forecast data table
            st.subheader("Forecast Data")
            forecast_df = pd.DataFrame({
                'Date': result['forecast'].index,
                'Forecasted Sales': result['forecast'].values.round(2),
                'Actual Sales': result['test_series'].values.round(2),
                'Error': (result['forecast'].values - result['test_series'].values).round(2)
            })
            st.dataframe(forecast_df, use_container_width=True)
            
            # Store statistics
            st.subheader("Store Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Historical Sales Statistics:**")
                st.write(f"- Mean: ${result['store_data']['Sales'].mean():.2f}")
                st.write(f"- Median: ${result['store_data']['Sales'].median():.2f}")
                st.write(f"- Std Dev: ${result['store_data']['Sales'].std():.2f}")
                st.write(f"- Min: ${result['store_data']['Sales'].min():.2f}")
                st.write(f"- Max: ${result['store_data']['Sales'].max():.2f}")
            
            with col2:
                st.write("**Forecast Period:**")
                st.write(f"- Start Date: {result['test_series'].index.min().date()}")
                st.write(f"- End Date: {result['test_series'].index.max().date()}")
                st.write(f"- Forecast Days: {len(result['forecast'])}")
                st.write(f"- Avg Forecast: ${result['forecast'].mean():.2f}")
                st.write(f"- Total Forecast: ${result['forecast'].sum():.2f}")
            
        else:
            st.error(f"❌ Forecast failed: {result['error']}")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📖 About
    This dashboard uses classical time series models:
    - **SARIMA**: Captures seasonality (weekly patterns)
    - **ARIMA**: Baseline model without seasonal terms
    
    ### 📊 Metrics
    - **MAE**: Mean Absolute Error
    - **RMSE**: Root Mean Squared Error  
    - **MAPE**: Mean Absolute Percentage Error
    
    ### 🎯 Use Cases
    - Inventory planning
    - Staff scheduling
    - Promotional planning
    - Supply chain optimization
    """)

if __name__ == '__main__':
    main()

"""
Interactive Dashboard for AutoML Lite.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import time
from pathlib import Path
import json
import warnings

STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    warnings.warn("Streamlit not available. Install with: pip install streamlit")

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AutoMLDashboard:
    """
    Interactive dashboard for AutoML Lite using Streamlit.
    """
    
    def __init__(self, title: str = "AutoML Lite Dashboard"):
        """
        Initialize AutoML Dashboard.
        
        Args:
            title: Dashboard title
        """
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit required for interactive dashboard")
        
        self.title = title
        self.data = {}
        self.metrics_history = []
        self.is_running = False
        
    def run_dashboard(self):
        """Run the interactive dashboard."""
        st.set_page_config(
            page_title=self.title,
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        st.title(f"🤖 {self.title}")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🏆 Model Performance", 
            "🎯 Feature Analysis",
            "⏱️ Training Progress",
            "🔍 Model Interpretability"
        ])
        
        with tab1:
            self._overview_tab()
        
        with tab2:
            self._model_performance_tab()
        
        with tab3:
            self._feature_analysis_tab()
        
        with tab4:
            self._training_progress_tab()
        
        with tab5:
            self._interpretability_tab()
    
    def _create_sidebar(self):
        """Create sidebar controls."""
        st.sidebar.header("🎛️ Controls")
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
        if auto_refresh:
            time.sleep(1)
            st.rerun()
        
        # Data upload
        st.sidebar.subheader("📁 Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your dataset for analysis"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(data)} rows, {len(data.columns)} columns")
                self.data['uploaded_data'] = data
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {str(e)}")
        
        # Configuration
        st.sidebar.subheader("⚙️ Configuration")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            time_budget = st.number_input("Time Budget (s)", min_value=60, value=300, step=60)
        with col2:
            max_models = st.number_input("Max Models", min_value=1, value=10, step=1)
        
        problem_type = st.sidebar.selectbox(
            "Problem Type",
            ["classification", "regression", "time_series"],
            index=0
        )
        
        enable_ensemble = st.sidebar.checkbox("Enable Ensemble", value=True)
        enable_interpretability = st.sidebar.checkbox("Enable Interpretability", value=True)
        
        # Start/Stop training
        st.sidebar.subheader("🚀 Training Control")
        
        if st.sidebar.button("Start Training", type="primary"):
            self.is_running = True
            st.sidebar.success("Training started!")
        
        if st.sidebar.button("Stop Training"):
            self.is_running = False
            st.sidebar.warning("Training stopped!")
        
        # Status
        st.sidebar.subheader("📈 Status")
        if self.is_running:
            st.sidebar.success("🟢 Training in progress...")
        else:
            st.sidebar.info("⚪ Ready to start")
    
    def _overview_tab(self):
        """Overview tab content."""
        st.header("📊 Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Best Score",
                value="0.85",
                delta="+0.02"
            )
        
        with col2:
            st.metric(
                label="Models Tried",
                value="8",
                delta="+2"
            )
        
        with col3:
            st.metric(
                label="Training Time",
                value="2m 30s",
                delta="-30s"
            )
        
        with col4:
            st.metric(
                label="Features",
                value="15",
                delta="+3"
            )
        
        # Progress bar
        st.subheader("Training Progress")
        progress = st.progress(0.75)
        st.caption("75% complete - 2 models remaining")
        
        # Recent activity
        st.subheader("Recent Activity")
        
        activity_data = {
            "Time": ["2:30 PM", "2:25 PM", "2:20 PM", "2:15 PM"],
            "Event": [
                "Random Forest completed (Score: 0.87)",
                "XGBoost started",
                "Logistic Regression completed (Score: 0.82)",
                "Training started"
            ],
            "Status": ["✅", "🔄", "✅", "✅"]
        }
        
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, hide_index=True)
    
    def _model_performance_tab(self):
        """Model performance tab content."""
        st.header("🏆 Model Performance")
        
        # Sample leaderboard data
        leaderboard_data = {
            "Model": ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "Decision Tree"],
            "Score": [0.87, 0.85, 0.82, 0.80, 0.78],
            "Training Time (s)": [45, 120, 15, 60, 10],
            "Parameters": [150, 200, 8, 25, 12]
        }
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Leaderboard chart
        fig = px.bar(
            leaderboard_df,
            x="Model",
            y="Score",
            color="Score",
            title="Model Performance Leaderboard",
            color_continuous_scale="viridis"
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed leaderboard table
        st.subheader("Detailed Leaderboard")
        st.dataframe(leaderboard_df, use_container_width=True)
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Training time comparison
            fig_time = px.scatter(
                leaderboard_df,
                x="Training Time (s)",
                y="Score",
                size="Parameters",
                hover_name="Model",
                title="Training Time vs Score"
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Score distribution
            fig_dist = px.histogram(
                leaderboard_df,
                x="Score",
                nbins=10,
                title="Score Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    def _feature_analysis_tab(self):
        """Feature analysis tab content."""
        st.header("🎯 Feature Analysis")
        
        # Sample feature importance data
        feature_data = {
            "Feature": [
                "age", "income", "education", "credit_score", "employment_length",
                "loan_amount", "debt_ratio", "payment_history", "collateral", "purpose"
            ],
            "Importance": [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.01, 0.01]
        }
        
        feature_df = pd.DataFrame(feature_data)
        
        # Feature importance chart
        fig = px.bar(
            feature_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            color="Importance",
            color_continuous_scale="plasma"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation
        st.subheader("Feature Correlation Matrix")
        
        # Sample correlation matrix
        np.random.seed(42)
        n_features = 8
        corr_matrix = np.random.rand(n_features, n_features)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1)  # Diagonal = 1
        
        feature_names = ["age", "income", "education", "credit_score", 
                        "employment", "loan_amount", "debt_ratio", "history"]
        
        corr_df = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)
        
        fig_corr = px.imshow(
            corr_df,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature statistics
        st.subheader("Feature Statistics")
        
        if 'uploaded_data' in self.data:
            st.dataframe(self.data['uploaded_data'].describe(), use_container_width=True)
        else:
            st.info("Upload data to see feature statistics")
    
    def _training_progress_tab(self):
        """Training progress tab content."""
        st.header("⏱️ Training Progress")
        
        # Real-time training metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Metrics")
            
            # Sample training history
            epochs = list(range(1, 21))
            train_loss = [0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38,
                         0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27]
            val_loss = [0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.47, 0.45, 0.43,
                       0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32]
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, name="Training Loss"))
            fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, name="Validation Loss"))
            fig_loss.update_layout(title="Training vs Validation Loss", height=300)
            
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            st.subheader("Model Convergence")
            
            # Learning curves
            train_score = [0.6, 0.65, 0.7, 0.75, 0.78, 0.8, 0.82, 0.84, 0.85, 0.86,
                          0.87, 0.87, 0.88, 0.88, 0.89, 0.89, 0.89, 0.9, 0.9, 0.9]
            val_score = [0.55, 0.6, 0.65, 0.7, 0.73, 0.75, 0.77, 0.79, 0.8, 0.81,
                        0.82, 0.82, 0.83, 0.83, 0.84, 0.84, 0.84, 0.85, 0.85, 0.85]
            
            fig_score = go.Figure()
            fig_score.add_trace(go.Scatter(x=epochs, y=train_score, name="Training Score"))
            fig_score.add_trace(go.Scatter(x=epochs, y=val_score, name="Validation Score"))
            fig_score.update_layout(title="Training vs Validation Score", height=300)
            
            st.plotly_chart(fig_score, use_container_width=True)
        
        # Current model training
        st.subheader("Current Model Training")
        
        if self.is_running:
            # Simulate training progress
            current_model = st.selectbox(
                "Current Model",
                ["Random Forest", "XGBoost", "Neural Network", "SVM"],
                index=1
            )
            
            # Progress indicators
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Epoch", "15/50")
                st.progress(0.3)
            
            with col2:
                st.metric("Current Loss", "0.32")
                st.progress(0.68)
            
            with col3:
                st.metric("Current Score", "0.85")
                st.progress(0.85)
            
            # Training log
            st.subheader("Training Log")
            
            log_entries = [
                "Epoch 15/50 - Loss: 0.32 - Score: 0.85",
                "Epoch 14/50 - Loss: 0.33 - Score: 0.84",
                "Epoch 13/50 - Loss: 0.34 - Score: 0.83",
                "Epoch 12/50 - Loss: 0.35 - Score: 0.82",
                "Epoch 11/50 - Loss: 0.36 - Score: 0.81"
            ]
            
            for entry in log_entries:
                st.text(entry)
        else:
            st.info("No model currently training. Start training from the sidebar.")
    
    def _interpretability_tab(self):
        """Model interpretability tab content."""
        st.header("🔍 Model Interpretability")
        
        # SHAP values
        st.subheader("SHAP Values")
        
        # Sample SHAP data
        shap_data = {
            "Feature": ["age", "income", "education", "credit_score", "employment_length"],
            "SHAP Value": [0.15, 0.12, 0.08, 0.06, 0.04],
            "Impact": ["Positive", "Positive", "Negative", "Positive", "Negative"]
        }
        
        shap_df = pd.DataFrame(shap_data)
        
        fig_shap = px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="Impact",
            title="SHAP Feature Importance",
            color_discrete_map={"Positive": "green", "Negative": "red"}
        )
        
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Partial dependence plots
        st.subheader("Partial Dependence Plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age partial dependence
            age_values = list(range(18, 80, 2))
            age_effect = [0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.23, 0.2, 0.18,
                         0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02, 0.0, -0.02, -0.05,
                         0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25]
            
            fig_age = px.line(
                x=age_values,
                y=age_effect,
                title="Age Partial Dependence",
                labels={"x": "Age", "y": "Effect on Prediction"}
            )
            
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Income partial dependence
            income_values = list(range(20000, 120000, 5000))
            income_effect = [0.05, 0.08, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3,
                            0.32, 0.35, 0.38, 0.4, 0.42, 0.45, 0.48, 0.5, 0.52, 0.55]
            
            fig_income = px.line(
                x=income_values,
                y=income_effect,
                title="Income Partial Dependence",
                labels={"x": "Income", "y": "Effect on Prediction"}
            )
            
            st.plotly_chart(fig_income, use_container_width=True)
        
        # Model explanation
        st.subheader("Model Explanation")
        
        explanation_text = """
        **Model Behavior Analysis:**
        
        - **Age**: The model shows a positive relationship with age up to 45, then a slight decline
        - **Income**: Higher income consistently leads to better predictions
        - **Education**: More education generally improves outcomes
        - **Credit Score**: Strong positive correlation with predictions
        - **Employment Length**: Longer employment history is beneficial
        
        **Key Insights:**
        - The model heavily relies on financial stability indicators
        - Age has a non-linear relationship with the target
        - Income and credit score are the most important features
        """
        
        st.markdown(explanation_text)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update dashboard metrics."""
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
    
    def save_dashboard_state(self, filepath: str):
        """Save dashboard state to file."""
        state = {
            'title': self.title,
            'data': self.data,
            'metrics_history': self.metrics_history,
            'is_running': self.is_running
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_dashboard_state(self, filepath: str):
        """Load dashboard state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.title = state.get('title', self.title)
        self.data = state.get('data', {})
        self.metrics_history = state.get('metrics_history', [])
        self.is_running = state.get('is_running', False)


def run_dashboard_app():
    """Run the dashboard application."""
    dashboard = AutoMLDashboard("AutoML Lite Dashboard")
    dashboard.run_dashboard()


if __name__ == "__main__":
    run_dashboard_app() 
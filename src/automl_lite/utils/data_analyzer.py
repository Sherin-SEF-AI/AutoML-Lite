"""
Advanced data analysis utilities for AutoML Lite.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from .logger import get_logger

logger = get_logger(__name__)


class DataAnalyzer:
    """
    Advanced data analysis and exploration utilities.
    
    Provides comprehensive data analysis including:
    - Statistical summaries
    - Correlation analysis
    - Distribution analysis
    - Outlier detection
    - Feature relationships
    - Data quality assessment
    - Dimensionality reduction visualization
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize DataAnalyzer.
        
        Args:
            verbose: Whether to show progress messages
        """
        self.verbose = verbose
        self.analysis_results = {}
        
    def analyze_dataset(self, data: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis.
        
        Args:
            data: Input DataFrame
            target: Target column name (optional)
            
        Returns:
            Dictionary containing all analysis results
        """
        if self.verbose:
            logger.info("Starting comprehensive dataset analysis...")
        
        results = {
            'basic_info': self._analyze_basic_info(data),
            'data_quality': self._analyze_data_quality(data),
            'statistical_summary': self._analyze_statistical_summary(data),
            'correlation_analysis': self._analyze_correlations(data),
            'distribution_analysis': self._analyze_distributions(data),
            'outlier_analysis': self._detect_outliers(data),
            'feature_relationships': self._analyze_feature_relationships(data),
        }
        
        if target and target in data.columns:
            results['target_analysis'] = self._analyze_target(data, target)
            results['feature_importance_preliminary'] = self._analyze_feature_importance_preliminary(data, target)
        
        self.analysis_results = results
        
        if self.verbose:
            logger.info("Dataset analysis completed successfully")
        
        return results
    
    def _analyze_basic_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic dataset information."""
        return {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': list(data.columns),
            'data_types': data.dtypes.value_counts().to_dict(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(data.select_dtypes(include=['datetime']).columns),
        }
    
    def _analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality issues."""
        missing_info = data.isnull().sum()
        duplicate_rows = data.duplicated().sum()
        
        return {
            'missing_values': {
                'total_missing': missing_info.sum(),
                'missing_percentage': (missing_info.sum() / (len(data) * len(data.columns))) * 100,
                'columns_with_missing': (missing_info > 0).sum(),
                'missing_by_column': missing_info[missing_info > 0].to_dict()
            },
            'duplicates': {
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': (duplicate_rows / len(data)) * 100
            },
            'data_consistency': self._check_data_consistency(data),
            'data_completeness': self._check_data_completeness(data)
        }
    
    def _analyze_statistical_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'message': 'No numeric columns found'}
        
        summary = numeric_data.describe().to_dict()
        
        # Add additional statistics
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                summary[col].update({
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data)),
                    'iqr': float(stats.iqr(col_data)),
                    'cv': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0
                })
        
        return summary
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between features."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or numeric_data.shape[1] < 2:
            return {'message': 'Insufficient numeric columns for correlation analysis'}
        
        # Pearson correlations
        pearson_corr = numeric_data.corr()
        
        # Spearman correlations
        spearman_corr = numeric_data.corr(method='spearman')
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                corr_val = pearson_corr.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        'feature1': pearson_corr.columns[i],
                        'feature2': pearson_corr.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'pearson_correlation': pearson_corr.to_dict(),
            'spearman_correlation': spearman_corr.to_dict(),
            'highly_correlated_pairs': high_corr_pairs,
            'correlation_matrix_shape': pearson_corr.shape
        }
    
    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature distributions."""
        numeric_data = data.select_dtypes(include=[np.number])
        categorical_data = data.select_dtypes(include=['object', 'category'])
        
        distributions = {
            'numeric_distributions': {},
            'categorical_distributions': {}
        }
        
        # Analyze numeric distributions
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                distributions['numeric_distributions'][col] = {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data)),
                    'is_normal': self._test_normality(col_data),
                    'percentiles': col_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
                }
        
        # Analyze categorical distributions
        for col in categorical_data.columns:
            col_data = categorical_data[col]
            value_counts = col_data.value_counts()
            distributions['categorical_distributions'][col] = {
                'unique_values': len(value_counts),
                'most_common': value_counts.head(5).to_dict(),
                'least_common': value_counts.tail(5).to_dict(),
                'entropy': self._calculate_entropy(value_counts),
                'cardinality': len(value_counts)
            }
        
        return distributions
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'message': 'No numeric columns for outlier detection'}
        
        outlier_results = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                outlier_results[col] = {
                    'iqr_method': self._detect_outliers_iqr(col_data),
                    'zscore_method': self._detect_outliers_zscore(col_data),
                    'isolation_forest': self._detect_outliers_isolation_forest(col_data),
                    'local_outlier_factor': self._detect_outliers_lof(col_data)
                }
        
        return outlier_results
    
    def _analyze_feature_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between features."""
        numeric_data = data.select_dtypes(include=[np.number])
        categorical_data = data.select_dtypes(include=['object', 'category'])
        
        relationships = {
            'numeric_relationships': {},
            'categorical_relationships': {},
            'mixed_relationships': {}
        }
        
        # Analyze numeric feature relationships
        if numeric_data.shape[1] >= 2:
            for i, col1 in enumerate(numeric_data.columns):
                for col2 in numeric_data.columns[i+1:]:
                    key = f"{col1}_vs_{col2}"
                    relationships['numeric_relationships'][key] = self._analyze_numeric_relationship(
                        numeric_data[col1], numeric_data[col2]
                    )
        
        # Analyze categorical feature relationships
        if categorical_data.shape[1] >= 2:
            for i, col1 in enumerate(categorical_data.columns):
                for col2 in categorical_data.columns[i+1:]:
                    key = f"{col1}_vs_{col2}"
                    relationships['categorical_relationships'][key] = self._analyze_categorical_relationship(
                        categorical_data[col1], categorical_data[col2]
                    )
        
        # Analyze mixed relationships
        if not numeric_data.empty and not categorical_data.empty:
            for num_col in numeric_data.columns:
                for cat_col in categorical_data.columns:
                    key = f"{num_col}_vs_{cat_col}"
                    relationships['mixed_relationships'][key] = self._analyze_mixed_relationship(
                        numeric_data[num_col], categorical_data[cat_col]
                    )
        
        return relationships
    
    def _analyze_target(self, data: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Analyze target variable."""
        target_data = data[target]
        
        analysis = {
            'basic_info': {
                'data_type': str(target_data.dtype),
                'missing_values': target_data.isnull().sum(),
                'unique_values': target_data.nunique()
            }
        }
        
        if target_data.dtype in ['int64', 'float64']:
            # Numeric target
            analysis['numeric_analysis'] = {
                'mean': float(target_data.mean()),
                'median': float(target_data.median()),
                'std': float(target_data.std()),
                'min': float(target_data.min()),
                'max': float(target_data.max()),
                'skewness': float(stats.skew(target_data.dropna())),
                'kurtosis': float(stats.kurtosis(target_data.dropna())),
                'is_normal': self._test_normality(target_data.dropna())
            }
            
            # Check if it's classification or regression
            unique_ratio = target_data.nunique() / len(target_data)
            if unique_ratio < 0.1:  # Likely classification
                analysis['problem_type'] = 'classification'
                analysis['class_distribution'] = target_data.value_counts().to_dict()
                analysis['class_imbalance'] = self._check_class_imbalance(target_data)
            else:
                analysis['problem_type'] = 'regression'
        else:
            # Categorical target
            analysis['problem_type'] = 'classification'
            analysis['class_distribution'] = target_data.value_counts().to_dict()
            analysis['class_imbalance'] = self._check_class_imbalance(target_data)
        
        return analysis
    
    def _analyze_feature_importance_preliminary(self, data: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Perform preliminary feature importance analysis."""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        X = data.drop(columns=[target])
        y = data[target]
        
        # Remove non-numeric columns for this analysis
        X_numeric = X.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            return {'message': 'No numeric features for importance analysis'}
        
        importance_results = {}
        
        # Mutual Information
        try:
            if y.dtype in ['int64', 'float64'] and y.nunique() / len(y) < 0.1:
                mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
            
            importance_results['mutual_information'] = dict(zip(X_numeric.columns, mi_scores))
        except Exception as e:
            importance_results['mutual_information'] = {'error': str(e)}
        
        # Random Forest Importance
        try:
            if y.dtype in ['int64', 'float64'] and y.nunique() / len(y) < 0.1:
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
            
            rf.fit(X_numeric, y)
            importance_results['random_forest'] = dict(zip(X_numeric.columns, rf.feature_importances_))
        except Exception as e:
            importance_results['random_forest'] = {'error': str(e)}
        
        return importance_results
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test for normality using multiple methods."""
        if len(data) < 3:
            return {'is_normal': False, 'reason': 'Insufficient data'}
        
        try:
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(data)
            
            # Anderson-Darling test
            anderson_result = stats.anderson(data)
            
            # D'Agostino K^2 test
            dagostino_stat, dagostino_p = stats.normaltest(data)
            
            # Combined assessment
            is_normal = (shapiro_p > 0.05 and dagostino_p > 0.05)
            
            return {
                'is_normal': is_normal,
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'anderson_darling': {'statistic': anderson_result.statistic, 'critical_values': anderson_result.critical_values},
                'dagostino_k2': {'statistic': dagostino_stat, 'p_value': dagostino_p}
            }
        except Exception as e:
            return {'is_normal': False, 'error': str(e)}
    
    def _detect_outliers_iqr(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_indices': outliers.index.tolist()
        }
    
    def _detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > threshold]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'threshold': threshold,
            'outlier_indices': outliers.index.tolist()
        }
    
    def _detect_outliers_isolation_forest(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape data for sklearn
            X = data.values.reshape(-1, 1)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(X)
            
            outliers = data[predictions == -1]
            
            return {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(data)) * 100,
                'outlier_indices': outliers.index.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_outliers_lof(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # Reshape data for sklearn
            X = data.values.reshape(-1, 1)
            
            lof = LocalOutlierFactor(contamination=0.1)
            predictions = lof.fit_predict(X)
            
            outliers = data[predictions == -1]
            
            return {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(data)) * 100,
                'outlier_indices': outliers.index.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy for categorical data."""
        probabilities = value_counts / value_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    
    def _check_class_imbalance(self, target: pd.Series) -> Dict[str, Any]:
        """Check for class imbalance in classification targets."""
        value_counts = target.value_counts()
        total = len(target)
        
        imbalance_ratios = {}
        for class_name, count in value_counts.items():
            imbalance_ratios[str(class_name)] = count / total
        
        # Determine if imbalanced
        max_ratio = max(imbalance_ratios.values())
        min_ratio = min(imbalance_ratios.values())
        imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': imbalance_ratio > 2.0,
            'class_ratios': imbalance_ratios,
            'majority_class': value_counts.index[0],
            'minority_class': value_counts.index[-1]
        }
    
    def _check_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency issues."""
        issues = []
        
        # Check for inconsistent data types
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if numeric data is stored as object
                try:
                    pd.to_numeric(data[col], errors='raise')
                    issues.append(f"Column '{col}' contains numeric data stored as object")
                except:
                    pass
        
        # Check for mixed data types in object columns
        for col in data.select_dtypes(include=['object']).columns:
            unique_types = data[col].apply(type).nunique()
            if unique_types > 1:
                issues.append(f"Column '{col}' contains mixed data types")
        
        return {
            'issues_found': len(issues),
            'issues': issues
        }
    
    def _check_data_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness."""
        completeness = {}
        
        for col in data.columns:
            non_null_count = data[col].count()
            completeness[col] = {
                'completeness_ratio': non_null_count / len(data),
                'missing_count': len(data) - non_null_count
            }
        
        return completeness
    
    def _analyze_numeric_relationship(self, col1: pd.Series, col2: pd.Series) -> Dict[str, Any]:
        """Analyze relationship between two numeric columns."""
        # Remove NaN values
        valid_data = pd.DataFrame({'col1': col1, 'col2': col2}).dropna()
        
        if len(valid_data) < 2:
            return {'error': 'Insufficient valid data'}
        
        correlation = valid_data['col1'].corr(valid_data['col2'])
        spearman_corr = valid_data['col1'].corr(valid_data['col2'], method='spearman')
        
        return {
            'pearson_correlation': correlation,
            'spearman_correlation': spearman_corr,
            'relationship_strength': self._interpret_correlation(correlation),
            'sample_size': len(valid_data)
        }
    
    def _analyze_categorical_relationship(self, col1: pd.Series, col2: pd.Series) -> Dict[str, Any]:
        """Analyze relationship between two categorical columns."""
        # Create contingency table
        contingency_table = pd.crosstab(col1, col2)
        
        # Chi-square test
        try:
            chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
            is_significant = chi2_p < 0.05
        except:
            chi2_stat, chi2_p, is_significant = None, None, False
        
        # Cramer's V
        try:
            n = len(col1)
            min_dim = min(contingency_table.shape) - 1
            cramer_v = np.sqrt(chi2_stat / (n * min_dim)) if chi2_stat else 0
        except:
            cramer_v = 0
        
        return {
            'contingency_table_shape': contingency_table.shape,
            'chi_square_test': {
                'statistic': chi2_stat,
                'p_value': chi2_p,
                'is_significant': is_significant
            },
            'cramers_v': cramer_v,
            'relationship_strength': self._interpret_correlation(cramer_v)
        }
    
    def _analyze_mixed_relationship(self, numeric_col: pd.Series, categorical_col: pd.Series) -> Dict[str, Any]:
        """Analyze relationship between numeric and categorical columns."""
        # Group by categorical column and analyze numeric distribution
        grouped = numeric_col.groupby(categorical_col)
        
        group_stats = {}
        for group_name, group_data in grouped:
            group_stats[str(group_name)] = {
                'count': len(group_data),
                'mean': float(group_data.mean()),
                'std': float(group_data.std()),
                'median': float(group_data.median())
            }
        
        # ANOVA test
        try:
            groups = [group_data.values for _, group_data in grouped if len(group_data) > 0]
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                is_significant = p_value < 0.05
            else:
                f_stat, p_value, is_significant = None, None, False
        except:
            f_stat, p_value, is_significant = None, None, False
        
        return {
            'group_statistics': group_stats,
            'anova_test': {
                'f_statistic': f_stat,
                'p_value': p_value,
                'is_significant': is_significant
            }
        }
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "Very Strong"
        elif abs_corr >= 0.6:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def generate_analysis_report(self, output_path: str = "data_analysis_report.html") -> None:
        """Generate HTML report of the analysis results."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_dataset() first.")
        
        # Create HTML report
        html_content = self._create_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        if self.verbose:
            logger.info(f"Analysis report saved to {output_path}")
    
    def _create_html_report(self) -> str:
        """Create HTML content for the analysis report."""
        # This is a simplified version - in practice, you'd want a more sophisticated template
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .metric { margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Data Analysis Report</h1>
        """
        
        # Add sections based on analysis results
        for section_name, section_data in self.analysis_results.items():
            html += f"<div class='section'><h2>{section_name.replace('_', ' ').title()}</h2>"
            html += self._format_section_data(section_data)
            html += "</div>"
        
        html += "</body></html>"
        return html
    
    def _format_section_data(self, data: Any) -> str:
        """Format section data for HTML display."""
        if isinstance(data, dict):
            html = "<table><tr><th>Metric</th><th>Value</th></tr>"
            for key, value in data.items():
                if isinstance(value, dict):
                    value_str = "<br>".join([f"{k}: {v}" for k, v in value.items()])
                else:
                    value_str = str(value)
                html += f"<tr><td>{key}</td><td>{value_str}</td></tr>"
            html += "</table>"
            return html
        else:
            return f"<p>{str(data)}</p>" 
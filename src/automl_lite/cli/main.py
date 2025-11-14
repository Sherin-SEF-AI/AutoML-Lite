"""
Enhanced CLI for AutoML Lite with advanced features.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from ..core.automl import AutoMLite
from ..ui.terminal_ui import AutoMLTerminalUI
from ..utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="AutoML Lite - Automated Machine Learning for Non-Experts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  automl-lite interactive

  # Train a model
  automl-lite train data.csv --target target --output model.pkl

  # Train with custom configuration
  automl-lite train data.csv --target target --config config.json

  # Make predictions
  automl-lite predict model.pkl new_data.csv --output predictions.csv

  # Generate report
  automl-lite report model.pkl --output report.html

  # Compare models
  automl-lite compare data.csv --target target --models 5

  # Batch processing
  automl-lite batch config.json

  # Validate data
  automl-lite validate data.csv --target target
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Launch interactive terminal UI')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('data', help='Path to training data file')
    train_parser.add_argument('--target', required=True, help='Target column name')
    train_parser.add_argument('--output', default='model.pkl', help='Output model file path')
    train_parser.add_argument('--config', help='Configuration file path')
    train_parser.add_argument('--time-budget', type=int, default=300, help='Time budget in seconds')
    train_parser.add_argument('--max-models', type=int, default=10, help='Maximum number of models')
    train_parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    train_parser.add_argument('--enable-ensemble', action='store_true', default=True, help='Enable ensemble methods')
    train_parser.add_argument('--enable-feature-selection', action='store_true', default=True, help='Enable feature selection')
    train_parser.add_argument('--enable-interpretability', action='store_true', default=True, help='Enable model interpretability')
    train_parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('model', help='Path to trained model file')
    predict_parser.add_argument('data', help='Path to prediction data file')
    predict_parser.add_argument('--output', default='predictions.csv', help='Output predictions file path')
    predict_parser.add_argument('--proba', action='store_true', help='Output prediction probabilities')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate HTML report')
    report_parser.add_argument('model', help='Path to trained model file')
    report_parser.add_argument('--output', default='report.html', help='Output report file path')
    report_parser.add_argument('--data', help='Path to data file for additional analysis')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('data', help='Path to data file')
    compare_parser.add_argument('--target', required=True, help='Target column name')
    compare_parser.add_argument('--models', type=int, default=5, help='Number of models to compare')
    compare_parser.add_argument('--output', default='comparison.html', help='Output comparison file path')
    compare_parser.add_argument('--time-budget', type=int, default=600, help='Time budget in seconds')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch processing with configuration file')
    batch_parser.add_argument('config', help='Path to batch configuration file')
    batch_parser.add_argument('--output-dir', default='./batch_results', help='Output directory')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data quality')
    validate_parser.add_argument('data', help='Path to data file')
    validate_parser.add_argument('--target', help='Target column name')
    validate_parser.add_argument('--output', help='Output validation report file path')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('model', help='Path to trained model file')
    
    # NAS architecture management commands
    nas_parser = subparsers.add_parser('nas', help='Neural Architecture Search management')
    nas_subparsers = nas_parser.add_subparsers(dest='nas_command', help='NAS commands')
    
    # NAS list command
    nas_list_parser = nas_subparsers.add_parser('list', help='List saved architectures')
    nas_list_parser.add_argument('--problem-type', help='Filter by problem type')
    nas_list_parser.add_argument('--min-accuracy', type=float, help='Minimum accuracy threshold')
    nas_list_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    nas_list_parser.add_argument('--limit', type=int, default=20, help='Maximum number of results')
    nas_list_parser.add_argument('--repository', default='~/.automl_lite/nas_architectures.db', 
                                 help='Path to architecture repository')
    
    # NAS export command
    nas_export_parser = nas_subparsers.add_parser('export', help='Export architecture to file')
    nas_export_parser.add_argument('architecture_id', help='Architecture ID to export')
    nas_export_parser.add_argument('--output', required=True, help='Output JSON file path')
    nas_export_parser.add_argument('--include-metadata', action='store_true', default=True,
                                   help='Include metadata in export')
    nas_export_parser.add_argument('--repository', default='~/.automl_lite/nas_architectures.db',
                                   help='Path to architecture repository')
    
    # NAS import command
    nas_import_parser = nas_subparsers.add_parser('import', help='Import architecture from file')
    nas_import_parser.add_argument('input', help='Input JSON file path')
    nas_import_parser.add_argument('--validate', action='store_true', default=True,
                                   help='Validate architecture after import')
    nas_import_parser.add_argument('--repository', default='~/.automl_lite/nas_architectures.db',
                                   help='Path to architecture repository')
    
    # NAS view command
    nas_view_parser = nas_subparsers.add_parser('view', help='View architecture details')
    nas_view_parser.add_argument('architecture_id', help='Architecture ID to view')
    nas_view_parser.add_argument('--repository', default='~/.automl_lite/nas_architectures.db',
                                 help='Path to architecture repository')
    
    # NAS delete command
    nas_delete_parser = nas_subparsers.add_parser('delete', help='Delete architecture from repository')
    nas_delete_parser.add_argument('architecture_id', help='Architecture ID to delete')
    nas_delete_parser.add_argument('--repository', default='~/.automl_lite/nas_architectures.db',
                                   help='Path to architecture repository')
    nas_delete_parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompt')
    
    # NAS stats command
    nas_stats_parser = nas_subparsers.add_parser('stats', help='Show repository statistics')
    nas_stats_parser.add_argument('--repository', default='~/.automl_lite/nas_architectures.db',
                                  help='Path to architecture repository')
    
    return parser


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various file formats."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = path.suffix.lower()
    
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_data(data: pd.DataFrame, file_path: str) -> None:
    """Save data to various file formats."""
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext == '.csv':
        data.to_csv(file_path, index=False)
    elif ext in ['.xlsx', '.xls']:
        data.to_excel(file_path, index=False)
    elif ext == '.parquet':
        data.to_parquet(file_path, index=False)
    elif ext == '.json':
        data.to_json(file_path, orient='records')
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def train_model(args) -> None:
    """Train a model with the given arguments."""
    console.print(f"[bold cyan]Training model on {args.data}[/bold cyan]")
    
    # Load data
    with Progress(SpinnerColumn(), TextColumn("Loading data..."), console=console) as progress:
        task = progress.add_task("Loading data...", total=None)
        data = load_data(args.data)
        progress.update(task, completed=100)
    
    console.print(f"✅ Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Validate target column
    if args.target not in data.columns:
        console.print(f"❌ Target column '{args.target}' not found in data", style="bold red")
        sys.exit(1)
    
    # Prepare data
    X = data.drop(columns=[args.target])
    y = data[args.target]
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override with command line arguments
    config.update({
        'time_budget': args.time_budget,
        'max_models': args.max_models,
        'cv_folds': args.cv_folds,
        'enable_ensemble': args.enable_ensemble,
        'enable_feature_selection': args.enable_feature_selection,
        'enable_interpretability': args.enable_interpretability,
        'verbose': args.verbose,
    })
    
    # Train model
    with Progress(SpinnerColumn(), TextColumn("Training models..."), console=console) as progress:
        task = progress.add_task("Training models...", total=None)
        
        automl = AutoMLite(**config)
        automl.fit(X, y)
        
        progress.update(task, completed=100)
    
    # Save model
    automl.save_model(args.output)
    console.print(f"✅ Model saved to {args.output}", style="bold green")
    
    # Show results
    show_training_results(automl)


def predict(args) -> None:
    """Make predictions with a trained model."""
    console.print(f"[bold cyan]Making predictions with {args.model}[/bold cyan]")
    
    # Load model
    with Progress(SpinnerColumn(), TextColumn("Loading model..."), console=console) as progress:
        task = progress.add_task("Loading model...", total=None)
        automl = AutoMLite()
        automl.load_model(args.model)
        progress.update(task, completed=100)
    
    # Load data
    with Progress(SpinnerColumn(), TextColumn("Loading data..."), console=console) as progress:
        task = progress.add_task("Loading data...", total=None)
        data = load_data(args.data)
        progress.update(task, completed=100)
    
    # Make predictions
    with Progress(SpinnerColumn(), TextColumn("Making predictions..."), console=console) as progress:
        task = progress.add_task("Making predictions...", total=None)
        predictions = None
        pred_df = None
        if args.proba and automl.problem_type == "classification":
            # Check if model supports predict_proba
            model = automl.ensemble_model if automl.ensemble_model is not None else automl.best_model
            if hasattr(model, "predict_proba"):
                predictions = automl.predict_proba(data)
                if predictions.ndim == 2:
                    pred_df = pd.DataFrame(predictions, columns=[f'prob_class_{i}' for i in range(predictions.shape[1])])
                else:
                    pred_df = pd.DataFrame({'probabilities': predictions})
            else:
                console.print("⚠️  Probability predictions not available for this model, using regular predictions", style="yellow")
                predictions = automl.predict(data)
                pred_df = pd.DataFrame({'predictions': predictions})
        else:
            predictions = automl.predict(data)
            pred_df = pd.DataFrame({'predictions': predictions})
        progress.update(task, completed=100)
    
    # Save predictions
    save_data(pred_df, args.output)
    console.print(f"✅ Predictions saved to {args.output}", style="bold green")
    
    # Show prediction summary
    show_prediction_summary(predictions, args.proba)


def generate_report(args) -> None:
    """Generate HTML report."""
    console.print(f"[bold cyan]Generating report for {args.model}[/bold cyan]")
    
    # Load model
    with Progress(SpinnerColumn(), TextColumn("Loading model..."), console=console) as progress:
        task = progress.add_task("Loading model...", total=None)
        automl = AutoMLite()
        automl.load_model(args.model)
        progress.update(task, completed=100)
    
    # Generate report
    with Progress(SpinnerColumn(), TextColumn("Generating report..."), console=console) as progress:
        task = progress.add_task("Generating report...", total=None)
        automl.generate_report(args.output)
        progress.update(task, completed=100)
    
    console.print(f"✅ Report generated: {args.output}", style="bold green")
    
    # Show report summary
    show_report_summary(automl)


def compare_models(args) -> None:
    """Compare multiple models."""
    console.print(f"[bold cyan]Comparing {args.models} models on {args.data}[/bold cyan]")
    
    # Load data
    data = load_data(args.data)
    X = data.drop(columns=[args.target])
    y = data[args.target]
    
    # Train multiple models with different configurations
    results = []
    
    with Progress(SpinnerColumn(), TextColumn("Training models..."), console=console) as progress:
        task = progress.add_task("Training models...", total=args.models)
        
        for i in range(args.models):
            # Vary configuration slightly
            config = {
                'time_budget': args.time_budget // args.models,
                'max_models': 3,
                'cv_folds': 5,
                'enable_ensemble': i % 2 == 0,
                'enable_feature_selection': i % 2 == 1,
                'enable_interpretability': True,
                'verbose': False,
            }
            
            try:
                automl = AutoMLite(**config)
                automl.fit(X, y)
                
                results.append({
                    'config_id': i + 1,
                    'best_model': automl.best_model_name,
                    'best_score': automl.best_score,
                    'ensemble': automl.ensemble_model is not None,
                    'feature_selection': automl.selected_features is not None,
                })
                
            except Exception as e:
                console.print(f"⚠️  Model {i+1} failed: {str(e)}", style="yellow")
            
            progress.update(task, advance=1)
    
    # Show comparison results
    show_comparison_results(results, args.output)


def batch_process(args) -> None:
    """Process multiple datasets in batch."""
    console.print(f"[bold cyan]Batch processing with {args.config}[/bold cyan]")
    
    # Load batch configuration
    batch_config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for i, dataset_config in enumerate(batch_config['datasets']):
        console.print(f"\n[bold]Processing dataset {i+1}/{len(batch_config['datasets'])}: {dataset_config['name']}[/bold]")
        
        try:
            # Load data
            data = load_data(dataset_config['data_path'])
            X = data.drop(columns=[dataset_config['target']])
            y = data[dataset_config['target']]
            
            # Train model
            config = batch_config.get('default_config', {}).copy()
            config.update(dataset_config.get('config', {}))
            
            automl = AutoMLite(**config)
            automl.fit(X, y)
            
            # Save model
            model_path = output_dir / f"{dataset_config['name']}_model.pkl"
            automl.save_model(str(model_path))
            
            # Generate report
            report_path = output_dir / f"{dataset_config['name']}_report.html"
            automl.generate_report(str(report_path))
            
            results.append({
                'dataset': dataset_config['name'],
                'status': 'success',
                'best_score': automl.best_score,
                'best_model': automl.best_model_name,
                'model_path': str(model_path),
                'report_path': str(report_path),
            })
            
        except Exception as e:
            results.append({
                'dataset': dataset_config['name'],
                'status': 'failed',
                'error': str(e),
            })
    
    # Save batch results
    results_path = output_dir / 'batch_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Show batch results
    show_batch_results(results, str(results_path))


def validate_data(args) -> None:
    """Validate data quality."""
    console.print(f"[bold cyan]Validating data: {args.data}[/bold cyan]")
    
    # Load data
    data = load_data(args.data)
    
    # Perform validation
    validation_results = perform_data_validation(data, args.target)
    
    # Show validation results
    show_validation_results(validation_results, args.output)


def show_model_info(args) -> None:
    """Show model information."""
    console.print(f"[bold cyan]Model information: {args.model}[/bold cyan]")
    
    # Load model
    automl = AutoMLite()
    automl.load_model(args.model)
    
    # Show model info
    show_model_summary(automl)


def show_training_results(automl: AutoMLite) -> None:
    """Show training results in a table."""
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Best Model", automl.best_model_name)
    table.add_row("Best Score", f"{automl.best_score:.4f}")
    table.add_row("Problem Type", automl.problem_type)
    table.add_row("Ensemble", "✅" if automl.ensemble_model else "❌")
    table.add_row("Feature Selection", "✅" if automl.selected_features else "❌")
    table.add_row("Interpretability", "✅" if automl.interpretability_results else "❌")
    
    console.print(table)


def show_prediction_summary(predictions: np.ndarray, proba: bool) -> None:
    """Show prediction summary."""
    table = Table(title="Prediction Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Number of Predictions", str(len(predictions)))
    table.add_row("Mean", f"{np.mean(predictions):.4f}")
    table.add_row("Std", f"{np.std(predictions):.4f}")
    table.add_row("Min", f"{np.min(predictions):.4f}")
    table.add_row("Max", f"{np.max(predictions):.4f}")
    
    if proba:
        table.add_row("Type", "Probabilities")
    else:
        table.add_row("Type", "Predictions")
    
    console.print(table)


def show_report_summary(automl: AutoMLite) -> None:
    """Show report summary."""
    table = Table(title="Report Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Model Performance", "✅")
    table.add_row("Feature Importance", "✅" if automl.feature_importance else "❌")
    table.add_row("Model Comparison", "✅" if automl.leaderboard is not None else "❌")
    table.add_row("Training History", "✅" if automl.training_history else "❌")
    table.add_row("Ensemble Info", "✅" if automl.ensemble_model else "❌")
    table.add_row("Interpretability", "✅" if automl.interpretability_results else "❌")
    
    console.print(table)


def show_comparison_results(results: list, output_path: str) -> None:
    """Show model comparison results."""
    table = Table(title="Model Comparison Results")
    table.add_column("Config ID", style="cyan")
    table.add_column("Best Model", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("Ensemble", style="yellow")
    table.add_column("Feature Selection", style="yellow")
    
    for result in results:
        table.add_row(
            str(result['config_id']),
            result['best_model'],
            f"{result['best_score']:.4f}",
            "✅" if result['ensemble'] else "❌",
            "✅" if result['feature_selection'] else "❌"
        )
    
    console.print(table)
    
    # Save comparison results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"✅ Comparison results saved to {output_path}", style="bold green")


def show_batch_results(results: list, results_path: str) -> None:
    """Show batch processing results."""
    table = Table(title="Batch Processing Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Best Score", style="yellow")
    table.add_column("Best Model", style="magenta")
    
    for result in results:
        if result['status'] == 'success':
            table.add_row(
                result['dataset'],
                "✅ Success",
                f"{result['best_score']:.4f}",
                result['best_model']
            )
        else:
            table.add_row(
                result['dataset'],
                "❌ Failed",
                result['error'],
                "N/A"
            )
    
    console.print(table)
    console.print(f"✅ Batch results saved to {results_path}", style="bold green")


def perform_data_validation(data: pd.DataFrame, target: Optional[str]) -> Dict[str, Any]:
    """Perform comprehensive data validation."""
    results = {
        'basic_info': {
            'rows': len(data),
            'columns': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum(),
        },
        'missing_values': {},
        'data_types': {},
        'duplicates': data.duplicated().sum(),
        'target_analysis': {},
    }
    
    # Missing values
    missing = data.isnull().sum()
    results['missing_values'] = {
        'total_missing': missing.sum(),
        'columns_with_missing': (missing > 0).sum(),
        'missing_percentage': (missing.sum() / (len(data) * len(data.columns))) * 100,
        'missing_by_column': missing[missing > 0].to_dict()
    }
    
    # Data types
    results['data_types'] = data.dtypes.value_counts().to_dict()
    
    # Target analysis
    if target and target in data.columns:
        target_data = data[target]
        results['target_analysis'] = {
            'unique_values': target_data.nunique(),
            'missing_values': target_data.isnull().sum(),
            'data_type': str(target_data.dtype),
        }
        
        if target_data.dtype in ['int64', 'float64']:
            results['target_analysis'].update({
                'mean': float(target_data.mean()),
                'std': float(target_data.std()),
                'min': float(target_data.min()),
                'max': float(target_data.max()),
            })
    
    return results


def show_validation_results(results: Dict[str, Any], output_path: Optional[str]) -> None:
    """Show data validation results."""
    # Basic info
    basic_table = Table(title="Basic Information")
    basic_table.add_column("Metric", style="cyan")
    basic_table.add_column("Value", style="green")
    
    for key, value in results['basic_info'].items():
        if key == 'memory_usage':
            value = f"{value / 1024 / 1024:.2f} MB"
        basic_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(basic_table)
    
    # Missing values
    missing_table = Table(title="Missing Values Analysis")
    missing_table.add_column("Metric", style="cyan")
    missing_table.add_column("Value", style="green")
    
    missing_info = results['missing_values']
    missing_table.add_row("Total Missing", str(missing_info['total_missing']))
    missing_table.add_row("Columns with Missing", str(missing_info['columns_with_missing']))
    missing_table.add_row("Missing Percentage", f"{missing_info['missing_percentage']:.2f}%")
    
    console.print(missing_table)
    
    # Data types
    type_table = Table(title="Data Types")
    type_table.add_column("Data Type", style="cyan")
    type_table.add_column("Count", style="green")
    
    for dtype, count in results['data_types'].items():
        type_table.add_row(str(dtype), str(count))
    
    console.print(type_table)
    
    # Target analysis
    if results['target_analysis']:
        target_table = Table(title="Target Analysis")
        target_table.add_column("Metric", style="cyan")
        target_table.add_column("Value", style="green")
        
        for key, value in results['target_analysis'].items():
            target_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(target_table)
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"✅ Validation results saved to {output_path}", style="bold green")


def show_model_summary(automl: AutoMLite) -> None:
    """Show model summary information."""
    table = Table(title="Model Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Best Model", automl.best_model_name)
    table.add_row("Best Score", f"{automl.best_score:.4f}")
    table.add_row("Problem Type", automl.problem_type)
    table.add_row("Is Fitted", "✅" if automl.is_fitted else "❌")
    table.add_row("Ensemble Model", "✅" if automl.ensemble_model else "❌")
    table.add_row("Feature Selection", "✅" if automl.selected_features else "❌")
    table.add_row("Interpretability", "✅" if automl.interpretability_results else "❌")
    
    if automl.leaderboard is not None:
        table.add_row("Models Trained", str(len(automl.leaderboard)))
    
    if automl.training_history:
        table.add_row("Training History", f"{len(automl.training_history)} entries")
    
    console.print(table)


def nas_list_architectures(args) -> None:
    """List saved architectures in the repository."""
    from ..nas.repository import ArchitectureRepository
    
    console.print(f"[bold cyan]Listing architectures from {args.repository}[/bold cyan]")
    
    try:
        with ArchitectureRepository(args.repository) as repo:
            # List architectures with filters
            architectures = repo.list_architectures(
                problem_type=args.problem_type,
                min_accuracy=args.min_accuracy,
                tags=args.tags,
                limit=args.limit
            )
            
            if not architectures:
                console.print("No architectures found matching the criteria", style="yellow")
                return
            
            # Create table
            table = Table(title=f"Saved Architectures ({len(architectures)} found)")
            table.add_column("ID", style="cyan")
            table.add_column("Problem Type", style="magenta")
            table.add_column("Samples", justify="right")
            table.add_column("Features", justify="right")
            table.add_column("Accuracy", justify="right", style="green")
            table.add_column("Latency (ms)", justify="right")
            table.add_column("Size (MB)", justify="right")
            table.add_column("Created", style="dim")
            
            for arch_id, summary in architectures:
                table.add_row(
                    arch_id[:12] + "...",
                    summary.get('problem_type', 'N/A'),
                    str(summary.get('n_samples', 'N/A')),
                    str(summary.get('n_features', 'N/A')),
                    f"{summary.get('accuracy', 0):.4f}" if summary.get('accuracy') else 'N/A',
                    f"{summary.get('latency_ms', 0):.2f}" if summary.get('latency_ms') else 'N/A',
                    f"{summary.get('model_size_mb', 0):.2f}" if summary.get('model_size_mb') else 'N/A',
                    summary.get('created_at', 'N/A')[:10]
                )
            
            console.print(table)
            console.print(f"✅ Listed {len(architectures)} architecture(s)")
    
    except Exception as e:
        console.print(f"❌ Error listing architectures: {e}", style="bold red")
        sys.exit(1)


def nas_export_architecture(args) -> None:
    """Export architecture to JSON file."""
    from ..nas.repository import ArchitectureRepository
    
    console.print(f"[bold cyan]Exporting architecture {args.architecture_id[:12]}...[/bold cyan]")
    
    try:
        with ArchitectureRepository(args.repository) as repo:
            success = repo.export_architecture(
                args.architecture_id,
                args.output,
                include_metadata=args.include_metadata
            )
            
            if success:
                console.print(f"✅ Architecture exported to {args.output}")
            else:
                console.print(f"❌ Failed to export architecture", style="bold red")
                sys.exit(1)
    
    except Exception as e:
        console.print(f"❌ Error exporting architecture: {e}", style="bold red")
        sys.exit(1)


def nas_import_architecture(args) -> None:
    """Import architecture from JSON file."""
    from ..nas.repository import ArchitectureRepository
    
    console.print(f"[bold cyan]Importing architecture from {args.input}[/bold cyan]")
    
    try:
        with ArchitectureRepository(args.repository) as repo:
            architecture = repo.import_architecture(
                args.input,
                validate=args.validate,
                save_to_repository=True
            )
            
            if architecture:
                console.print(f"✅ Architecture imported successfully: {architecture.id}")
            else:
                console.print(f"❌ Failed to import architecture", style="bold red")
                sys.exit(1)
    
    except Exception as e:
        console.print(f"❌ Error importing architecture: {e}", style="bold red")
        sys.exit(1)


def nas_view_architecture(args) -> None:
    """View architecture details."""
    from ..nas.repository import ArchitectureRepository
    
    console.print(f"[bold cyan]Viewing architecture {args.architecture_id[:12]}...[/bold cyan]")
    
    try:
        with ArchitectureRepository(args.repository) as repo:
            result = repo.load_architecture(args.architecture_id)
            
            if not result:
                console.print(f"❌ Architecture not found: {args.architecture_id}", style="bold red")
                sys.exit(1)
            
            architecture, metadata = result
            
            # Display architecture details
            console.print(Panel(f"[bold]Architecture: {architecture.id}[/bold]"))
            
            # Basic info
            console.print("\n[bold]Basic Information:[/bold]")
            console.print(f"  Layers: {len(architecture.layers)}")
            console.print(f"  Connections: {len(architecture.connections)}")
            console.print(f"  Created: {metadata.get('created_at', 'N/A')}")
            
            # Dataset metadata
            if 'dataset_metadata' in metadata:
                dm = metadata['dataset_metadata']
                console.print("\n[bold]Dataset:[/bold]")
                console.print(f"  Problem Type: {dm.get('problem_type', 'N/A')}")
                console.print(f"  Samples: {dm.get('n_samples', 'N/A')}")
                console.print(f"  Features: {dm.get('n_features', 'N/A')}")
                if dm.get('n_classes'):
                    console.print(f"  Classes: {dm.get('n_classes')}")
            
            # Performance metrics
            if 'performance_metrics' in metadata:
                pm = metadata['performance_metrics']
                console.print("\n[bold]Performance:[/bold]")
                if pm.get('accuracy'):
                    console.print(f"  Accuracy: {pm['accuracy']:.4f}")
                if pm.get('val_accuracy'):
                    console.print(f"  Val Accuracy: {pm['val_accuracy']:.4f}")
                if pm.get('training_time'):
                    console.print(f"  Training Time: {pm['training_time']:.2f}s")
            
            # Hardware metrics
            if 'hardware_metrics' in metadata:
                hm = metadata['hardware_metrics']
                console.print("\n[bold]Hardware:[/bold]")
                if hm.get('latency_ms'):
                    console.print(f"  Latency: {hm['latency_ms']:.2f} ms")
                if hm.get('memory_mb'):
                    console.print(f"  Memory: {hm['memory_mb']:.2f} MB")
                if hm.get('model_size_mb'):
                    console.print(f"  Model Size: {hm['model_size_mb']:.2f} MB")
                if hm.get('num_parameters'):
                    console.print(f"  Parameters: {hm['num_parameters']:,}")
            
            # Layers
            console.print("\n[bold]Layers:[/bold]")
            for i, layer in enumerate(architecture.layers):
                console.print(f"  {i}: {layer.layer_type} - {layer.params}")
            
            console.print("\n✅ Architecture details displayed")
    
    except Exception as e:
        console.print(f"❌ Error viewing architecture: {e}", style="bold red")
        sys.exit(1)


def nas_delete_architecture(args) -> None:
    """Delete architecture from repository."""
    from ..nas.repository import ArchitectureRepository
    
    # Confirm deletion
    if not args.confirm:
        response = input(f"Are you sure you want to delete architecture {args.architecture_id[:12]}...? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            console.print("❌ Deletion cancelled", style="yellow")
            return
    
    console.print(f"[bold cyan]Deleting architecture {args.architecture_id[:12]}...[/bold cyan]")
    
    try:
        with ArchitectureRepository(args.repository) as repo:
            success = repo.delete_architecture(args.architecture_id)
            
            if success:
                console.print(f"✅ Architecture deleted successfully")
            else:
                console.print(f"❌ Architecture not found", style="bold red")
                sys.exit(1)
    
    except Exception as e:
        console.print(f"❌ Error deleting architecture: {e}", style="bold red")
        sys.exit(1)


def nas_show_stats(args) -> None:
    """Show repository statistics."""
    from ..nas.repository import ArchitectureRepository
    
    console.print(f"[bold cyan]Repository Statistics[/bold cyan]")
    
    try:
        with ArchitectureRepository(args.repository) as repo:
            stats = repo.get_statistics()
            
            if not stats:
                console.print("No statistics available", style="yellow")
                return
            
            # Display statistics
            console.print(f"\n[bold]Total Architectures:[/bold] {stats.get('total_architectures', 0)}")
            
            if stats.get('by_problem_type'):
                console.print("\n[bold]By Problem Type:[/bold]")
                for ptype, count in stats['by_problem_type'].items():
                    console.print(f"  {ptype}: {count}")
            
            if stats.get('avg_accuracy'):
                console.print(f"\n[bold]Average Accuracy:[/bold] {stats['avg_accuracy']:.4f}")
            if stats.get('max_accuracy'):
                console.print(f"[bold]Max Accuracy:[/bold] {stats['max_accuracy']:.4f}")
            
            if stats.get('top_tags'):
                console.print("\n[bold]Top Tags:[/bold]")
                for tag, count in list(stats['top_tags'].items())[:10]:
                    console.print(f"  {tag}: {count}")
            
            console.print("\n✅ Statistics displayed")
    
    except Exception as e:
        console.print(f"❌ Error getting statistics: {e}", style="bold red")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'interactive':
            ui = AutoMLTerminalUI()
            ui.run()
        elif args.command == 'train':
            train_model(args)
        elif args.command == 'predict':
            predict(args)
        elif args.command == 'report':
            generate_report(args)
        elif args.command == 'compare':
            compare_models(args)
        elif args.command == 'batch':
            batch_process(args)
        elif args.command == 'validate':
            validate_data(args)
        elif args.command == 'info':
            show_model_info(args)
        elif args.command == 'nas':
            # Handle NAS subcommands
            if not args.nas_command:
                console.print("❌ No NAS subcommand specified", style="bold red")
                sys.exit(1)
            
            if args.nas_command == 'list':
                nas_list_architectures(args)
            elif args.nas_command == 'export':
                nas_export_architecture(args)
            elif args.nas_command == 'import':
                nas_import_architecture(args)
            elif args.nas_command == 'view':
                nas_view_architecture(args)
            elif args.nas_command == 'delete':
                nas_delete_architecture(args)
            elif args.nas_command == 'stats':
                nas_show_stats(args)
            else:
                console.print(f"❌ Unknown NAS command: {args.nas_command}", style="bold red")
                sys.exit(1)
        else:
            console.print(f"❌ Unknown command: {args.command}", style="bold red")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n👋 Operation cancelled by user", style="bold yellow")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="bold red")
        logger.error(f"CLI Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
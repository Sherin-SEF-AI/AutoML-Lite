"""
Terminal UI for AutoML Lite.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.align import Align

from ..core.automl import AutoMLite
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AutoMLTerminalUI:
    """Interactive terminal UI for AutoML Lite."""
    
    def __init__(self):
        self.console = Console()
        self.automl = None
        self.data = None
        self.target_column = None
        
    def run(self):
        """Run the main UI loop."""
        self.console.clear()
        self._show_welcome()
        
        while True:
            try:
                choice = self._show_main_menu()
                
                if choice == "1":
                    self._load_data()
                elif choice == "2":
                    self._configure_automl()
                elif choice == "3":
                    self._train_model()
                elif choice == "4":
                    self._view_results()
                elif choice == "5":
                    self._make_predictions()
                elif choice == "6":
                    self._generate_report()
                elif choice == "7":
                    self._save_load_model()
                elif choice == "8":
                    self._show_help()
                elif choice == "9":
                    if Confirm.ask("Are you sure you want to exit?"):
                        self.console.print("👋 Goodbye!", style="bold green")
                        break
                else:
                    self.console.print("❌ Invalid choice. Please try again.", style="bold red")
                    
            except KeyboardInterrupt:
                if Confirm.ask("\nAre you sure you want to exit?"):
                    self.console.print("👋 Goodbye!", style="bold green")
                    break
            except Exception as e:
                self.console.print(f"❌ Error: {str(e)}", style="bold red")
                logger.error(f"UI Error: {str(e)}")
    
    def _show_welcome(self):
        """Show welcome screen."""
        welcome_text = """
╔══════════════════════════════════════════════════════════════╗
║                    🚀 AutoML Lite Terminal UI                ║
║                                                              ║
║  Automated Machine Learning for Non-Experts                 ║
║  Powered by scikit-learn, Optuna, and SHAP                  ║
║                                                              ║
║  Features:                                                   ║
║  • Automatic problem detection                               ║
║  • Intelligent preprocessing                                 ║
║  • Model selection & optimization                           ║
║  • Ensemble methods                                         ║
║  • Model interpretability                                   ║
║  • Beautiful reports                                        ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        panel = Panel(
            welcome_text,
            title="[bold blue]Welcome to AutoML Lite[/bold blue]",
            border_style="blue"
        )
        self.console.print(panel)
    
    def _show_main_menu(self) -> str:
        """Show main menu and get user choice."""
        menu_text = """
[bold cyan]Main Menu:[/bold cyan]

1. 📁 Load Data
2. ⚙️  Configure AutoML
3. 🚀 Train Model
4. 📊 View Results
5. 🔮 Make Predictions
6. 📈 Generate Report
7. 💾 Save/Load Model
8. ❓ Help
9. 🚪 Exit

[dim]Current Status:[/dim]
"""
        
        # Add current status
        if self.data is not None:
            menu_text += f"📁 Data: {self.data.shape[0]} samples, {self.data.shape[1]} features\n"
        else:
            menu_text += "📁 Data: [red]Not loaded[/red]\n"
            
        if self.automl is not None and self.automl.is_fitted:
            menu_text += f"🤖 Model: {self.automl.best_model_name} ({self.automl.best_score:.4f})\n"
        else:
            menu_text += "🤖 Model: [red]Not trained[/red]\n"
        
        panel = Panel(menu_text, title="[bold blue]AutoML Lite[/bold blue]", border_style="blue")
        self.console.print(panel)
        
        return Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"])
    
    def _load_data(self):
        """Load data from file."""
        self.console.print("\n[bold cyan]📁 Load Data[/bold cyan]")
        
        # Get file path
        file_path = Prompt.ask("Enter data file path (CSV, Excel, or Parquet)")
        
        if not os.path.exists(file_path):
            self.console.print(f"❌ File not found: {file_path}", style="bold red")
            return
        
        try:
            # Load data based on file extension
            ext = Path(file_path).suffix.lower()
            if ext == '.csv':
                self.data = pd.read_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            elif ext == '.parquet':
                self.data = pd.read_parquet(file_path)
            else:
                self.console.print(f"❌ Unsupported file format: {ext}", style="bold red")
                return
            
            self.console.print(f"✅ Data loaded successfully!", style="bold green")
            self.console.print(f"   Shape: {self.data.shape[0]} samples, {self.data.shape[1]} features")
            
            # Show data preview
            self._show_data_preview()
            
            # Ask for target column
            self._select_target_column()
            
        except Exception as e:
            self.console.print(f"❌ Error loading data: {str(e)}", style="bold red")
    
    def _show_data_preview(self):
        """Show data preview."""
        if self.data is None:
            return
        
        # Create preview table
        table = Table(title="Data Preview")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Missing", style="yellow")
        table.add_column("Unique", style="green")
        table.add_column("Sample Values", style="white")
        
        for col in self.data.columns[:10]:  # Show first 10 columns
            col_data = self.data[col]
            missing = col_data.isnull().sum()
            unique = col_data.nunique()
            sample_values = ", ".join(str(x) for x in col_data.dropna().head(3).tolist())
            
            table.add_row(
                col,
                str(col_data.dtype),
                str(missing),
                str(unique),
                sample_values[:50] + "..." if len(sample_values) > 50 else sample_values
            )
        
        self.console.print(table)
    
    def _select_target_column(self):
        """Select target column."""
        if self.data is None:
            return
        
        self.console.print("\n[bold cyan]Select Target Column:[/bold cyan]")
        
        # Show available columns
        for i, col in enumerate(self.data.columns, 1):
            self.console.print(f"{i}. {col}")
        
        while True:
            try:
                choice = Prompt.ask("Enter column number or name")
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.data.columns):
                        self.target_column = self.data.columns[idx]
                        break
                else:
                    if choice in self.data.columns:
                        self.target_column = choice
                        break
                
                self.console.print("❌ Invalid choice. Please try again.", style="bold red")
                
            except ValueError:
                self.console.print("❌ Invalid input. Please try again.", style="bold red")
        
        self.console.print(f"✅ Target column selected: {self.target_column}", style="bold green")
    
    def _configure_automl(self):
        """Configure AutoML settings."""
        self.console.print("\n[bold cyan]⚙️ Configure AutoML[/bold cyan]")
        
        config = {}
        
        # Time budget
        time_budget = Prompt.ask(
            "Time budget (seconds)",
            default="300",
            show_default=True
        )
        config['time_budget'] = int(time_budget)
        
        # Max models
        max_models = Prompt.ask(
            "Maximum number of models",
            default="10",
            show_default=True
        )
        config['max_models'] = int(max_models)
        
        # CV folds
        cv_folds = Prompt.ask(
            "Cross-validation folds",
            default="5",
            show_default=True
        )
        config['cv_folds'] = int(cv_folds)
        
        # Advanced features
        config['enable_ensemble'] = Confirm.ask("Enable ensemble methods?", default=True)
        config['enable_early_stopping'] = Confirm.ask("Enable early stopping?", default=True)
        config['enable_feature_selection'] = Confirm.ask("Enable feature selection?", default=True)
        config['enable_interpretability'] = Confirm.ask("Enable model interpretability?", default=True)
        
        # Create AutoML instance
        self.automl = AutoMLite(**config)
        
        self.console.print("✅ AutoML configured successfully!", style="bold green")
        
        # Show configuration
        self._show_configuration(config)
    
    def _show_configuration(self, config: Dict[str, Any]):
        """Show current configuration."""
        table = Table(title="AutoML Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in config.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        self.console.print(table)
    
    def _train_model(self):
        """Train the model with live progress."""
        if self.data is None or self.target_column is None:
            self.console.print("❌ Please load data and select target column first.", style="bold red")
            return
        
        if self.automl is None:
            self.console.print("❌ Please configure AutoML first.", style="bold red")
            return
        
        self.console.print("\n[bold cyan]🚀 Training Model[/bold cyan]")
        
        # Prepare data
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        self.console.print(f"Training on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Train with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Training models...", total=None)
            
            try:
                # Override the progress callback
                original_info = logger.info
                
                def progress_callback(message):
                    if "Training completed" in message:
                        progress.update(task, completed=100)
                    elif "Training models" in message:
                        progress.update(task, description=message)
                
                # Train the model
                self.automl.fit(X, y)
                
                progress.update(task, completed=100)
                
                self.console.print("✅ Training completed successfully!", style="bold green")
                self._show_training_results()
                
            except Exception as e:
                self.console.print(f"❌ Training failed: {str(e)}", style="bold red")
                logger.error(f"Training error: {str(e)}")
    
    def _show_training_results(self):
        """Show training results."""
        if self.automl is None or not self.automl.is_fitted:
            return
        
        # Create results table
        table = Table(title="Training Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Best Model", self.automl.best_model_name)
        table.add_row("Best Score", f"{self.automl.best_score:.4f}")
        table.add_row("Problem Type", self.automl.problem_type)
        
        if self.automl.ensemble_model:
            table.add_row("Ensemble", "✅ Created")
        
        if self.automl.selected_features:
            table.add_row("Feature Selection", f"✅ {len(self.automl.selected_features)} features selected")
        
        self.console.print(table)
        
        # Show leaderboard
        self._show_leaderboard()
    
    def _show_leaderboard(self):
        """Show model leaderboard."""
        if self.automl.leaderboard is None:
            return
        
        table = Table(title="Model Leaderboard")
        table.add_column("Rank", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Score", style="green")
        table.add_column("Parameters", style="yellow")
        
        for _, row in self.automl.leaderboard.iterrows():
            params_str = str(row['params'])[:50] + "..." if len(str(row['params'])) > 50 else str(row['params'])
            table.add_row(
                str(row['rank']),
                row['model_name'],
                f"{row['score']:.4f}",
                params_str
            )
        
        self.console.print(table)
    
    def _view_results(self):
        """View detailed results."""
        if self.automl is None or not self.automl.is_fitted:
            self.console.print("❌ No trained model available.", style="bold red")
            return
        
        self.console.print("\n[bold cyan]📊 View Results[/bold cyan]")
        
        # Show feature importance
        if self.automl.feature_importance:
            self._show_feature_importance()
        
        # Show ensemble info
        if self.automl.ensemble_model:
            self._show_ensemble_info()
        
        # Show interpretability results
        if self.automl.interpretability_results:
            self._show_interpretability_info()
    
    def _show_feature_importance(self):
        """Show feature importance."""
        importance_df = self.automl.get_feature_importance()
        
        if importance_df.empty:
            return
        
        table = Table(title="Feature Importance")
        table.add_column("Rank", style="cyan")
        table.add_column("Feature", style="magenta")
        table.add_column("Importance", style="green")
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            table.add_row(
                str(i),
                row['feature'],
                f"{row['importance']:.4f}"
            )
        
        self.console.print(table)
    
    def _show_ensemble_info(self):
        """Show ensemble information."""
        ensemble_info = self.automl.get_ensemble_info()
        
        if not ensemble_info:
            return
        
        table = Table(title="Ensemble Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in ensemble_info.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        self.console.print(table)
    
    def _show_interpretability_info(self):
        """Show interpretability information."""
        interpretability = self.automl.get_interpretability_report()
        
        if not interpretability:
            return
        
        table = Table(title="Model Interpretability")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in interpretability.items():
            if key != 'shap_values':  # Don't show SHAP values in table
                table.add_row(key.replace('_', ' ').title(), str(value))
        
        self.console.print(table)
    
    def _make_predictions(self):
        """Make predictions on new data."""
        if self.automl is None or not self.automl.is_fitted:
            self.console.print("❌ No trained model available.", style="bold red")
            return
        
        self.console.print("\n[bold cyan]🔮 Make Predictions[/bold cyan]")
        
        # Get prediction data
        choice = Prompt.ask(
            "Choose prediction method",
            choices=["1", "2"],
            default="1"
        )
        
        if choice == "1":
            # Use existing data
            if self.data is None:
                self.console.print("❌ No data available.", style="bold red")
                return
            
            X_pred = self.data.drop(columns=[self.target_column])
            predictions = self.automl.predict(X_pred)
            
            # Show predictions
            self._show_predictions(predictions, self.data[self.target_column])
            
        else:
            # Load new data
            file_path = Prompt.ask("Enter prediction data file path")
            
            if not os.path.exists(file_path):
                self.console.print(f"❌ File not found: {file_path}", style="bold red")
                return
            
            try:
                # Load prediction data
                ext = Path(file_path).suffix.lower()
                if ext == '.csv':
                    pred_data = pd.read_csv(file_path)
                elif ext in ['.xlsx', '.xls']:
                    pred_data = pd.read_excel(file_path)
                elif ext == '.parquet':
                    pred_data = pd.read_parquet(file_path)
                else:
                    self.console.print(f"❌ Unsupported file format: {ext}", style="bold red")
                    return
                
                predictions = self.automl.predict(pred_data)
                
                # Save predictions
                output_path = Prompt.ask("Enter output file path for predictions", default="predictions.csv")
                pred_df = pd.DataFrame({'predictions': predictions})
                pred_df.to_csv(output_path, index=False)
                
                self.console.print(f"✅ Predictions saved to {output_path}", style="bold green")
                self._show_predictions(predictions)
                
            except Exception as e:
                self.console.print(f"❌ Error making predictions: {str(e)}", style="bold red")
    
    def _show_predictions(self, predictions: np.ndarray, actual: Optional[pd.Series] = None):
        """Show prediction results."""
        table = Table(title="Predictions")
        table.add_column("Sample", style="cyan")
        table.add_column("Prediction", style="green")
        
        if actual is not None:
            table.add_column("Actual", style="yellow")
            table.add_column("Error", style="red")
        
        for i, pred in enumerate(predictions[:10]):  # Show first 10
            row = [str(i + 1), f"{pred:.4f}"]
            
            if actual is not None:
                actual_val = actual.iloc[i]
                error = abs(pred - actual_val)
                row.extend([f"{actual_val:.4f}", f"{error:.4f}"])
            
            table.add_row(*row)
        
        self.console.print(table)
        
        if len(predictions) > 10:
            self.console.print(f"... and {len(predictions) - 10} more predictions")
    
    def _generate_report(self):
        """Generate HTML report."""
        if self.automl is None or not self.automl.is_fitted:
            self.console.print("❌ No trained model available.", style="bold red")
            return
        
        self.console.print("\n[bold cyan]📈 Generate Report[/bold cyan]")
        
        output_path = Prompt.ask("Enter report file path", default="automl_report.html")
        
        try:
            with Progress(SpinnerColumn(), TextColumn("Generating report..."), console=self.console) as progress:
                task = progress.add_task("Generating report...", total=None)
                
                self.automl.generate_report(output_path)
                
                progress.update(task, completed=100)
            
            self.console.print(f"✅ Report generated successfully: {output_path}", style="bold green")
            
            # Ask if user wants to open the report
            if Confirm.ask("Open report in browser?"):
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
                
        except Exception as e:
            self.console.print(f"❌ Error generating report: {str(e)}", style="bold red")
    
    def _save_load_model(self):
        """Save or load model."""
        self.console.print("\n[bold cyan]💾 Save/Load Model[/bold cyan]")
        
        choice = Prompt.ask(
            "Choose action",
            choices=["1", "2"],
            default="1"
        )
        
        if choice == "1":
            # Save model
            if self.automl is None or not self.automl.is_fitted:
                self.console.print("❌ No trained model to save.", style="bold red")
                return
            
            file_path = Prompt.ask("Enter model file path", default="automl_model.pkl")
            
            try:
                self.automl.save_model(file_path)
                self.console.print(f"✅ Model saved successfully: {file_path}", style="bold green")
            except Exception as e:
                self.console.print(f"❌ Error saving model: {str(e)}", style="bold red")
        
        else:
            # Load model
            file_path = Prompt.ask("Enter model file path")
            
            if not os.path.exists(file_path):
                self.console.print(f"❌ File not found: {file_path}", style="bold red")
                return
            
            try:
                self.automl = AutoMLite()
                self.automl.load_model(file_path)
                self.console.print(f"✅ Model loaded successfully: {file_path}", style="bold green")
            except Exception as e:
                self.console.print(f"❌ Error loading model: {str(e)}", style="bold red")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
[bold cyan]AutoML Lite Help[/bold cyan]

[bold]Getting Started:[/bold]
1. Load your data (CSV, Excel, or Parquet format)
2. Select the target column you want to predict
3. Configure AutoML settings (or use defaults)
4. Train the model
5. View results and generate reports

[bold]Supported File Formats:[/bold]
• CSV (.csv)
• Excel (.xlsx, .xls)
• Parquet (.parquet)

[bold]Advanced Features:[/bold]
• Ensemble Methods: Combines multiple models for better performance
• Early Stopping: Stops training when no improvement is seen
• Feature Selection: Automatically selects the most important features
• Model Interpretability: Provides SHAP explanations for predictions

[bold]Tips:[/bold]
• For large datasets, increase the time budget
• Enable ensemble methods for better performance
• Use feature selection for high-dimensional data
• Generate reports for detailed analysis

[bold]Keyboard Shortcuts:[/bold]
• Ctrl+C: Exit the application
• Enter: Confirm selection
        """
        
        panel = Panel(help_text, title="[bold blue]Help[/bold blue]", border_style="blue")
        self.console.print(panel)
        
        Prompt.ask("Press Enter to continue")


def main():
    """Main entry point for terminal UI."""
    ui = AutoMLTerminalUI()
    ui.run()


if __name__ == "__main__":
    main() 
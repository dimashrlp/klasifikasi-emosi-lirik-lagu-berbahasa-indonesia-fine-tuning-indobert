#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1 Macro Evaluator - Performance Data Loader
For Indonesian Lyrics Emotion Detection Thesis

This module helps load and process model performance data
for the Streamlit application and thesis evaluation.

Author: Thesis Project
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Configuration
BASE_PATH = Path.cwd()
if "streamlit_app" in str(BASE_PATH):
    BASE_PATH = BASE_PATH.parent

MODELS_PATH = BASE_PATH / "models"
DATASETS_PATH = BASE_PATH / "datasets"

class F1MacroAppEvaluator:
    """
    Evaluator class for loading model performance data
    Specifically designed for thesis application needs
    """
    
    def __init__(self):
        self.models_path = MODELS_PATH
        self.datasets_path = DATASETS_PATH
        self.emotion_labels = {0: 'bahagia', 1: 'sedih', 2: 'marah', 3: 'takut'}
        self.available_configs = self._get_available_configs()
    
    def _get_available_configs(self) -> List[int]:
        """Get list of available model configuration IDs"""
        configs = []
        if not self.models_path.exists():
            return configs
        
        for i in range(1, 13):
            model_path = self.models_path / f"model-config-{i}"
            if model_path.exists() and (model_path / 'config.json').exists():
                configs.append(i)
        
        return sorted(configs)
    
    def load_training_f1_results(self) -> Dict[int, Dict]:
        """
        Load F1-Macro and other training results from all model configurations
        
        Returns:
            Dict with config_id as key and performance metrics as value
        """
        results = {}
        
        for config_id in range(1, 13):
            config_data = {
                'config_id': config_id,
                'available': False,
                'test_f1_macro': 0.0,
                'test_accuracy': 0.0,
                'test_f1_weighted': 0.0,
                'val_f1_macro': 0.0,
                'val_accuracy': 0.0,
                'training_duration': 0.0,
                'model_path': str(self.models_path / f"model-config-{config_id}")
            }
            
            # Check if model exists
            model_path = self.models_path / f"model-config-{config_id}"
            if model_path.exists():
                config_data['available'] = True
                
                # Load performance from results_summary.txt
                results_file = model_path / 'results_summary.txt'
                if results_file.exists():
                    performance = self._parse_results_file(results_file)
                    config_data.update(performance)
            
            results[config_id] = config_data
        
        return results
    
    def _parse_results_file(self, results_file: Path) -> Dict:
        """Parse results_summary.txt file to extract performance metrics"""
        performance = {
            'test_f1_macro': 0.0,
            'test_accuracy': 0.0,
            'test_f1_weighted': 0.0,
            'val_f1_macro': 0.0,
            'val_accuracy': 0.0,
            'training_duration': 0.0
        }
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Patterns to extract metrics
            patterns = {
                'test_f1_macro': r'Test F1-Macro:\s*(\d+\.?\d*)',
                'test_accuracy': r'Test Accuracy:\s*(\d+\.?\d*)%?',
                'test_f1_weighted': r'Test F1-Weighted:\s*(\d+\.?\d*)',
                'val_f1_macro': r'Validation F1-Macro:\s*(\d+\.?\d*)',
                'val_accuracy': r'Validation Accuracy:\s*(\d+\.?\d*)%?',
                'training_duration': r'Training Duration:\s*(\d+\.?\d*)\s*minutes?'
            }
            
            for metric, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = float(match.group(1))
                    # Convert percentage to decimal if needed
                    if 'accuracy' in metric and value > 1:
                        value = value / 100
                    performance[metric] = value
        
        except Exception as e:
            print(f"Error parsing {results_file}: {e}")
        
        return performance
    
    def get_best_model(self) -> Tuple[int, Dict]:
        """
        Get the best performing model based on F1-Macro score
        
        Returns:
            Tuple of (config_id, performance_data)
        """
        results = self.load_training_f1_results()
        
        # Filter available models with valid F1-Macro scores
        valid_results = {
            config_id: data for config_id, data in results.items()
            if data['available'] and data['test_f1_macro'] > 0
        }
        
        if not valid_results:
            return None, None
        
        # Find best model by F1-Macro
        best_config_id = max(valid_results.keys(), 
                           key=lambda x: valid_results[x]['test_f1_macro'])
        
        return best_config_id, valid_results[best_config_id]
    
    def get_top_models(self, n: int = 3) -> List[Tuple[int, Dict]]:
        """
        Get top N performing models based on F1-Macro score
        
        Args:
            n: Number of top models to return
            
        Returns:
            List of tuples (config_id, performance_data)
        """
        results = self.load_training_f1_results()
        
        # Filter available models with valid F1-Macro scores
        valid_results = {
            config_id: data for config_id, data in results.items()
            if data['available'] and data['test_f1_macro'] > 0
        }
        
        if not valid_results:
            return []
        
        # Sort by F1-Macro descending
        sorted_results = sorted(valid_results.items(), 
                              key=lambda x: x[1]['test_f1_macro'], 
                              reverse=True)
        
        return sorted_results[:n]
    
    def create_performance_dataframe(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame with all model performance data
        Suitable for display in Streamlit
        
        Returns:
            DataFrame with model performance metrics
        """
        results = self.load_training_f1_results()
        
        # Filter available models
        valid_results = {
            config_id: data for config_id, data in results.items()
            if data['available']
        }
        
        if not valid_results:
            return pd.DataFrame()
        
        # Create DataFrame
        df_data = []
        for config_id, data in valid_results.items():
            row = {
                'Config ID': config_id,
                'Model': f"model-config-{config_id}",
                'F1-Macro': data['test_f1_macro'],
                'Accuracy': data['test_accuracy'],
                'F1-Weighted': data['test_f1_weighted'],
                'Val F1-Macro': data['val_f1_macro'],
                'Val Accuracy': data['val_accuracy'],
                'Training Time (min)': data['training_duration'],
                'Available': data['available']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Sort by F1-Macro descending
        df = df.sort_values('F1-Macro', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        columns = ['Rank', 'Config ID', 'Model', 'F1-Macro', 'Accuracy', 
                  'F1-Weighted', 'Val F1-Macro', 'Val Accuracy', 'Training Time (min)']
        df = df[columns]
        
        return df
    
    def get_performance_summary(self) -> Dict:
        """
        Get overall performance summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        results = self.load_training_f1_results()
        
        # Filter available models with valid scores
        valid_results = [
            data for data in results.values()
            if data['available'] and data['test_f1_macro'] > 0
        ]
        
        if not valid_results:
            return {}
        
        f1_scores = [r['test_f1_macro'] for r in valid_results]
        accuracies = [r['test_accuracy'] for r in valid_results]
        training_times = [r['training_duration'] for r in valid_results if r['training_duration'] > 0]
        
        summary = {
            'total_models': len(valid_results),
            'best_f1_macro': max(f1_scores) if f1_scores else 0,
            'worst_f1_macro': min(f1_scores) if f1_scores else 0,
            'avg_f1_macro': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'worst_accuracy': min(accuracies) if accuracies else 0,
            'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'avg_training_time': sum(training_times) / len(training_times) if training_times else 0,
            'total_training_time': sum(training_times) if training_times else 0
        }
        
        return summary
    
    def generate_model_comparison_report(self) -> str:
        """
        Generate a text report comparing all models
        Suitable for thesis documentation
        
        Returns:
            Formatted text report
        """
        results = self.load_training_f1_results()
        summary = self.get_performance_summary()
        best_config_id, best_model = self.get_best_model()
        top_models = self.get_top_models(5)
        
        report = f"""
=================================================================
INDONESIAN LYRICS EMOTION DETECTION - MODEL COMPARISON REPORT
=================================================================

SUMMARY STATISTICS:
- Total Models Evaluated: {summary.get('total_models', 0)}
- Best F1-Macro Score: {summary.get('best_f1_macro', 0):.4f}
- Average F1-Macro Score: {summary.get('avg_f1_macro', 0):.4f}
- Best Accuracy: {summary.get('best_accuracy', 0)*100:.2f}%
- Average Accuracy: {summary.get('avg_accuracy', 0)*100:.2f}%
- Total Training Time: {summary.get('total_training_time', 0):.1f} minutes

BEST PERFORMING MODEL:
"""
        
        if best_model:
            report += f"""
- Model: model-config-{best_config_id}
- F1-Macro Score: {best_model['test_f1_macro']:.4f}
- Accuracy: {best_model['test_accuracy']*100:.2f}%
- F1-Weighted: {best_model['test_f1_weighted']:.4f}
- Training Time: {best_model['training_duration']:.1f} minutes
"""
        
        report += f"""
TOP 5 MODELS RANKING:
"""
        
        for i, (config_id, model_data) in enumerate(top_models, 1):
            report += f"""
{i}. model-config-{config_id}
   F1-Macro: {model_data['test_f1_macro']:.4f} | Accuracy: {model_data['test_accuracy']*100:.2f}%
"""
        
        report += f"""
=================================================================
Report generated for thesis documentation purposes.
Primary evaluation metric: F1-Macro Score for balanced classification.
=================================================================
"""
        
        return report
    
    def save_performance_data(self, output_file: str = "model_performances.json"):
        """
        Save all performance data to JSON file for external use
        
        Args:
            output_file: Output filename
        """
        results = self.load_training_f1_results()
        summary = self.get_performance_summary()
        
        output_data = {
            'summary': summary,
            'detailed_results': results,
            'evaluation_info': {
                'primary_metric': 'F1-Macro',
                'total_configs': 12,
                'emotion_classes': list(self.emotion_labels.values()),
                'evaluation_purpose': 'Thesis - Indonesian Lyrics Emotion Detection'
            }
        }
        
        output_path = BASE_PATH / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Performance data saved to: {output_path}")

# ========================================
# STANDALONE USAGE FOR TESTING
# ========================================

def main():
    """Main function for standalone testing"""
    print("🔍 F1 Macro Evaluator - Model Performance Analysis")
    print("=" * 60)
    
    evaluator = F1MacroAppEvaluator()
    
    # Load and display results
    print("\n📊 Loading model performance data...")
    results = evaluator.load_training_f1_results()
    
    # Display available models
    available_models = [config_id for config_id, data in results.items() if data['available']]
    print(f"✅ Found {len(available_models)} available models: {available_models}")
    
    # Get best model
    best_config_id, best_model = evaluator.get_best_model()
    if best_model:
        print(f"\n🏆 Best Model: model-config-{best_config_id}")
        print(f"   F1-Macro: {best_model['test_f1_macro']:.4f}")
        print(f"   Accuracy: {best_model['test_accuracy']*100:.2f}%")
    
    # Display performance summary
    summary = evaluator.get_performance_summary()
    if summary:
        print(f"\n📈 Performance Summary:")
        print(f"   Average F1-Macro: {summary['avg_f1_macro']:.4f}")
        print(f"   Average Accuracy: {summary['avg_accuracy']*100:.2f}%")
        print(f"   Total Training Time: {summary['total_training_time']:.1f} minutes")
    
    # Generate and save report
    print(f"\n📄 Generating comparison report...")
    report = evaluator.generate_model_comparison_report()
    
    # Save to file
    report_path = BASE_PATH / "model_comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_path}")
    
    # Save performance data
    evaluator.save_performance_data()
    
    print(f"\n✨ Analysis complete! Ready for Streamlit app.")

if __name__ == "__main__":
    main()
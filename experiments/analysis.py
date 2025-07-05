import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

class NEHAnalyzer:
    """Analyzer for Noise Equivalence Hypothesis experiments"""
    
    def __init__(self, results_base_dir):
        self.results_dir = Path(results_base_dir)
        
    def load_experiment_data(self, alg_name, run_id="run_001"):
        """Load all data for a specific algorithm run"""
        run_dir = self.results_dir / alg_name / run_id
        
        # Load case-level edit results
        case_files = list(run_dir.glob("*_edits-case_*.json"))
        edit_data = []
        
        for case_file in case_files:
            with open(case_file, 'r') as f:
                data = json.load(f)
                
            # Extract key metrics
            case_metrics = {
                'case_id': data['case_id'],
                'edit_num': data['num_edits'],
                'alg_name': alg_name,
                
                # Edit success metrics
                'rewrite_success': np.mean(data['post']['rewrite_prompts_correct']),
                'paraphrase_success': np.mean(data['post']['paraphrase_prompts_correct']),
                'neighborhood_success': np.mean(data['post']['neighborhood_prompts_correct']),
                
                # Probability differences (if available)
                'rewrite_prob_diff': self._extract_prob_diff(data['post'], 'rewrite_prompts_probs'),
                'paraphrase_prob_diff': self._extract_prob_diff(data['post'], 'paraphrase_prompts_probs'),
                'neighborhood_prob_diff': self._extract_prob_diff(data['post'], 'neighborhood_prompts_probs'),
            }
            
            edit_data.append(case_metrics)
        
        # Load GLUE downstream task results
        glue_dir = run_dir / "glue_eval"
        glue_data = []
        
        if glue_dir.exists():
            glue_files = list(glue_dir.glob("*_glue.json"))
            
            for glue_file in glue_files:
                with open(glue_file, 'r') as f:
                    data = json.load(f)
                
                glue_metrics = {
                    'alg_name': alg_name,
                    'edit_num': data.get('edit_num', -1),
                }
                
                # Extract downstream task performance
                for task in ['sst', 'nli', 'mmlu']:
                    if task in data:
                        glue_metrics[f'{task}_accuracy'] = data[task].get('correct', 0) / 100.0
                        glue_metrics[f'{task}_f1'] = data[task].get('f1', 0)
                
                # Extract distance metrics
                if 'distance_from_original' in data:
                    for layer, distance in data['distance_from_original'].items():
                        glue_metrics[f'distance_layer_{layer}'] = distance
                
                glue_data.append(glue_metrics)
        
        return pd.DataFrame(edit_data), pd.DataFrame(glue_data)
    
    def _extract_prob_diff(self, post_data, key):
        """Extract probability difference from post data"""
        if key not in post_data or not post_data[key]:
            return np.nan
        
        probs = post_data[key]
        if isinstance(probs, list) and len(probs) > 0:
            if isinstance(probs[0], dict):
                # Extract target_new - target_true probability difference
                diffs = []
                for p in probs:
                    try:
                        # Handle different possible data structures
                        if 'target_new_prob' in p and 'target_true_prob' in p:
                            # New format with explicit prob fields
                            new_prob = float(p['target_new_prob'])
                            true_prob = float(p['target_true_prob'])
                            diffs.append(new_prob - true_prob)
                        elif 'target_new' in p and 'target_true' in p:
                            # Try to convert to float if they're numeric strings
                            new_val = p['target_new']
                            true_val = p['target_true']
                            
                            if isinstance(new_val, (int, float)) and isinstance(true_val, (int, float)):
                                diffs.append(float(new_val) - float(true_val))
                            elif isinstance(new_val, str) and isinstance(true_val, str):
                                try:
                                    diffs.append(float(new_val) - float(true_val))
                                except ValueError:
                                    # If they're text strings, skip this entry
                                    continue
                    except (ValueError, TypeError, KeyError):
                        continue
                        
                return np.mean(diffs) if diffs else np.nan
        return np.nan
    
    def compare_algorithms(self, alg1_name, alg2_name, run1_id="run_001", run2_id=None):
        """Compare two algorithms (e.g., ROME vs ROME_NOISE)"""
        
        # Use same run_id for both if run2_id not specified
        if run2_id is None:
            run2_id = run1_id
        
        # Load data for both algorithms
        edit1, glue1 = self.load_experiment_data(alg1_name, run1_id)
        edit2, glue2 = self.load_experiment_data(alg2_name, run2_id)
        
        print(f"\n{'='*70}")
        print(f"{alg1_name} ({run1_id}) vs {alg2_name} ({run2_id}) Comparison")
        print(f"{'='*70}")
        print(f"Data loaded: {len(edit1)} {alg1_name} cases, {len(edit2)} {alg2_name} cases")
        print(f"GLUE evals: {len(glue1)} {alg1_name} evals, {len(glue2)} {alg2_name} evals")
        
        # Compare edit-level metrics
        print(f"\n📝 EDIT SUCCESS METRICS:")
        edit_metrics = ['rewrite_success', 'paraphrase_success', 'neighborhood_success']
        self._compare_metrics(edit1, edit2, edit_metrics, alg1_name, alg2_name)
        
        # Compare downstream task performance (the key NEH metric!)
        print(f"\n🎯 DOWNSTREAM TASK PERFORMANCE (Key NEH Metric!):")
        downstream_metrics = [col for col in glue1.columns if any(task in col for task in ['sst', 'nli', 'mmlu'])]
        if not downstream_metrics:
            print("❌ No downstream task metrics found! Check GLUE evaluation.")
            print(f"Available GLUE columns: {list(glue1.columns)}")
        else:
            self._compare_metrics(glue1, glue2, downstream_metrics, alg1_name, alg2_name)
        
        # Compare representation distances
        print(f"\n📏 REPRESENTATION DISTANCES:")
        distance_metrics = [col for col in glue1.columns if col.startswith('distance_')]
        if distance_metrics:
            self._compare_metrics(glue1, glue2, distance_metrics, alg1_name, alg2_name)
        else:
            print("❌ No distance metrics found!")
        
        return edit1, edit2, glue1, glue2
    
    def _compare_metrics(self, df1, df2, metrics, name1, name2):
        """Compare specific metrics between two dataframes"""
        
        for metric in metrics:
            if metric not in df1.columns or metric not in df2.columns:
                continue
                
            vals1 = df1[metric].dropna()
            vals2 = df2[metric].dropna()
            
            if len(vals1) < 2 or len(vals2) < 2:
                continue
            
            # Check if values are essentially identical (within floating point precision)
            if abs(vals1.mean() - vals2.mean()) < 1e-10:
                print(f"{metric:30s}: {name1}={vals1.mean():7.3f} {name2}={vals2.mean():7.3f} "
                    f"Δ={0.000:+7.3f} p=N/A     IDENTICAL | 🟢 NEH SUPPORTED")
                continue
            
            # Statistical test
            _, p_val = ttest_ind(vals1, vals2, equal_var=False)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(vals1)-1)*vals1.std()**2 + (len(vals2)-1)*vals2.std()**2) / (len(vals1)+len(vals2)-2))
            cohens_d = (vals2.mean() - vals1.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Significance markers
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            # NEH interpretation
            neh_support = "🟢 NEH SUPPORTED" if p_val > 0.05 else "🔴 SIGNIFICANTLY DIFFERENT"
            
            diff = vals2.mean() - vals1.mean()
            print(f"{metric:30s}: {name1}={vals1.mean():7.3f} {name2}={vals2.mean():7.3f} "
                  f"Δ={diff:+7.3f} p={p_val:6.4f}{sig:3s} d={cohens_d:+5.2f} | {neh_support}")
    
    def plot_downstream_degradation(self, alg1_name, alg2_name, run1_id="run_001", run2_id=None):
        """Plot downstream task degradation over sequential edits"""
        
        if run2_id is None:
            run2_id = run1_id
        
        edit1, glue1 = self.load_experiment_data(alg1_name, run1_id)
        edit2, glue2 = self.load_experiment_data(alg2_name, run2_id)
        
        # Focus on downstream tasks
        tasks = ['sst_accuracy', 'nli_accuracy', 'mmlu_accuracy']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, task in enumerate(tasks):
            if task in glue1.columns and task in glue2.columns:
                # Sort by edit number
                g1_sorted = glue1.sort_values('edit_num')
                g2_sorted = glue2.sort_values('edit_num')
                
                axes[i].plot(g1_sorted['edit_num'], g1_sorted[task], 'o-', label=f"{alg1_name} ({run1_id})", linewidth=2)
                axes[i].plot(g2_sorted['edit_num'], g2_sorted[task], 's--', label=f"{alg2_name} ({run2_id})", linewidth=2)
                
                axes[i].set_xlabel('Number of Edits')
                axes[i].set_ylabel(f'{task.replace("_", " ").title()}')
                axes[i].set_title(f'{task.split("_")[0].upper()} Performance Degradation')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Downstream Task Degradation: {alg1_name} vs {alg2_name}', y=1.02, fontsize=14)
        return fig
    
    def neh_summary_report(self, alg1_name="ROME", alg2_name="ROME_NOISE", run1_id="run_001", run2_id=None):
        """Generate a comprehensive NEH analysis report"""
        
        if run2_id is None:
            run2_id = run1_id
        
        print(f"\n🔬 NOISE EQUIVALENCE HYPOTHESIS ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"Comparing: {alg1_name} ({run1_id}) vs {alg2_name} ({run2_id})")
        print(f"Hypothesis: Parameter changes from {alg2_name} are statistically")
        print(f"           indistinguishable from {alg1_name} on downstream tasks")
        print(f"{'='*80}")
        
        # Run the comparison
        edit1, edit2, glue1, glue2 = self.compare_algorithms(alg1_name, alg2_name, run1_id, run2_id)
        
        # Count supporting evidence
        downstream_metrics = [col for col in glue1.columns if any(task in col for task in ['sst', 'nli', 'mmlu'])]
        non_significant_count = 0
        total_count = 0
        
        for metric in downstream_metrics:
            if metric in glue1.columns and metric in glue2.columns:
                vals1 = glue1[metric].dropna()
                vals2 = glue2[metric].dropna()
                if len(vals1) >= 2 and len(vals2) >= 2:
                    _, p_val = ttest_ind(vals1, vals2, equal_var=False)
                    if p_val > 0.05:
                        non_significant_count += 1
                    total_count += 1
        
        neh_support_pct = (non_significant_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n📊 NEH VERDICT:")
        print(f"   Downstream metrics showing statistical equivalence: {non_significant_count}/{total_count} ({neh_support_pct:.1f}%)")
        
        if neh_support_pct > 70:
            print(f"   🟢 STRONG SUPPORT for NEH - Noise injection effects are largely equivalent to targeted editing")
        elif neh_support_pct > 40:
            print(f"   🟡 MIXED EVIDENCE for NEH - Some metrics show equivalence, others differ significantly")
        else:
            print(f"   🔴 WEAK SUPPORT for NEH - Targeted editing and noise show significant differences")
        
        return edit1, edit2, glue1, glue2


# Usage example:
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = NEHAnalyzer('/mnt/sda/hoyeon/unified-model-editing/results/')
    
    # Run comprehensive NEH analysis with different run IDs
    analyzer.neh_summary_report("ROME", "ROME_NOISE", "run_009", "run_013")
    
    # Generate degradation plots
    fig = analyzer.plot_downstream_degradation("ROME", "ROME_NOISE", "run_009", "run_013")
    plt.savefig('degradation_plot.png', dpi=300, bbox_inches='tight')
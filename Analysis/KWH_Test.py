import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load CSV
df = pd.read_csv('c:/Users/vagfl/Python Uni/Dissertation/results_0.2.csv')

# Kruskal-Wallis Test
for model in df['Model'].unique():
    model_data = df[df['Model'] == model]
    print(f'Model: {model}')
    
    # Kruskal-Wallis test
    h_stat, p_value = stats.kruskal(*[model_data[model_data['Clutter_Level'] == level]['F1_Score'] for level in model_data['Clutter_Level'].unique()])
    print(f'H-statistic: {h_stat:.3f}')
    print(f'p-value: {p_value:.4f}')
    
    if p_value < 0.05:
        print(' > Statistically significant difference in F1 scores across clutter levels.')
        
        # Dunn's post-hoc test
        posthoc = pairwise_tukeyhsd(model_data['F1_Score'], model_data['Clutter_Level'])
        print(posthoc)

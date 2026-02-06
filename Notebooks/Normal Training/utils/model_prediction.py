from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate
import matplotlib.pyplot as plt

def make_predictions(model, data, y_mean, y_std, model_name,):
    
    predictions={}
    true={}
    for temp in ['25', '35', '45', 'complete']:
        true[temp] = data[temp]['y']
        predictions[temp] = model.predict([data[temp]['X_ic'], data[temp]['X_ctx']]).flatten()

    metrics = get_metrics(predictions, true, y_mean, y_std)
    scatter_plot(predictions, true, y_std, y_mean, model_name)

    return predictions, metrics

def get_metrics(predictions, true, y_mean, y_std):
    metrics = {}

    for d in ['normalized', 'original']:
        metrics[d] = {}
        for t in ['25', '35', '45', 'complete']:
            metrics[d][t] = {}
            if(d == 'normalized'):
                metrics[d][t]['MAE'] = mean_absolute_error(predictions[t],true[t])
                metrics[d][t]['MSE'] = mean_squared_error(predictions[t], true[t])
                metrics[d][t]['R2'] = r2_score(true[t], predictions[t])
            
            elif(d=='original'):
                # Denormalize predictions and true values
                pred_original = predictions[t] * y_std + y_mean
                true_original = true[t] * y_std + y_mean
                
                metrics[d][t]['MAE'] = mean_absolute_error(pred_original, true_original)
                metrics[d][t]['MSE'] = mean_squared_error(pred_original, true_original)
                metrics[d][t]['R2'] = r2_score(true_original, pred_original)
    
    return metrics


def format_metrics(metrics):
    """
    Format and print metrics in a table format using tabulate.
    
    Args:
        metrics: Dictionary containing metrics organized by normalization type and temperature
    """
    for metric_type in ['normalized', 'original']:
        print(f"\n{'='*60}")
        print(f"Metrics - {metric_type.upper()}")
        print(f"{'='*60}\n")
        
        table_data = []
        headers = ['Temperature', 'MAE', 'MSE', 'R2 Score']
        
        for temp in ['25', '35', '45', 'complete']:
            if temp in metrics[metric_type]:
                row = [
                    temp,
                    f"{metrics[metric_type][temp]['MAE']:.6f}",
                    f"{metrics[metric_type][temp]['MSE']:.6f}",
                    f"{metrics[metric_type][temp]['R2']:.6f}"
                ]
                table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()

def scatter_plot(predictions, true, y_std, y_mean, model_name):

    plt.figure(figsize=(22,6))
    i = 1
    for key in ['25', '35', '45']:

        y_true = true[key] * y_std + y_mean
        y_pred = predictions[key] * y_std + y_mean
        plt.subplot(1,3,i)
        plt.scatter(y_true, y_pred, alpha=0.6, s=15)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.title(model_name+" SOH Predictions @ "+key+" degC", fontsize=14)
        plt.xlabel("True SOH", fontsize=12)
        plt.ylabel("Predicted SOH", fontsize=12)
        i+=1
        plt.grid(False)





from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from tabulate import tabulate


def get_metrics(predictions, true, y_mean, y_std):
    metrics = {}

    for d in ['normalized', 'original']:
        metrics[d] = {}
        for t in ['25', '35', '45', 'complete']:
            metrics[d][t] = {}
            if d == 'normalized':
                metrics[d][t]['MAE'] = mean_absolute_error(true[t], predictions[t])
                metrics[d][t]['MSE'] = mean_squared_error(true[t], predictions[t])
                metrics[d][t]['RMSE'] = root_mean_squared_error(true[t], predictions[t])
                metrics[d][t]['R2'] = r2_score(true[t], predictions[t])
            elif d == 'original':
                pred_original = predictions[t] * y_std + y_mean
                true_original = true[t] * y_std + y_mean

                metrics[d][t]['MAE'] = mean_absolute_error(true_original, pred_original)
                metrics[d][t]['MSE'] = mean_squared_error(true_original, pred_original)
                metrics[d][t]['RMSE'] = root_mean_squared_error(true[t], predictions[t])
                metrics[d][t]['R2'] = r2_score(true_original, pred_original)

    return metrics


def format_metrics(metrics, model_name):
    '''
    Format and print metrics in a table format using tabulate.

    Args:
        metrics: Dictionary containing metrics organized by normalization type and temperature
    '''
    for metric_type in ['normalized', 'original']:
        print(f"\n{'='*60}")
        print(model_name+f" Metrics - {metric_type.upper()}")
        print(f"{'='*60}\n")

        table_data = []
        headers = ['Temperature', 'MAE', 'MSE', 'RMSE', 'R2 Score']

        for temp in ['25', '35', '45', 'complete']:
            if temp in metrics[metric_type]:
                row = [
                    temp,
                    f"{metrics[metric_type][temp]['MAE']:.6f}",
                    f"{metrics[metric_type][temp]['MSE']:.6f}",
                    f"{metrics[metric_type][temp]['RMSE']:.6f}",
                    f"{metrics[metric_type][temp]['R2']:.6f}",
                ]
                table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()

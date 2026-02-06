import matplotlib.pyplot as plt


def scatter_plot(predictions, true, y_std, y_mean, model_name):
    plt.figure(figsize=(22, 6))
    i = 1
    for key in ['25', '35', '45']:
        y_true = true[key] * y_std + y_mean
        y_pred = predictions[key] * y_std + y_mean
        plt.subplot(1, 3, i)
        plt.scatter(y_true, y_pred, alpha=0.6, s=15)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.title(model_name + ' SOH Predictions @ ' + key + ' °C', fontsize=14)
        plt.xlabel('True SOH', fontsize=12)
        plt.ylabel('Predicted SOH', fontsize=12)
        i += 1
        plt.grid(False)

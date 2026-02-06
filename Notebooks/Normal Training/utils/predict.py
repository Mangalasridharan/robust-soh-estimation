from utils.metrics import get_metrics
from utils.plots import scatter_plot


def make_predictions(model, data, y_mean, y_std):
    predictions = {}
    true = {}

    for temp in ['25', '35', '45', 'complete']:
        true[temp] = data[temp]['y']
        predictions[temp] = model.predict([data[temp]['X_ic'], data[temp]['X_ctx']]).flatten()

    metrics = get_metrics(predictions, true, y_mean, y_std)
    
    return predictions,true, metrics

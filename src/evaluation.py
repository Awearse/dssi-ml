import logging
from sklearn.metrics import confusion_matrix
from src.config import appconfig
from src import model_registry

logging.basicConfig(level=logging.INFO)

fdr_max = float(appconfig['Evaluation']['fdr'])
recall_min = float(appconfig['Evaluation']['recall'])
_model_name = appconfig['Model']['name']

def __get_classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    return TP, TN, FP, FN

def get_fdr(y_true, y_pred):
    """
    Calculate False Discovery Rate (FDR).
        Parameters:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
        Returns:
            float: FDR value
    """
    TP, TN, FP, FN = __get_classification_metrics(y_true, y_pred)
    if (TP + FP) == 0:
        return 0.0
    return round((FP / (TP + FP)), 2)

def get_recall(y_true, y_pred):
    """
    Calculate Recall.
        Parameters:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
        Returns:
            float: Recall value
    """
    TP, TN, FP, FN = __get_classification_metrics(y_true, y_pred)
    if (TP + FN) == 0:
        return 0.0
    return round((TP / (TP + FN)), 2)

def get_eval_metrics(y_true, y_pred):
    """
    Return model evaluation metrics.
        Parameters:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
        Returns:
            dict: Dictionary containing evaluation metrics
    """
    results = {
        'fdr': get_fdr(y_true, y_pred),
        'recall': get_recall(y_true, y_pred),
    }
    return results

def run(y_true, y_pred):
    """ Main script to evaluate model performance based on FDR and Recall.
        Parameters:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
        Returns:
            bool: True if evaluation passes, False otherwise
    """
    logging.info('Evaluating model...')
    fdr = get_fdr(y_true, y_pred)
    recall = get_recall(y_true, y_pred)

    if fdr > fdr_max or recall < recall_min:
        logging.warning(f"Model evaluation failed config thresholds: FDR={fdr:.2f} (max {fdr_max:.2f}), Recall={recall:.2f} (min {recall_min:.2f})")
        return False

    current_metadata = model_registry.get_metadata(_model_name)
    if current_metadata is not None:
        current_metrics = current_metadata['metrics']
        if fdr > current_metrics['fdr'] or recall < current_metrics['recall']:
            logging.warning(
                f"Model evaluation failed vs current model v{current_metadata['version']}: "
                f"FDR={fdr:.2f} vs {current_metrics['fdr']:.2f}, "
                f"Recall={recall:.2f} vs {current_metrics['recall']:.2f}"
            )
            return False

    logging.info('Model evaluation passed.')
    return True

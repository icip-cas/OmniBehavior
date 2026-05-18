import numpy as np
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def calculate_accuracy(y_true: List[int], y_pred_labels: List[int], field_name: str = "") -> float:
    try:
        correct = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == pred)
        accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0
        return accuracy
    except Exception as e:
        print(f"Error computing Accuracy ({field_name}): {e}")
        return np.nan


def calculate_logloss(y_true: List[int], y_pred: List[float]) -> float:
    try:
        # clip to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return log_loss(y_true, y_pred_clipped, labels=[0, 1])
    except Exception as e:
        print(f"Error computing LogLoss: {e}")
        return np.nan


def calculate_precision(y_true: List[int], y_pred_labels: List[int]) -> float:
    try:
        tp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 1)
        fp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 0 and pred == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return precision
    except Exception as e:
        print(f"Error computing Precision: {e}")
        return np.nan


def calculate_recall(y_true: List[int], y_pred_labels: List[int]) -> float:
    try:
        tp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 1)
        fn = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return recall
    except Exception as e:
        print(f"Error computing Recall: {e}")
        return np.nan


def calculate_f1(y_true: List[int], y_pred_labels: List[int]) -> float:
    try:
        precision = calculate_precision(y_true, y_pred_labels)
        recall = calculate_recall(y_true, y_pred_labels)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return f1
    except Exception as e:
        print(f"Error computing F1: {e}")
        return np.nan


def calculate_ece(y_true: List[int], y_pred: List[float], n_bins: int = 10) -> float:
    """Expected Calibration Error — measures probability calibration quality."""
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_confidence = np.mean(y_pred[in_bin])
                bin_accuracy = np.mean(y_true[in_bin])
                bin_weight = np.sum(in_bin) / len(y_true)
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    except Exception as e:
        print(f"Error computing ECE: {e}")
        return np.nan


def calculate_auc(y_true: List[int], y_pred: List[float]) -> float:
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(np.unique(y_true)) < 2:
            print(f"Warning: only one class in sample, AUC undefined")
            return np.nan

        return roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(f"Error computing AUC: {e}")
        return np.nan


def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
    try:
        return mean_absolute_error(y_true, y_pred)
    except Exception as e:
        print(f"Error computing MAE: {e}")
        return np.nan


def calculate_mse(y_true: List[float], y_pred: List[float]) -> float:
    try:
        return mean_squared_error(y_true, y_pred)
    except Exception as e:
        print(f"Error computing MSE: {e}")
        return np.nan


def calculate_rmse(y_true: List[float], y_pred: List[float]) -> float:
    try:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    except Exception as e:
        print(f"Error computing RMSE: {e}")
        return np.nan


def calculate_nmae(y_true: List[float], y_pred: List[float], 
                   normalizers: List[float]) -> float:
    """
    Normalized MAE: mean of clip(|y_true-y_pred|, 0, normalizer)/normalizer * 100.
    Samples with normalizer <= 0 or None are skipped.
    """
    try:
        if len(y_true) != len(y_pred) or len(y_true) != len(normalizers):
            print(f"Error: mismatched input lengths for NMAE (y_true={len(y_true)}, y_pred={len(y_pred)}, normalizers={len(normalizers)})")
            return np.nan
        
        normalized_errors = []
        skipped_count = 0
        for true_val, pred_val, norm in zip(y_true, y_pred, normalizers):
            if norm is None or norm <= 0:
                skipped_count += 1
                continue
            abs_error = abs(true_val - pred_val)
            clipped_error = min(max(abs_error, 0), norm)
            normalized_error = clipped_error / norm
            normalized_errors.append(normalized_error)

        if not normalized_errors:
            print(f"Warning: no valid normalizers, NMAE unavailable ({skipped_count} samples skipped)")
            return np.nan

        nmae = np.mean(normalized_errors) * 100
        return nmae
    except Exception as e:
        print(f"Error computing NMAE: {e}")
        return np.nan


def calculate_all_binary_metrics(
    y_true: List[int],
    y_pred_labels: List[int],
    y_pred_probs: List[float],
    n_bins: int = 10,
    field_name: str = ""
) -> Dict[str, float]:
    """Compute all binary classification metrics; includes TP/FP/FN for Micro F1 aggregation."""
    tp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 0)
    
    return {
        "Accuracy": calculate_accuracy(y_true, y_pred_labels, field_name=field_name),
        "Precision": calculate_precision(y_true, y_pred_labels),
        "Recall": calculate_recall(y_true, y_pred_labels),
        "F1": calculate_f1(y_true, y_pred_labels),
        "LogLoss": calculate_logloss(y_true, y_pred_probs),
        "AUC": calculate_auc(y_true, y_pred_probs),
        "ECE": calculate_ece(y_true, y_pred_probs, n_bins),
        "sample_count": len(y_true),
        "positive_rate": np.mean(y_true),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }



def calculate_micro_macro_f1(binary_metrics: Dict[str, Dict]) -> Dict[str, float]:
    """
    Compute Micro and Macro F1 across fields.
    NaN F1 values are treated as 0.0 (strict). Excludes video_downloaded and ad_converted.
    """
    if not binary_metrics:
        return {
            "Micro_F1": 0.0, "Macro_F1": 0.0,
            "Micro_Precision": 0.0, "Micro_Recall": 0.0,
            "Total_TP": 0, "Total_FP": 0, "Total_FN": 0,
        }
    
    # exclude these fields from aggregate metrics
    EXCLUDED_FIELDS = {"video_downloaded", "ad_converted"}
    filtered_metrics = {k: v for k, v in binary_metrics.items() if k not in EXCLUDED_FIELDS}
    
    if not filtered_metrics:
        return {
            "Micro_F1": 0.0, "Macro_F1": 0.0,
            "Micro_Precision": 0.0, "Micro_Recall": 0.0,
            "Total_TP": 0, "Total_FP": 0, "Total_FN": 0,
        }
    
    total_tp = sum(m.get("TP", 0) for m in filtered_metrics.values())
    total_fp = sum(m.get("FP", 0) for m in filtered_metrics.values())
    total_fn = sum(m.get("FN", 0) for m in filtered_metrics.values())

    denom_prec = total_tp + total_fp
    denom_rec = total_tp + total_fn
    
    micro_precision = total_tp / denom_prec if denom_prec > 0 else 0.0
    micro_recall = total_tp / denom_rec if denom_rec > 0 else 0.0
    
    denom_f1 = micro_precision + micro_recall
    micro_f1 = (2 * micro_precision * micro_recall) / denom_f1 if denom_f1 > 0 else 0.0

    raw_f1_values = []
    for m in filtered_metrics.values():
        val = m.get("F1", 0.0)
        if val is None:
            val = 0.0
        raw_f1_values.append(val)
    
    f1_array = np.array(raw_f1_values, dtype=float)
    f1_array = np.nan_to_num(f1_array, nan=0.0)  # Note: NaN treated as 0.0 — stricter than ignoring
    macro_f1 = np.mean(f1_array) if len(f1_array) > 0 else 0.0

    return {
        "Micro_F1": micro_f1,
        "Macro_F1": macro_f1,
        "Micro_Precision": micro_precision,
        "Micro_Recall": micro_recall,
        "Total_TP": total_tp,
        "Total_FP": total_fp,
        "Total_FN": total_fn,
    }

def calculate_relative_accuracy(y_true: List[float], y_pred: List[float]) -> float:
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        scores = []
        for t, p in zip(y_true, y_pred):
            max_val = max(p, t)
            if max_val == 0:
                scores.append(100.0)  # Both zero — treat as perfect
            else:
                score = (1 - (abs(t - p) / max_val)) * 100
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        print(f"Error computing Relative Accuracy: {e}")
        return np.nan


def calculate_symmetry_consistency(y_true: List[float], y_pred: List[float]) -> float:
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        scores = []
        for t, p in zip(y_true, y_pred):
            sum_val = p + t
            if sum_val == 0:
                scores.append(100.0)  # Both zero — treat as perfect
            else:
                score = (1 - (abs(t - p) / sum_val)) * 100
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        print(f"Error computing Symmetry Consistency: {e}")
        return np.nan


def calculate_all_continuous_metrics(
    y_true: List[float],
    y_pred: List[float],
    normalizers: List[float] = None
) -> Dict[str, float]:
    """Compute all continuous metrics; also computes NMAE if normalizers are provided."""
    result = {
        "MAE": calculate_mae(y_true, y_pred),
        "MSE": calculate_mse(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "relative_accuracy": calculate_relative_accuracy(y_true, y_pred),
        "symmetry_consistency": calculate_symmetry_consistency(y_true, y_pred),
        "sample_count": len(y_true),
        "mean_true": np.mean(y_true),
        "mean_pred": np.mean(y_pred),
    }
    
    if normalizers is not None and len(normalizers) > 0:
        nmae = calculate_nmae(y_true, y_pred, normalizers)
        result["NMAE"] = nmae
    
    return result



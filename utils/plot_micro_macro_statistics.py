# Correcting the compute_macro_metrics_with_youden function to use label indices instead of names
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_macro_metrics_all_with_youden(df, prediction_suffix):
    ROCs = []
    PRs = []
    Sensitivities = []
    Specificities = []

    for column in df.columns:
        if prediction_suffix in column:
            y_true = df[column.replace(prediction_suffix, "")]
            y_score = df[column]

            if len(np.unique(y_true)) > 1 and len(np.unique(y_score)) > 1:
                ROC = roc_auc_score(y_true, y_score)
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                PR = auc(recall, precision)

                # Calculate ROC curve and find optimal threshold using Youden Index
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                youden_index = tpr - fpr
                optimal_idx = np.argmax(youden_index)
                optimal_threshold = thresholds[optimal_idx]
                sensitivity = tpr[optimal_idx]
                specificity = 1 - fpr[optimal_idx]

                ROCs.append(ROC)
                PRs.append(PR)
                Sensitivities.append(sensitivity)
                Specificities.append(specificity)

    ROC_macro_avg = np.mean(ROCs) if ROCs else None
    PR_macro_avg = np.mean(PRs) if PRs else None
    Sensitivity_macro_avg = np.mean(Sensitivities) if Sensitivities else None
    Specificity_macro_avg = np.mean(Specificities) if Specificities else None

    return ROC_macro_avg, PR_macro_avg, Sensitivity_macro_avg, Specificity_macro_avg


def compute_micro_metrics_all_with_youden(df, prediction_suffix):
    y_trues = []
    y_scores = []

    for column in df.columns:
        if prediction_suffix in column:
            y_trues.append(df[column.replace(prediction_suffix, "")])
            y_scores.append(df[column])

    y_true_micro = np.concatenate(y_trues)
    y_score_micro = np.concatenate(y_scores)

    ROC_micro = roc_auc_score(y_true_micro, y_score_micro)
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_micro, y_score_micro)
    PR_micro = auc(recall_micro, precision_micro)

    # Calculate ROC curve and find optimal threshold using Youden Index
    fpr, tpr, thresholds = roc_curve(y_true_micro, y_score_micro)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    sensitivity_micro = tpr[optimal_idx]
    specificity_micro = 1 - fpr[optimal_idx]

    return ROC_micro, PR_micro, sensitivity_micro, specificity_micro


def compute_micro_sensitivity_specificity(y_true, y_scores):
    # Flatten the arrays for micro level calculation
    y_true_flat = y_true.ravel()
    y_scores_flat = y_scores.ravel()

    # Calculate optimal threshold using Youden Index
    fpr, tpr, thresholds = roc_curve(y_true_flat, y_scores_flat)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate confusion matrix at optimal threshold
    y_pred_optimal = (y_scores_flat >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_optimal).ravel()

    # Calculate sensitivity and specificity
    sensitivity_micro = tp / (tp + fn)
    specificity_micro = tn / (tn + fp)

    return sensitivity_micro, specificity_micro


def compute_macro_metrics_with_youden(y_true, y_scores):
    ROCs_macro = []
    PRs_macro = []
    Sensitivities_macro = []
    Specificities_macro = []

    for i in range(y_scores.shape[1]):
        if len(np.unique(y_true[:, i])) > 1 and len(np.unique(y_scores[:, i])) > 1:
            ROC_macro = roc_auc_score(y_true[:, i], y_scores[:, i], average=None)
            precision_macro, recall_macro, _ = precision_recall_curve(
                y_true[:, i], y_scores[:, i]
            )
            PR_macro = auc(recall_macro, precision_macro)

            # Calculate ROC curve and find optimal threshold using Youden Index
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_scores[:, i])
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
            sensitivity = tpr[optimal_idx]
            specificity = 1 - fpr[optimal_idx]

            ROCs_macro.append(ROC_macro)
            PRs_macro.append(PR_macro)
            Sensitivities_macro.append(sensitivity)
            Specificities_macro.append(specificity)

    ROC_macro_avg = np.mean(ROCs_macro) if ROCs_macro else None
    PR_macro_avg = np.mean(PRs_macro) if PRs_macro else None
    Sensitivity_macro_avg = np.mean(Sensitivities_macro) if Sensitivities_macro else None
    Specificity_macro_avg = np.mean(Specificities_macro) if Specificities_macro else None

    return ROC_macro_avg, PR_macro_avg, Sensitivity_macro_avg, Specificity_macro_avg


def compute_metrics_for_categories(df, categories):
    metrics_results = {}

    for category, labels in categories.items():
        metrics_results[category] = {"CARDIOLOGIST": {}, "MUSE": {}}

        # Prepare true labels and predicted scores for both CARDIOLOGIST and MUSE
        y_true = df[labels].values
        y_scores_cardiologist = df[[f"{label}_CARDIOLOGIST" for label in labels]].values
        y_scores_muse = df[[f"{label}_MUSE" for label in labels]].values

        # Compute micro metrics for CARDIOLOGIST
        if y_scores_cardiologist.size > 0:
            ROC_micro_cardiologist = roc_auc_score(
                y_true.ravel(), y_scores_cardiologist.ravel(), average="micro"
            )
            precision_micro_cardiologist, recall_micro_cardiologist, _ = precision_recall_curve(
                y_true.ravel(), y_scores_cardiologist.ravel()
            )
            PR_micro_cardiologist = auc(recall_micro_cardiologist, precision_micro_cardiologist)
            (
                Sensitivity_micro_cardiologist,
                Specificity_micro_cardiologist,
            ) = compute_micro_sensitivity_specificity(y_true, y_scores_cardiologist)

            metrics_results[category]["CARDIOLOGIST"]["micro_ROC"] = ROC_micro_cardiologist
            metrics_results[category]["CARDIOLOGIST"]["micro_PR"] = PR_micro_cardiologist
            metrics_results[category]["CARDIOLOGIST"][
                "micro_Sensitivity"
            ] = Sensitivity_micro_cardiologist
            metrics_results[category]["CARDIOLOGIST"][
                "micro_Specificity"
            ] = Specificity_micro_cardiologist

            # Compute macro metrics for CARDIOLOGIST
            (
                ROC_macro_cardiologist,
                PR_macro_cardiologist,
                Sensitivity_macro_cardiologist,
                Specificity_macro_cardiologist,
            ) = compute_macro_metrics_with_youden(y_true, y_scores_cardiologist)
            metrics_results[category]["CARDIOLOGIST"]["macro_ROC"] = ROC_macro_cardiologist
            metrics_results[category]["CARDIOLOGIST"]["macro_PR"] = PR_macro_cardiologist
            metrics_results[category]["CARDIOLOGIST"][
                "macro_Sensitivity"
            ] = Sensitivity_macro_cardiologist
            metrics_results[category]["CARDIOLOGIST"][
                "macro_Specificity"
            ] = Specificity_macro_cardiologist

        if y_scores_muse.size > 0:
            ROC_micro_muse = roc_auc_score(y_true.ravel(), y_scores_muse.ravel(), average="micro")
            precision_micro_muse, recall_micro_muse, _ = precision_recall_curve(
                y_true.ravel(), y_scores_muse.ravel()
            )
            PR_micro_muse = auc(recall_micro_muse, precision_micro_muse)
            (
                Sensitivity_micro_muse,
                Specificity_micro_muse,
            ) = compute_micro_sensitivity_specificity(y_true, y_scores_muse)

            metrics_results[category]["MUSE"]["micro_ROC"] = ROC_micro_muse
            metrics_results[category]["MUSE"]["micro_PR"] = PR_micro_muse
            metrics_results[category]["MUSE"]["micro_Sensitivity"] = Sensitivity_micro_muse
            metrics_results[category]["MUSE"]["micro_Specificity"] = Specificity_micro_muse

            # Compute macro metrics for MUSE
            (
                ROC_macro_muse,
                PR_macro_muse,
                Sensitivity_macro_muse,
                Specificity_macro_muse,
            ) = compute_macro_metrics_with_youden(y_true, y_scores_muse)
            metrics_results[category]["MUSE"]["macro_ROC"] = ROC_macro_muse
            metrics_results[category]["MUSE"]["macro_PR"] = PR_macro_muse
            metrics_results[category]["MUSE"]["macro_Sensitivity"] = Sensitivity_macro_muse
            metrics_results[category]["MUSE"]["macro_Specificity"] = Specificity_macro_muse

    return metrics_results


def compute_individual_metrics(merged_df, new_label_names):
    metrics_results = {}

    for label in new_label_names:
        cardiologist_col = label + "_CARDIOLOGIST"
        muse_col = label + "_MUSE"

        if cardiologist_col in merged_df.columns and muse_col in merged_df.columns:
            print(f"Computing metrics for {label}")
            metrics_results[label] = {"_CARDIOLOGIST": {}, "_MUSE": {}}

            for suffix in ["_CARDIOLOGIST", "_MUSE"]:
                col_name = label + suffix
                y_true = merged_df[label]
                y_score = merged_df[col_name]

                # Check if both classes are present
                if np.unique(y_true).size > 1 and np.unique(y_score).size > 1:
                    ROC = roc_auc_score(y_true, y_score, average=None)
                    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
                    PR = auc(recall, precision)

                    # Calculate sensitivity, specificity, and Youden's index
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    youden_index = tpr - fpr
                    optimal_threshold = thresholds[np.argmax(youden_index)]
                    sensitivity = tpr[np.argmax(youden_index)]
                    specificity = 1 - fpr[np.argmax(youden_index)]

                    metrics_results[label][suffix] = {
                        "ROC": ROC,
                        "PR": PR,
                        "Precision": precision,
                        "Recall": recall,
                        "Thresholds": thresholds,
                        "Sensitivity": sensitivity,
                        "Specificity": specificity,
                        "Youden Index": youden_index,
                    }
                else:
                    metrics_results[label][suffix] = {
                        "Error": "Insufficient class variation to compute metrics"
                    }

    return metrics_results

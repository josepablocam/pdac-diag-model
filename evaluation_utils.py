import numpy as np
import sklearn.metrics
import tqdm

def get_bootstrap_ci(
        metric_fun,
        val,
        y_test,
        y_scores,
        num_iters,
        alpha,
        seed=None,
):
    diffs = []
    num_obs = len(y_test)
    ixs = np.arange(0, num_obs)
    if seed is not None:
        np.random.seed(seed)
    for _ in tqdm.tqdm(range(num_iters)):
        boot_ixs = np.random.choice(ixs, num_obs, replace=True)
        y_test_boot = y_test[boot_ixs]
        y_score_boot = y_scores[boot_ixs]
        metric_boot = metric_fun(y_test_boot, y_score_boot)
        diffs.append(metric_boot - val)

    # compute percentile
    alpha_percentile = (alpha / 2) * 100
    diffs = np.sort(diffs)
    below = np.percentile(diffs, alpha_percentile)
    above = np.percentile(diffs, 100 - alpha_percentile)
    return val - above, val - below



def specificity_and_sensitivity(
        y_test,
        y_prob,
        bootstrap_iters=1000,
        bootstrap_alpha=0.05,
):
    # note that we compute thresholds here with precision_recall_curve
    # and *not* with roc_curve (and use 1 - fpr), since roc_curve
    # drops some thresholds, while pr_curve keeps them
    # in the sklearn implementation
    _, specificity, thresholds = sklearn.metrics.precision_recall_curve(
        ~y_test,
        y_prob[:, 0],
    )

    # drop last which is defined to be zero
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve
    specificity = specificity[:-1]
    pos_thresholds = 1 - thresholds
    specificity_breaks = [(0.0, 0.949), (0.95, 0.989), (0.99, 1.0)]

    results = {}
    for (lower, upper) in specificity_breaks:
        chosen_thresholds = pos_thresholds[(lower <= specificity)
                                           & (specificity <= upper)]

        if len(chosen_thresholds) > 1:
            # recall (i.e. sensitivity) is monotonic
            upper_threshold = max(chosen_thresholds)
            lower_threshold = min(chosen_thresholds)

            # check what specificity we actually got with
            # these cutoffs, should be very close to the bounds
            # but may not be exactly
            # if we wanted exact would likely have to interpolate
            # but that seems hacky as well
            spec_high_achieved = sklearn.metrics.recall_score(
                ~y_test,
                ~(y_prob[:, 1] >= upper_threshold),
            )
            spec_low_achieved = sklearn.metrics.recall_score(
                ~y_test,
                ~(y_prob[:, 1] >= lower_threshold),
            )

            sens_high = sklearn.metrics.recall_score(
                y_test,
                y_prob[:, 1] >= lower_threshold,
            )
            sens_high_ci = get_bootstrap_ci(
                sklearn.metrics.recall_score,
                sens_high,
                y_test,
                y_prob[:, 1] >= lower_threshold,
                num_iters=bootstrap_iters,
                alpha=bootstrap_alpha,
            )

            sens_low = sklearn.metrics.recall_score(
                y_test,
                y_prob[:, 1] >= upper_threshold,
            )
            sens_low_ci = get_bootstrap_ci(
                sklearn.metrics.recall_score,
                sens_low,
                y_test,
                y_prob[:, 1] >= upper_threshold,
                num_iters=bootstrap_iters,
                alpha=bootstrap_alpha,
            )

            ppv_lb = sklearn.metrics.precision_score(
                y_test, y_prob[:, 1] >= lower_threshold)
            ppv_lb_ci = get_bootstrap_ci(
                sklearn.metrics.precision_score,
                ppv_lb,
                y_test,
                y_prob[:, 1] >= lower_threshold,
                num_iters=bootstrap_iters,
                alpha=bootstrap_alpha,
            )

            ppv_ub = sklearn.metrics.precision_score(
                y_test, y_prob[:, 1] >= upper_threshold)
            ppv_ub_ci = get_bootstrap_ci(
                sklearn.metrics.precision_score,
                ppv_ub,
                y_test,
                y_prob[:, 1] >= upper_threshold,
                num_iters=bootstrap_iters,
                alpha=bootstrap_alpha,
            )
        else:
            # nothing satisfied the thresholds we want
            spec_high_achieved = np.nan
            spec_low_achieved = np.nan
            sens_high = np.nan
            sens_high_ci = (np.nan, np.nan)
            sens_low = np.nan
            sens_low_ci = (np.nan, np.nan)
            ppv_lb = np.nan
            ppv_lb_ci = (np.nan, np.nan)
            ppv_ub = np.nan
            ppv_ub_ci = (np.nan, np.nan)

        values_and_ci = {
            "spec_ub_achieved": (spec_high_achieved, (np.nan, np.nan)),
            "spec_lb_achieved": (spec_low_achieved, (np.nan, np.nan)),
            "sens_ub": (sens_high, sens_high_ci),
            "sens_lb": (sens_low, sens_low_ci),
            "ppv_lb": (ppv_lb, ppv_lb_ci),
            "ppv_ub": (ppv_ub, ppv_ub_ci),
        }
        key = "spec_{}_{}".format(lower, upper)
        for value_name, (val, (ci_lb, ci_ub)) in values_and_ci.items():
            results[key + "_" + value_name] = val
            results[key + "_" + value_name + "_ci_lb"] = ci_lb
            results[key + "_" + value_name + "_ci_ub"] = ci_ub
    return results


def quick_evaluate(
        y_test,
        y_prob,
        bootstrap_iters=1000,
        bootstrap_alpha=0.05,
        uids=None,
):
    results = {}
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.values
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_prob[:, 1])
    results["y_true"] = y_test
    results["y_prob"] = y_prob[:, 1]
    if uids is not None:
        results["UID"] = uids
    results["fpr"] = fpr
    results["tpr"] = tpr
    results["thresholds"] = thresholds
    results["roc_auc"] = sklearn.metrics.roc_auc_score(y_test, y_prob[:, 1])
    roc_auc_ci = get_bootstrap_ci(
        sklearn.metrics.roc_auc_score,
        results["roc_auc"],
        y_test,
        y_prob[:, 1],
        num_iters=bootstrap_iters,
        alpha=bootstrap_alpha,
    )
    results["roc_auc_ci_lb"] = roc_auc_ci[0]
    results["roc_auc_ci_ub"] = roc_auc_ci[1]

    results["aps"] = sklearn.metrics.average_precision_score(
        y_test, y_prob[:, 1])
    aps_ci = get_bootstrap_ci(
        sklearn.metrics.average_precision_score,
        results["aps"],
        y_test,
        y_prob[:, 1],
        num_iters=bootstrap_iters,
        alpha=bootstrap_alpha,
    )
    results["aps_ci_lb"] = aps_ci[0]
    results["aps_ci_ub"] = aps_ci[1]

    precision, recall, pr_thresholds = sklearn.metrics.precision_recall_curve(
        y_test, y_prob[:, 1])
    results["pr_precision"] = precision
    results["pr_recall"] = recall
    results["pr_thresholds"] = thresholds
    results["count_known_cancer"] = y_test.sum()
    results["total_count"] = len(y_test)
    results["avg_prob"] = y_prob[:, 1].mean()
    spec_sense_results = specificity_and_sensitivity(
        y_test,
        y_prob,
        bootstrap_iters=bootstrap_iters,
        bootstrap_alpha=bootstrap_alpha,
    )
    results.update(spec_sense_results)
    # monitor anybody with more than 5 times risk of
    # test population
    mult = 5
    is_high_risk = y_prob[:, 1] >= (mult * y_prob[:, 1].mean())
    results["ct_high_risk"] = is_high_risk.sum()
    results["ct_cancer_high_risk"] = y_test[is_high_risk].sum()
    if is_high_risk.sum() > 0:
        results["ppv_high_risk"] = results["ct_cancer_high_risk"] / float(results["ct_high_risk"])
    else:
        results["ppv_high_risk"] = np.nan
    return results

from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier


def get_scale_pos_weight(y):
    """
    Compute ratio of negative to positive (i.e. controls to cancer) patients
    """
    return ((~y).sum()) / y.sum()


class XGBWrapped(XGBClassifier):
    def __init__(self,
                 max_depth=3,
                 learning_rate=0.1,
                 n_estimators=100,
                 verbosity=1,
                 silent=None,
                 objective='binary:logistic',
                 booster='gbtree',
                 n_jobs=1,
                 nthread=None,
                 gamma=0,
                 min_child_weight=1,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 colsample_bynode=1,
                 reg_alpha=0,
                 reg_lambda=1,
                 scale_pos_weight=1,
                 base_score=0.5,
                 random_state=0,
                 seed=None,
                 missing=None,
                 **kwargs):
        super().__init__(max_depth=3,
                         learning_rate=0.1,
                         n_estimators=100,
                         verbosity=1,
                         silent=None,
                         objective='binary:logistic',
                         booster='gbtree',
                         n_jobs=1,
                         nthread=None,
                         gamma=0,
                         min_child_weight=1,
                         max_delta_step=0,
                         subsample=1,
                         colsample_bytree=1,
                         colsample_bylevel=1,
                         colsample_bynode=1,
                         reg_alpha=0,
                         reg_lambda=1,
                         scale_pos_weight=1,
                         base_score=0.5,
                         random_state=0,
                         seed=None,
                         missing=None,
                         **kwargs)

    def fit(self, X, y):
        # set scale_pos_weight
        scale_pos_weight = get_scale_pos_weight(y.astype(bool))
        self.set_params(scale_pos_weight=scale_pos_weight)
        return super().fit(X, y)


def xgb_ensemble(model, **kwargs):
    return BalancedBaggingClassifier(base_estimator=model, **kwargs)

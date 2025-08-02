from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import numpy as np

from utils.GeneralTool import GeneralTool


class IsolationForestClassification:
    def __init__(self, cfg):
        self.model = IsolationForest(
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
            random_state=GeneralTool.seed
        )
        self.pca = PCA(
            n_components=cfg["n_components"],
            random_state=GeneralTool.seed
        )

    def train(self, X):
        self.pca.fit(X)
        X_pca = self.pca.transform(X)
        self.model.fit(X_pca)
        y = self.model.predict(X_pca)
        return np.where(y == -1, 1, 0)

    def test(self, X):
        X_pca = self.pca.transform(X)
        y = self.model.predict(X_pca)
        return np.where(y == -1, 1, 0)


from sklearn.decomposition import PCA

from utils.GeneralTool import GeneralTool


class PCA:
    def __init__(self, cfg):
        self.model = PCA(
            n_components=cfg["n_components"],
            random_state=GeneralTool.seed
        )

    def train(self, X):
        return self.model.fit_transform(X)

    def test(self, X):
        return self.model.transform(X)

from ltp import StnSplit
from scipy.spatial import distance


class NLPTool:

    ltp = StnSplit()

    @classmethod
    def split_sentences(cls, text):
        return cls.ltp.split(text)

    @classmethod
    def compute_cosine_similarity(cls, vec_1, vec_2):
        return float(1 - distance.cosine(vec_1, vec_2))

import numpy as np
import pandas as pd

from .identity_split import IdentitySplit


class MinimalGallerySplit(IdentitySplit):
    #returns only one database image per individual
    def __init__(self, ratio_train : float = 1, **kwargs):
        self.ratio_train = ratio_train
        super().__init__(**kwargs)

    def split(self, df : pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        lcg = self.initialize_lcg()
        y_counts = df[self.col_label].value_counts()
        class_names = y_counts.index.to_numpy()
        idx_test = lcg.random_permutation(df.shape[0])
        
        idx_train = []
        classes_included = []
        #draw random items until we have one from each class
        for i in range(df.shape[0] - 1, -1, -1):
            idx = idx_test[i]
            current_class = df.at[idx, "identity"]
            if current_class not in classes_included:
                classes_included.append(current_class)
                idx_train.append(int(idx))
                idx_test = np.delete(idx_test, i)
            if (len(class_names) * self.ratio_train) <= len(classes_included):
                break
        idx_train = np.array(idx_train)
        
        return [(idx_train, idx_test)]

from sklearn.datasets import make_classification
# 总共有5000个样本，分为三个类别，样本量比例为1：5：94
X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)
from imblearn.over_sampling import SMOTE
# 进行SMOTE上采样
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
# 打印处理后的样本量
from collections import Counter
print(sorted(Counter(y_resampled).items()))
# 输出结果为（类别标号，样本量）：
# [(0, 4674), (1, 4674), (2, 4674)]
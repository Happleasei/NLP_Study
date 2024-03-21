from sklearn.linear_model import LogisticRegression
def test_LogisticRegression(X_train, X_test, y_train, y_test):
    # 初始化模型
    cls = LogisticRegression()
    # 模型训练
    cls.fit(X_train, y_train)
    # 模型预测
    print('Score: %.2f' % cls.score(X_test, y_test))

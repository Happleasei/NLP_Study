from sklearn.tree import DecisionTreeClassifier
def test_DecisionTree(X_train, X_test, y_train, y_test):
    # 初始化模型
    cls = DecisionTreeClassifier(criterion="entropy")
    # 模型训练
    cls.fit(X_train, y_train)
    # 模型预测
    print('Score: %.2f' % cls.score(X_test, y_test))

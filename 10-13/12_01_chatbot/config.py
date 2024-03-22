class DefaultConfig:
    # 批训练大小
    batch_size = 10
    # 训练轮次
    epochs = 3000
    # 词向量维度
    w2v_size = 128
    # 隐层维度
    hidden_dim = 100
    # 优化方式
    optimizer = 'adam'
    # 推理过程中目标句子最大长度
    n_steps = 80
    # 词典保存路径
    dict_path = 'data/dict.pkl'
    # 文件路径
    data_path = 'data/subtitle.txt'
    # 模型保存路径
    model_path = 'model/model_best.h'

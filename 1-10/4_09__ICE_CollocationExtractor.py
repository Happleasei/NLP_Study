from ICE import CollocationExtractor
# 输入文本
input=["he and Chazz duel with all keys on the line."]
# 提取器
extractor = CollocationExtractor.with_collocation_pipeline("T1", bing_key = "Temp",pos_check = False)
# 输出结果为：
# [“on the line”]
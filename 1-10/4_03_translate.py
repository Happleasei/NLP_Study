import string
# 输入文本
input_str = "The 5 biggest countries by population in 2019 are China, India, United States, Indonesia, and Brazil."
# 剔除标点
output_str = input_str.translate(string.maketrans("",""), string.punctuation)
print(output_str)
# 输出结果为：
# The biggest countries by population in are China India United States Indonesia and
# Brazil
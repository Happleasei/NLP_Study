from config import DefaultConfig
from model.seq2seq import Seq2Seq
from data.data_parse import DataParse
import pickle

myModel = Seq2Seq(DefaultConfig)
dataParse = DataParse(DefaultConfig)

def train():
    # 获取数据
    input_texts, target_texts = dataParse.data_read()
    # 处理数据
    dict_len, encoder_input, decoder_input, decoder_output = dataParse.data_parse(input_texts, target_texts)
    print(encoder_input)
    print(decoder_input)
    # 训练模型
    myModel.basic_model(dict_len, encoder_input, decoder_input, decoder_output)

def test():
    # 加载推理模型
    encoder_infer, decoder_infer = myModel.infer_model()

    while True:
        source = input("请输入句子：")

        # 提取词和标号的字典
        with open(DefaultConfig.dict_path, 'rb') as f:
            dict_word_index = pickle.load(f)

        # 生成标号和词的字典
            dict_reverse = dict( [(i, word) for word, i in dict_word_index.items()])

        # 测试
        output = myModel.inference(source, encoder_infer, decoder_infer, dict_word_index, dict_reverse)
        print(output)

if __name__ == '__main__':
    train()
    #test()

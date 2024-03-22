import numpy as np
import jieba
import pickle

class DataParse:
    def __init__(self, DefaultConfig):
        self.opts = DefaultConfig

    def data_read(self):
        # 读取数据
        with open(self.opts.data_path, 'r', encoding='utf8', errors='ignore') as f:
            data_raw = f.readlines()

        # 分词, 把\t,\n作为句子的开始以及结束标志符号，将句子以数组方式保存
        texts = [jieba.lcut('\t' + line[4:]) for line in data_raw]
        print(texts)
        # 奇数项的为输入的句子（源语句），偶数项的为输出的句子（目标语句）
        input_texts = texts[::2]
        target_texts = texts[1::2]
        return input_texts, target_texts

    def data_parse(self, input_texts, target_texts):
        """
        :param input_texts: 源语句
        :param target_texts: 目标语句
       	:return:
         	 dict_len：字典长度
             encoder_input_data：编码器输入
             decoder_input_data：解码器输入
             decoder_target_data：解码器输出
        """
        # 计算词典
        texts = input_texts + target_texts
        dict_words = set()
        for text in texts:
            for w in text:
                if w not in texts:
                    dict_words.add(w)

        # 获取字典长度
        dict_len = len(dict_words)

        # 生成词以及标号对应表并且保存
        dict_word_index = dict(
            [(word, i) for i, word in enumerate(dict_words)]
        )
        with open(self.opts.dict_path, 'wb') as f:
            pickle.dump(dict_word_index, f)

        # 分别算出输入句子和输出句子的最大长度
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        # 初始化矩阵
        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length),
            dtype=np.int32)
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length),
            dtype=np.int32)
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, dict_len),
            dtype=np.float32)

        # 将文字输入转化为张量
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            # 将编码器和解码器的输入文字转化为字典中对应的标号
            input_index = self.seq2index(input_text, dict_word_index)
            encoder_input_data[i, :len(input_index)] = input_index

            target_index = self.seq2index(target_text, dict_word_index)
            decoder_input_data[i, :len(target_index)] = target_index

            # 将解码器的输出文字转为one-hot向量，与解码器的输入相比往后偏移一位
            for t, index in enumerate(decoder_input_data[i, 1:]):
                decoder_target_data[i, t, index] = 1.0

        return dict_len, encoder_input_data, decoder_input_data, decoder_target_data

    @staticmethod
    def seq2index(text, dict_word_index):
        """
        :param text: 中文语句
        :param dict_word_index: 词与标号对应词典
        :return: 转化为标号的语句
        """
        # 将输入文字转化为字典中对应的标号,找不到的则记为0
        return [dict_word_index.get(word, 0) for word in text]
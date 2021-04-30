from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Dense
from keras import callbacks
import numpy as np

class Seq2Seq:
    def __init__(self, DefaultConfig):
        self.opts = DefaultConfig

    def basic_model(self, dict_len, encoder_input, decoder_input, decoder_output):
        """
        :param dict_len: 字典长度
        :param encoder_input: 编码器输入
        :param decoder_input: 解码器输入
        :param decoder_output: 解码器输出
        :return:
        """
        # 训练阶段

        # 编码器模型
        # 输入的是one-hot向量
        encoder_inputs = Input(shape=(None,), name='encoder_inputs')

        # 经过一层embedding将词转化为词向量
        encoder_embedding = Embedding(dict_len, self.opts.w2v_size,
                                      name='encoder_embedding')(encoder_inputs)

        # 编码器模型为LSTM模型，每一步都会输出state, 即h,c
        encoder = LSTM(self.opts.hidden_dim, return_state=True, return_sequences=True, name='encoder_lstm')

        # 计算得编码器的输出隐状态
        _, *encoder_states = encoder(encoder_embedding)

        # 解码器模型
        # 输入的维度为one-hot向量
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')

        # 经过一层embedding将词转化为词向量
        decoder_embedding = Embedding(dict_len, self.opts.w2v_size,
                                      name='decoder_embedding')(decoder_inputs)

        # 解码器模型为LSTM, 每一步都会输出state，以及sequence，sequence输出用于与真实结果对比优化
        decoder = LSTM(self.opts.hidden_dim, return_state=True, return_sequences=True, name='decoder_lstm')
        # 解码器的初始状态为编码器的输出状态,获取输出
        decoder_outputs, *decoder_states = decoder(decoder_embedding, initial_state=encoder_states)

        # 加全连接层，维度为词典长度，相当于多分类问题，类别大小为词语的个数
        decoder_dense = Dense(dict_len, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # 训练参数
        model.compile(optimizer=self.opts.optimizer, loss='categorical_crossentropy')

        # 查看模型概要
        model.summary()

        # 回调函数，只保存最佳模型
        callback_list = [callbacks.ModelCheckpoint(self.opts.model_path, save_best_only=True)]

        # 模型训练
        model.fit([encoder_input, decoder_input], decoder_output,
                  batch_size=self.opts.batch_size, epochs=self.opts.epochs,
                  validation_split=0.2, callbacks=callback_list)

    def infer_model(self):
        """
        :return:
            encoder_infer：推理过程编码器
            decoder_infer：推理过程解码器
        """
        # 推理阶段，用于预测

        # 加载训练好的模型
        model = load_model(self.opts.model_path)

        # 推理阶段encoder
        # encoder输入
        encoder_inputs = Input(shape=(None,))
        # 获取词embedding层的输出
        encoder_embedding = model.get_layer('encoder_embedding')(encoder_inputs)
        # 获取编码后的状态
        _, *encoder_states = model.get_layer('encoder_lstm')(encoder_embedding)
        encoder_infer = Model(encoder_inputs, encoder_states)

        # 推理阶段decoder
        # 解码器的输入
        decoder_inputs = Input(shape=(None,))
        # 解码器的输入状态
        decoder_state_input_h = Input(shape=(self.opts.hidden_dim,))
        decoder_state_input_c = Input(shape=(self.opts.hidden_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # 经历embedding层的输出
        decoder_embedding = model.get_layer('decoder_embedding')(decoder_inputs)

        # 解码器的输出，以及状态
        decoder_infer_output, *decoder_infer_states = model.get_layer('decoder_lstm')(decoder_embedding, initial_state=decoder_states_inputs)

        # 经过全连接层，得到当前时刻的输出
        decoder_infer_output = model.get_layer('decoder_dense')(decoder_infer_output)
        decoder_infer = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_infer_output] + decoder_infer_states)

        return encoder_infer, decoder_infer

    def inference(self, source, encoder_infer, decoder_infer, dict_word_index, dict_reverse):
        """
        :param source: 输入语句
        :param encoder_infer: 推理过程编码器
        :param decoder_infer: 推理过程解码器
        :param dict_word_index: 词与标号的对应
        :param dict_reverse: 标号与词的对应
        :return: 预测的回答
        """
        text = self.seq2index(source, dict_word_index)

        # 通过编码器得到源句子的隐状态
        state = encoder_infer.predict(text)

        # 解码器的初始输入字符为'\t'
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = dict_word_index['\t']

        output = ''
        # 通过编码器得到的state作为解码器的初始状态输入
        # 解码过程中，每次利用上次预测的词作为输入来预测下一次的词，直到预测出终止符'\n'
        for i in range(self.opts.n_steps):
            # 每一次输出单词以及隐状态
            y, h, c = decoder_infer.predict([target_seq] + state)

            # 获取可能性最大的词
            word_index = np.argmax(y[0, -1, :])
            word = dict_reverse[word_index]

            # 如果预测出'\n'则终止循环
            if word == '\n':
                break

            # 将新预测出的单词添加到输出中
            output = output + " " + word

            # 更新下一步要输入的隐状态
            state = [h, c]

            # 更新下一步要输入的词
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = dict_word_index[word]

        return output

    @staticmethod
    def seq2index(text, dict_word_index):
        """
        :param text: 中文语句
        :param dict_word_index: 词与标号对应词典
        :return: 转化为标号的语句
        """
        # 将输入文字转化为字典中对应的标号,找不到的则记为0
        return [dict_word_index.get(word, 0) for word in text]

# -- encoding:utf-8 --
"""
Create on 19/8/24 10:54
"""
import tensorflow as tf
from .base import BaseModel


class FCClassifier():
    def __init__(self, output_layer, labels, is_training, num_labels):
        with tf.variable_scope("fc_cls"):
            # 直接输入向量
            self.output_layer = output_layer
            self.labels = labels
            self.is_training = is_training
            self.num_labels = num_labels

            # 构建模型
            tf.logging.info("**** Start initialization FC model. ****")
            self.build_model()

    def build_model(self):
        with tf.variable_scope("loss"):
            output_layer = self.output_layer
            # 获取得到输入的维度大小
            hidden_size = output_layer.shape[-1].value

            output_weights = tf.get_variable(
                "output_weights", [self.num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

            if self.is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            predictions = tf.argmax(logits, axis=-1, name="predictions")
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(self.labels, depth=self.num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            # 返回的顺序为所有样本的总损失、每个样本的损失、网络输出值(置信度)、经过softmax转换后的概率值
            self.loss = loss
            self.per_example_loss = per_example_loss
            self.logits = logits
            self.probabilities = probabilities
            self.predictions = predictions

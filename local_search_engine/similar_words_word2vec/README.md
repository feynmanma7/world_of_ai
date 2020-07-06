<h1>用word2vec来查找相似词语</h1>


# 1. 使用效果

## 1.1 输入

输入一个词语，如“推荐”。

## 1.2 输出

输出词语的相似词语，

> ['趋势', '预测', '分类', '过滤', '相关', '图表', '表现', '语义', '统计', '图像', '意义', '波动', '方式', '原始', '集合', '获取', '场景', '研究所', '程度']

训练语料是笔者的电脑中的pdf文件。本文介绍的模型只针对单个词，不能查找类似“推荐系统”这样的短语的相似短语。

# 2. 使用过程

## 2.1 模型定义
### 2.1.1 问题定义

采用CBOW, 即对于每一个中心词语（target_word），选择其左右两边的词作为上下文（context_words）。

问题定义成：给定上下文，最大化中心词语的概率。

> $\max p(target\_word|context\_words)$

分别用$T$与$C$来代表target_word与context_words，$V$表示词典。

> $p(T|C) = \frac{p(C, T)}{p(C)}=\frac{p(C, T)}{\sum_{t\in V}(C, t)}$

$p(T|C)$可以用softmax回归来表示，

> $p(T|C) = \frac{exp(g(f_{in}(C), f_{out}(T))}{\sum_{t}exp(g(f_{in}(C), f_{out}(t)))}$

word2vec是为了学习词语的向量表示。

1. 输入向量表示$f_{in}$：假设词语的输入向量表示为$f_{in}(word)$。
2. 输出向量表示$f_{out}$：假设输出向量表示为$f_{out}(word)$。
3. 输入、输出向量运算关系$g$：用$g(a, b)$表示的$a$与$b$的运算函数。原始的word2vec用的是cosine相似度，也可以用MLP或者其它运算方式。

输入、输出均用一个Embedding来表示，用cosine相似度来表示$g$。

> $p(T|C)=\frac{exp(cos(W_{in}^T C, W_{out}^T T))}{\sum_{t \in V} exp(cos(W_{in}^T C, W_{out}^T t))}$

## 2.1.2 负采样 

现实应用中，词典$V$的维度是非常高的（根据不同的应用，可能会大至上千万、上亿），softmax运算量会特别大（分母需要遍历整个词典，计算词典中每个词与context的相似度）。word2vec创造性地采用了Hierachical Softmax以及Negative Sampling两种方式来大幅度减少计算量。

本文介绍在实践中效果相对更好，应用更广泛的**负采样**Negative Sampling方法。

为了**大幅度**减少计算量，负采样通过采样**一小部分**负样本来参与计算，让模型能较好地区分真实的中心词（正样本）与假的中心词（负样本），从而**近似**模拟遍历全部词典计算softmax的效果。

问题转化为：应该怎样采样负样本，能更好地接近softmax的效果。

给定上下文，绝大部分词并不太可能是中心词。模型需要准确识别很有可能是但实际上不是中心词的词语；因此，一种比较合理的采样方式是根据语料中的**词频**采样，让词频高的词语，以更大的可能性成为负样本。

### 2.1.3 损失函数

使用负采样方式，损失函数定义为**负log似然**。

对于log似然依然采用**近似计算**的方式，请参见word2vec原始论文[1]。

对于正样本$T$，假设

> $p(T=1|C) = \sigma (cos(C, T))= \frac{1}{1 + exp(- cos(C, T))}$

对于负样本$t \in NEG$，假设

> $p(t=0|C) = 1 - \sigma(cos(C, t)) = 1 - \frac{1}{1 + exp(-cos(C, t))}$
>
> $=\frac{exp(-cos(C, t))}{1 + exp(-cos(C, t))}$
>
> $=\frac{1}{exp(+cos(C, t)) + 1} = \frac{1}{1 + exp(- (-cos(C, t)))}$
>
> $=\sigma(-cos(C, t))$

损失函数定义为负log似然，

> $loss = - \log p(Y|X; \theta) = - \log (p(T|C) \prod_{i \in NEG} p(t | C)) $
>
> $= -\log p(T|C) - \sum_{t \in NEG} \log p(t | C)$
>
> $=-log \sigma(cos(C, T)) - \sum_{t \in NEG} \log \sigma(-cos(C, t))$

## 2.2 数据处理

仍然处理本地的吃灰文件（[动手做一个简单的本地搜索引擎](https://mp.weixin.qq.com/s?__biz=MzUyNTEwMjM4NQ==&mid=2247483691&idx=1&sn=815c42b657291bb78a3c599ed2f5b1ca&chksm=fa227d75cd55f463d68e7c16dbf651094c05f724f4824f11f7c7bcfe2f6de1a85da3e572ccab&token=575378049&lang=zh_CN#rd)）。

1. 将pdf文件处理成文本文件：进行分词、去除停用词等操作，每一页文章处理完放到一行。
2. 统计词频: {word: count}。
3. 得到模型词典：去除低频词，按照词频排序，得到模型词典word2id: {word: id}。

# 3. 模型训练
## 3.1 模型定义
使用tensorflow2.0+的自定义训练方式。

1. 定义两个不同的Embedding层，分别对应输入、输出的Embedding。
2. 对于上下文的多个词语，对于每一个词语的Embedding向量取平均。

```python
class Word2vec(tf.keras.Model):
    def __init__(self,vocab_size=None, 
                 window_size=None, 
                 num_neg=None,
                 embedding_dim=None):
        super(Word2vec, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.num_neg = num_neg
        self.embedding_dim = embedding_dim

        self.input_embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.output_embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim)

    def call(self, contexts, target, negatives):
        # === Embedding
        # [None, w*2, emb_dim]
        contexts_embedding = self.input_embedding_layer(contexts)

        # [None, 1, emb_dim]
        contexts = tf.reduce_mean(contexts_embedding, axis=1, keepdims=True)

        # [None, 1, emb_dim]
        target_embedding = self.output_embedding_layer(target)

        # [None, num_neg, emb_dim]
        negatives_embedding = self.output_embedding_layer(negatives)

        # === Cosine similarity
        # [None, 1, 1]
        target_cos = tf.keras.layers.Dot(axes=(2, 2), normalize=True)\
            ([target_embedding, contexts])

        # [None, 1, num_neg]
        negatives_cos = tf.keras.layers.Dot(axes=(2, 2), normalize=True)\
            ([target_embedding, negatives_embedding])
			
      	# === Loss
        pos_loss = tf.reduce_mean(-tf.math.log(1 / (1 + tf.math.exp(-target_cos))))
        neg_loss = tf.reduce_mean(-tf.math.log(1 / (1 + tf.math.exp(negatives_cos))))

        loss = pos_loss + neg_loss

        return loss

```
## 3.2 输入处理

将输入数据处理成一个分词序列，词语之间用空格分隔，如：

> 推荐 算法 基于 **用户** 评分 矩阵 rating matrix 基于 用户 评论 用户 隐式 反馈 数据 方法 推荐 越来越 关注 研究 长期以来 文本 挖掘 用户 数据 收集 难点 制约 研究 解决 推荐 解释性 冷启动 确实 潜力 推荐 系统 输出 数据 一个 特定 用户 推荐 系统 输出 一个 推荐 列表 推荐 列表 优先级 顺序 给出 用户 感兴趣 物品 一个 实用 推荐 系统 仅仅 给出 推荐 列表 用户 系统 给出 推荐 不太会 采纳 系统 给出 推荐 解决 推荐 系统 一个 输出 推荐 理由 表述 系统 推荐 物品 购买 商品 用户 购买 商品 解决 推荐 合理性 推荐 理由 产业


然后产生一条一条的训练需要的数据格式，

> (contexts: window_size * 2, target: 1, negatives: num_neg)

window_size指中心词的上下文的单侧窗口大小，如window_size设置为3，假设选择“用户”为中心词，则上下文为

> [推荐 算法 基于 评分 矩阵 rating]

使用numpy的按权重采样来采样负样本。采样权重按照论文中给出的，词频的3/4次方。

```python
numpy.random.choice(list(range(vocab_size)), num_sample, p=weights)
```

产生批量训练数据，

```python
def get_dataset(input_path=None,
            		dict_dir=None,
                epochs=10,
                shuffle_buffer_size=128,
                batch_size=8,
                window_size=3,
                num_neg=5):

    word_cnt_dict_path = os.path.join(dict_dir, "word_cnt_dict.pkl")
    word2id_dict_path = os.path.join(dict_dir, "word2id_dict.pkl")
    id2word_dict_path = os.path.join(dict_dir, "id2word_dict.pkl")

    word_cnt_dict = load_dictionary(dict_path=word_cnt_dict_path)
    word2id_dict = load_dictionary(dict_path=word2id_dict_path)
    id2word_dict = load_dictionary(dict_path=id2word_dict_path)

    sampler = Sampler(word_cnt_dict=word_cnt_dict, id2word_dict=id2word_dict)

    def generator():
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                buf = line[:-1].split(' ')

                if len(buf) < window_size * 2 + 1:
                    continue
								
                # === word to index
                index_buf = []
                for word in buf:
                    if word not in word2id_dict:
                        continue
                    index_buf.append(word2id_dict[word])

                if len(index_buf) < window_size * 2 + 1:
                    continue

                # === Simply remove region out of window_size, use (window_size, target, window_size)
                for i in range(window_size, len(index_buf) - window_size):
                    target = [index_buf[i]]
                    contexts = index_buf[i - window_size: i] + index_buf[i + 1: i + 1 + window_size]
                    negatives = sampler.sample(num_sample=num_neg, method="weighted")

                    yield contexts, target, negatives
		
    # === get tensorflow dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_shapes=((window_size*2, ), (1, ), (num_neg, )),
        output_types=(tf.int32, tf.int32, tf.int32)
    )

    return dataset.repeat(count=epochs)\
        .shuffle(buffer_size=shuffle_buffer_size)\
        .batch(batch_size=batch_size)
```

## 3.3 模型训练

自定义训练步，

```python
@tf.function
def train_step(model, optimizer, contexts, target, negatives):

    with tf.GradientTape() as tape:
        batch_loss = model(contexts, target, negatives)

    variables = model.trainable_variables
    gradients = tape.gradient(batch_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss
```

整个模型训练过程如下，

```python
# === model
model = Word2vec(vocab_size=vocab_size,
                     window_size=window_size,
                     num_neg=num_neg,
                     embedding_dim=embedding_dim)
# === optimizer
optimizer = tf.keras.optimizers.Adam(0.001)

# === checkpoint 
checkpoint_dir = os.path.join(get_model_dir(), "word2vec")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

total_train_batch = total_num_train // batch_size + 1

for epoch in range(epochs):
  total_loss = 0
  batch_loss = 0

  epoch_start = time.time()
  i = 0
  for batch_idx, (contexts, target, negatives) in zip(range(total_train_batch), train_dataset):
    i += 1
    
    # === self-defined batch train
    cur_loss = train_step(model, optimizer, contexts, target, negatives)
    batch_loss += cur_loss

    if i % 100 == 0:
      batch_end = time.time()
      batch_last = batch_end - start
      print("Epoch: %d/%d, batch: %d/%d, batch_loss: %.4f, cur_loss: %.4f, lasts: %.2fs" %
            (epoch+1, epochs, batch_idx+1, total_train_batch, batch_loss/(batch_idx+1), cur_loss, batch_last))

      assert i > 0
      batch_loss /= i

      total_loss += batch_loss
      epoch_end = time.time()
      epoch_last = epoch_end - epoch_start
      print("Epoch: %d/%d, loss: %.4f, lasts: %.2fs" % (epoch+1, epochs, total_loss/(epoch+1), epoch_last))
		
    	# === model save
      checkpoint.save(file_prefix=checkpoint_prefix)
```

## 3.4 查找相似词语

与训练一样，定义模型，从checkpoint中恢复模型。

```python
# === model definition
w2v = Word2vec(vocab_size=vocab_size,
               window_size=window_size,
               num_neg=num_neg,
               embedding_dim=embedding_dim)

# === optimizer
optimizer = tf.keras.optimizers.Adam(0.001)

# === Load model from checkpoint
checkpoint_dir = os.path.join(get_model_dir(), "word2vec")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=w2v)
latest = tf.train.latest_checkpoint(checkpoint_dir)
status = checkpoint.restore(latest)
status.assert_existing_objects_matched()
```



需要注意的是，tensorflow2.0使用checkpoint恢复模型，需要先用测试数据跑一遍模型，整个网络的权重才会真正恢复。

```python
print("Run ONCE")
print(test_word2vec_once(model=w2v))

def test_word2vec_once(model=None):
    contexts = tf.random.uniform((8, model.window_size*2),
                                 minval=0, maxval=model.vocab_size, dtype=tf.int32)
    target = tf.random.uniform((8, 1),
                               minval=0, maxval=model.vocab_size, dtype=tf.int32)
    negatives = tf.random.uniform((8, model.num_neg),
                                  minval=0, maxval=model.vocab_size, dtype=tf.int32)

    loss = model(contexts, target, negatives)
    return loss
```



获取模型输出层的Embedding权重，经过归一化保存下来。

对于要查找的词语，直接计算归一化之后的向量相似度，从大到小排序；根据word2id词典构造出根据index查找word的反向词典id2word来查找相似word。

```python
sims = numpy.dot(word_vecs, word_vec)
ranks = np.argsort(-sims)
sim_words = [id2word_dict[idx] for idx in ranks[:topN]]
```

# 4. 总结

本文简单介绍了word2vec以及其CBOW实现方式。词语的分布式表示想法来自Bengio[3]，word2vec以其快速的计算，便捷的应用及有趣的效果启发了近年机器学习领域对Embedding的探索，基本上万物都可Embedding，在工业界Embedding也早已成为标配。


# 5. 参考资料

1. Mikolov T , Chen K , Corrado G , et al. Efficient Estimation of Word Representations in Vector Space[J]. Computer ence, 2013.
2. Rong X . word2vec Parameter Learning Explained[J]. Computer ence, 2014.
3. Bengio Y, Ducharme R, Vincent P, et al. A Neural Probabilistic Language Model[C]. neural information processing systems, 2000: 932-938.
4. 《动手学深度学习》，阿斯顿·张（Aston Zhang） / 李沐（Mu Li） / [美] 扎卡里·C. 立顿（Zachary C. Lipton） / [德] 亚历山大·J. 斯莫拉（Alexander J. Smola）
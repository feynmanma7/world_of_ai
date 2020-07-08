# 基于词典匹配的中文分词

中文分词在中文的自然语言处理里非常重要，比如在中文的搜索引擎（[动手做一个简单的本地搜索引擎](https://mp.weixin.qq.com/s?__biz=MzUyNTEwMjM4NQ==&mid=2247483691&idx=1&sn=815c42b657291bb78a3c599ed2f5b1ca&chksm=fa227d75cd55f463d68e7c16dbf651094c05f724f4824f11f7c7bcfe2f6de1a85da3e572ccab&token=1125906106&lang=zh_CN#rd)）或者语义分析中，词语粒度的分析效果是非常好的。

本篇介绍**基于词典匹配**的中文分词算法。

# 1. 使用效果

## 1.1 输入

输入待分词的句子或文章，如：

> 黑夜给了我黑色的眼睛，我却用它寻找光明。

## 1.2 输出

返回分词结果：

> 黑夜/给/了/我/黑色/的/眼睛/，/我/却/用/它/寻找/光明/。

# 2. 使用过程

## 2.1 模型训练

1. 根据已经分好词的训练语料，统计词频，得到词典。

**Done**。

基于词典匹配的中文分词就是这么简单。

当然，其它的效果更好的模型，训练过程就复杂得多了，后续会不断分享。

## 2.2 模型测试

1. 加载词典。

2. 对输入语句进行分词。

## 2.3 模型评估

### 2.3.1 测试数据

自然语言是一个不好量化的领域，中文分词的结果没有事实上的标准，语言学家标注的分词结果有的时候也存在争议。

在对比不同的中文分词模型的效果与性能时，选择同样的标注数据进行评估，可以在一定程度上对比下模型之间的优劣。

本篇选用第二届国际汉语分词评测比赛(http://sighan.cs.uchicago.edu/bakeoff2005/)提供的数据中的msr数据，包括:

> 训练数据：training/msr_training.utf8

> 测试数据：testing/msr_test.utf8

> 测试数据基准数据：gold/msr_test_gold.utf8

### 2.3.2 评价指标

标准的中文分词评测的维度很多，包括词典内外的分词效果等，本篇只关注3个指标：

> 精准率: Precision

> 召回率: Recall

> Precision与Recall的调和均值: F1_score

分词结果对比示例，

> 测试结果: [中国/人/真/伟大]， 一共4个词语。  

> 基准结果: [中国人/真/伟大]，一共3个词语。

正确分出的词语是2个：[真，伟大]

> Precision = 2 / 4

> Recall = 2 / 3

### 2.3.3 测试过程

1. 使用分词算法对待测试文件testing/msr_test.utf8分词。

2. 将分词结果与测试数据基准数据gold/msr_test_gold.utf8进行比对。


# 3. 模型训练

## 3.1 生成词典

```python
def genearte_dictionary(train_path=None, dict_path=None):
    """
    train_data:  Labeled words split by space.
    word_dict:  {word: word_count}
    """

    # Word counting.
    word_dict = {}
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line[:-1].split(' '):
                word = word.strip()
                if len(word) == 0:
                    continue

                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1

    # Save word_dict to disk.
    with open(dict_path, 'wb') as fw:
        pickle.dump(word_dict, fw)
```

# 4. 模型测试：分词
## 4.1 加载词典

以下正向、反向以及双向最大匹配算法均需要首先加载词典。

```python
def load_dictionary(dict_path=None):
    with open(dict_path, 'rb') as fr:
        word_dict = pickle.load(fr)
        return word_dict

    return None
```

## 4.2 正向最大匹配算法

从左到右扫描待分词句子，从最大长度开始匹配，如假定最大分词长度为5。

> 这钟声，传递出中华民族从磨难中奋起的昂扬斗志，彰显出伟大民族精神在新时代焕发出的熠熠光辉。

1. 依次判断“这钟声，传”、“这钟声，”、“这钟声”、“这钟”以及“这”是否在词典里，如果有一个在词典里，则在该词语处切分。

2. 如果所有词语都不在词典里，则在第一个字处切分。

3. 往右继续相同切分方式。

最终得到正向最大匹配分词结果：

> 这/钟声/，/传递/出/中华民族/从/磨难/中/奋起/的/昂扬/斗志/，/彰/显出/伟大/民族/精神/在/新时代/ **焕发/出/** 的/熠熠/光辉/。

```python
def forward_maximum_matching(word_dict=None,
                             max_num_char=None,
                             sentence=None):

    words = []
    start = 0

    while start < len(sentence):
        cur_word = None
        for len_word in range(max_num_char, 0, -1):
            maybe_word = sentence[start: start+len_word]
            if maybe_word in word_dict:
                cur_word = maybe_word
                start += len_word
                break

        if cur_word is None:
            cur_word = sentence[start]
            start += 1

        words.append(cur_word)

    return words
```

## 4.3 反向最大匹配算法

一些学者发现的现象是，在中文里，往往作者表达更含蓄，最想要表达的含义在句子的末尾部分；在现实中反向最大匹配算法有时候表现得更好。

反向最大匹配算法从右往左从句子的末尾开始匹配，假定仍然选择最大词语长度为5，

1. 依次判断“熠熠光辉。”、“熠光辉。”、“光辉。”、“辉。”以及“。”是否在词典里，如果在，则在该词语处切分。

2. 如果所有词语都不在词典里，则在最后一个字处切分。

3. 往左继续相同切分方式。

反向最大匹配分词结果：

> 这/钟声/，/传递/出/中华民族/从/磨难/中/奋起/的/昂扬/斗志/，/彰/显出/伟大/民族/精神/在/新时代/ **焕/发出/** 的/熠熠/光辉/。

与正向最大匹配最明显的不同是，反向会把“**焕发出**”分成“焕”与“发出”；而正向分成“焕发”与“出”。方向性非常明显。

```python
def backward_maximum_matching(word_dict=None,
                              max_num_char=None,
                              sentence=None):

    words = []
    start = len(sentence)

    while start > 0:
        cur_word = None
        for len_word in range(max_num_char, 0, -1):
            maybe_word = sentence[max(0, start-len_word): start]
            if maybe_word in word_dict:
                cur_word = maybe_word
                start -= len_word
                break

        if cur_word is None:
            cur_word = sentence[start-1]
            start -= 1

        words.append(cur_word)

    words.reverse()
    return words

```

## 4.4 双向最大匹配算法

双向最大匹配算法同时进行正向与反向最大匹配，从两个结果里启发式地选择最好的一个：

1. 比较两种分词结果分出的词的总数量，选择词语数量少的那一个；如在“中华/民族/真/伟大”与“中华民族/真/伟大”里选择后者，前者有4个词语，后者有3个词语，认为后者对词典利用得更好。

2. 如果两种分词词语数量相等，选择单字个数少的那一个；如在“中国人/真/伟/大”与“中国/人/真/伟大”中选择后者，二者都有4个词语，后者只有2个单字，而前者有3个单字，单字少的意味着两字以上的词语更多，多字词语比单字往往能表达更多含义。

```python
def bidirectional_maximum_matching(word_dict=None,
                                   max_num_char=None,
                                   sentence=None):

    forward_words = forward_maximum_matching(word_dict=word_dict,
                                             max_num_char=max_num_char,
                                             sentence=sentence)

    backward_words = backward_maximum_matching(word_dict=word_dict,
                                              max_num_char=max_num_char,
                                              sentence=sentence)

    """
    1. The less number of words, the better.
    2. If equal number of words, the less number of single character word, the better.
    """

    if len(forward_words) < len(backward_words):
        return forward_words
    elif len(forward_words) > len(backward_words):
        return backward_words
    else:
        if count_single_char(forward_words) < count_single_char(backward_words):
            return forward_words
        else:
            return backward_words

def count_single_char(words):
    cnt = 0
    for word in words:
        if len(word) == 1:
           cnt += 1
    return cnt
```

# 5. 模型评估

## 5.1 对测试文件分词

对测试文件遍历每行分词。

## 5.2 模型评估


### 5.2.1 模型评估计算

同步对比测试与基准数据。

精准率Precision等于测试数据正确分词个数除以测试数据总的分词个数。

> Precision = num_words_correct / num_words_predict

召回率Recall等于测试数据正确分词个数除以基准数据总的词语个数。

> Recall = num_words_correct / num_words_baseline

因此，核心是要计算测试数据正确分词个数。

这里不能直接计算两个数组的交集个数，因为分词结果是有序的。如：

> 测试数据：['大', '家', '都是', '大学生']

> 基准数据：['大家', '都', '是', '大', '学生']

直接考虑交集，会认为'大'是一个正确的分词。然而一个'大'在句子的开头，一个在句子的靠后位置，实际上测试数据一个词也没有切分正确。

在计算交集的时候，必须要考虑词语在句子中的**位置**。

将每一个词在句子中的位置用起始位置表示出来，用python的区间表示方式，左边为闭区间，右边为开区间，则上述数据的位置表示为，

> 测试数据：[(0, 1), (1, 2), (2, 4), (4, 7)]

> 基准数据：[(0, 2), (2, 3), (3, 4), (4, 5), (5, 7)]

得到了互相不重叠的顺序的区间起始集，从而可以转换成集合set来取交集。

```python
def compute_num_intersect(buf_1, buf_2):
    if len(buf_1) == 0 or len(buf_2) == 0:
        return 0

    indexes_1 = to_index(buf_1)
    indexes_2 = to_index(buf_2)

    # Computing intersects between sets, using `&`.
    intersects = indexes_1 & indexes_2

    if intersects is not None:
        return len(intersects)

    return 0

def to_index(buf):
    """
    buf:  word_0, word1, ..., word_{N-1}
    indexes(python style [start, end)):
        [start_0, end_0], [end_0, end_1], ..., [end_{N-2}, end_{N-1}=end]
    """

    indexes = []
    start = 0

    for word in buf:
        end = start + len(word)
        indexes.append((start, end))
        start = end

    return set(indexes)
```

计算起来就比较直接了。

```python
def compute_fscore(y_true_path=None, y_pred_path=None):
    fr_true = open(y_true_path, 'r', encoding='utf-8')
    fr_pred = open(y_pred_path, 'r', encoding='utf-8')

    n_correct = 0
    n_true = 0
    n_pred = 0

    for true_line, pred_line in zip(fr_true, fr_pred):
        true_buf = []
        for word in true_line[:-1].split(' '):
            if len(word) == 0:
                continue
            true_buf.append(word)

        pred_buf = []
        for word in pred_line[:-1].split(' '):
            if len(word) == 0:
                continue
            pred_buf.append(word)

        cur_n_correct = compute_num_intersect(true_buf, pred_buf)

        n_correct += cur_n_correct
        n_true += len(true_buf)
        n_pred += len(pred_buf)

    if n_correct == 0:
        return False

    precision = n_correct / n_pred
    recall = n_correct / n_true
    f1_score = 2 * precision * recall / (precision + recall)

    print("precision\trecall\tf1_score")
    print("%.4f\t\t%.4f\t%.4f" %
          (precision, recall, f1_score))

    fr_true.close()
    fr_pred.close()

    return True
```

### 5.2.2 模型评估结果

|模型|Precision|Recall|F1_score|
|---|---------|------|--------|
|正向最大匹配|0.9120|**0.9541**|**0.9325**|
|反向最大匹配|0.9101|0.9522|0.9306|
|双向最大匹配|**0.9121**|0.9536|0.9324|

注：最大词语长度设置为6。


# 6. 总结

本篇介绍了简单的基于词典匹配的中文分词算法，简单易用。

当然，也因为过于简单，在工业界生产系统上使用效果会不够好；我们需要研究更好的中文分词算法。

# 7. 参考资料

1. jieba分词：https://github.com/fxsjy/jieba

2. HanLP分词：https://github.com/hankcs/HanLP

3. 《自然语言处理入门》，何晗（HanLP作者）。


详细代码见我的[Github](https://github.com/wadefall/chinese_segmentation/tree/master/cangjie/matching)，有问题欢迎一起讨论。


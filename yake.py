import re
from math import log10, sqrt
from typing import List, Optional, Tuple, Union

import pandas as pd
from jieba import cut, posseg


def remove_punctuation(text):
    # 定义一个正则表达式，匹配中文和英文标点符号
    punctuation_pattern = re.compile(
        "[\u3000-\u303F\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFFE0-\uFFE5]|[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]|[\u2026]"
    )

    # 使用正则表达式将匹配的标点符号替换为空字符串
    clean_text = re.sub(punctuation_pattern, "", text)

    return clean_text


def get_word_pos(words_lst, start=0):
    # 分词的位置列表
    # 为避免开头的词必然成为关键词的情形，将全文词位置（索引）从 1 开始计算
    return list(zip(words_lst, range(start, len(words_lst) + 1)))


def get_pos_score(pos_lst, word):
    # 单个词的位置得分, 位置越靠前得分越低，则越重要
    word_positions = [item[1] for item in pos_lst if word in item[0]]
    word_positions.sort()
    num_positions = len(word_positions)
    if num_positions == 0:
        return 0
    half_index = num_positions // 2
    median_position = (word_positions[half_index] + word_positions[~half_index]) / 2

    return log10(log10(median_position + 10)) + 1


def words_count(words_lst):  # 全文词频统计，返回字典
    word_count = {}
    for word in words_lst:
        word_count[word] = word_count.get(word, 0) + 1
    mean_tf = sum(word_count.values()) / len(words_lst)  # 词频均值
    std_tf = std(list(word_count.values()))  # 词频标准差
    max_tf = max(list(word_count.values()))  # 词频最大值
    min_tf = min(list(word_count.values()))  # 词频最小值
    return word_count, mean_tf, std_tf, max_tf, min_tf


def related_content(words_lst, word, windows_size):
    # 单个词 上下文关系 得分
    split_lst = " ".join(words_lst).split(word)
    left_lst = split_lst[:-1]
    right_lst = split_lst[1:]
    DL = 0
    for i in left_lst:  # 从 word 出现的每个地方往左取 windows_size 个词，计算不重复词数
        left_words = i.split(" ")[-2 : -2 - windows_size : -1]
        DL += len(set(left_words))

    DR = 0
    for i in right_lst:  # 从 word 出现的每个地方往右
        right_words = i.split(" ")[1 : windows_size + 1]
        DR += len(set(right_words))

    return DL / len(left_lst) + DR / len(right_lst)


def get_word_sentence(words_lst, split_content):
    # 单个词 句间词频 得分 包含候选词的句子数量 / 总句子数量
    len_content = len(split_content)
    WSpread = {}
    for w in words_lst:
        for sentence in split_content:
            if w in sentence:
                WSpread[w] = WSpread.get(w, 0) + 1
    return [(i[0], i[1] / len_content) for i in list(WSpread.items())]


def get_pseg(x):  # 词性
    return [p for w, p in list(posseg.cut(x))][0]


def std(lst):
    ex = float(sum(lst)) / len(lst)
    s = 0
    for i in lst:
        s += (i - ex) ** 2
    return sqrt(float(s) / len(lst))


def get_yake_score(
    content: str,
    start: int = 0,
    upper_en: bool = True,
    pos_type="sentence",
    tf_normal="yake",
    adjust=1,
    windows_size=10,
    stopwords=None,
):
    """打分函数
    Args:
        content (str): 待提取关键词的文本
        start (int, optional): 词位置起始值, 如果从1开始则打压第0位置的词. Defaults to 0.
        upper_en (bool, optional): 提高英文的权重占比. Defaults to True.
        pos_type (str, optional): 词位置指标计算方式, "sentence" or "word". Defaults to "sentence".
            如果是"word"，则计算全文词位置，如果是"sentence"，则计算含词的句子位置
        tf_normal (str, optional): _description_. Defaults to "yake".
        adjust (int, optional): 分母调整值, 替换原版Yake的大写特征. Defaults to 1.
        r_size (int, optional): 上下文关系指标的窗口大小. Defaults to 10.
        stopwords (set, optional): 停用词表. Defaults to None.
    Returns:
        _type_: _description_
    """
    # 按照 \n\t\r，；。？！,拆分句子并去除标点符号，过滤单字
    split_content = [
        remove_punctuation(i)
        for i in re.split(r"[\n\t\r，；。？！,]", content)
        if len(i) > 1
    ]
    # 整篇文章分词
    clean_str = remove_punctuation(content)  # 去除标点符号

    jieba_words = [w for w in cut(clean_str) if len(w) > 1]  # 分词过滤单字

    if stopwords is not None:  # 停用词
        jieba_words = [w for w in jieba_words if w not in stopwords]

    sorted_ngram_lst = sorted(list(set(jieba_words)), key=jieba_words.index)

    # 位置 pos 得分表
    if pos_type == "word":
        pos_lst = get_word_pos(jieba_words, start)  # 计算全文词位置
    elif pos_type == "sentence":
        pos_lst = get_word_pos(split_content, start)  # 计算含词的句子位置
    else:
        raise ValueError("pos_type must be 'word' or 'sentence'")

    # 计算词位置得分
    word_pos_scores = []
    for word in sorted_ngram_lst:
        word_pos_scores.append((word, get_pos_score(pos_lst, word)))

    # 是否提高英文的权重占比
    if upper_en:
        wrod_sentence_scores = [(word, score * 1.5) for word, score in word_pos_scores]

    # 全文词频 TF_norm 得分表
    word_count, mean_tf, std_tf, max_tf, min_tf = words_count(jieba_words)
    if max_tf - min_tf == 0:
        tf_normal = "yake"
    tf_norm_scores = []
    TF_norm = 0
    for w in sorted_ngram_lst:
        if tf_normal == "yake":
            TF_norm = word_count.get(w) / (mean_tf + std_tf)  # yake版归一化
        if tf_normal == "max-min":
            TF_norm = (word_count.get(w) - min_tf) / (max_tf - min_tf)  # max-min归一化
        tf_norm_scores.append((w, TF_norm))

    # 上下文 word_Rel 得分表
    word_rel_scores = []
    all_words = len(sorted_ngram_lst)
    for w in sorted_ngram_lst:
        DL_RL = related_content(jieba_words, w, windows_size)
        T_Rel = 1 + DL_RL * word_count.get(w) / all_words
        word_rel_scores.append((w, T_Rel))

    # 句间词频 wrod_sentence 得分表
    wrod_sentence_scores = get_word_sentence(sorted_ngram_lst, split_content)

    # 重要性 S_t 总分表
    df_scores = pd.DataFrame(
        {
            "word": sorted_ngram_lst,
            "fre": [word_count.get(i) for i in sorted_ngram_lst],
            "t_pos": [i[1] for i in word_pos_scores],
            "tf_norm": [i[1] for i in tf_norm_scores],
            "t_rel": [i[1] for i in word_rel_scores],
            "t_sentence": [i[1] for i in wrod_sentence_scores],
        }
    )
    df_scores["pseg"] = df_scores["word"].apply(get_pseg)
    df_scores.eval(
        f"yake_socre = t_pos*t_rel / ({adjust} + (tf_norm + t_sentence)/t_rel)",
        inplace=True,
    )
    return df_scores


def get_key_words(df_scores, top=10, ascend=True, p=None):  # 获取关键词列表，默认前10个，升序
    if p is not None:
        df_scores = df_scores[df_scores["pseg"] == p]
    result = df_scores.sort_values("yake_socre", ascending=ascend)
    key_words = result["word"].to_list()
    return key_words[:top]


def get_stopwords(txt_file):
    return set(
        [line.strip() for line in open(txt_file, "r", encoding="utf-8").readlines()]
    )


if __name__ == "__main__":
    df = get_yake_score(
        "Yake 是一种轻量级、无监督的自动关键词提取方法，它依赖于从单个文档中提取的统计文本特征来识别文本中最相关的关键词。该方法不需要针对特定的文档集进行训练，也不依赖于字典、文本大小、领域或语言。Yake 定义了一组五个特征来捕捉关键词特征，这些特征被启发式地组合起来，为每个关键词分配一个分数。分数越低，关键字越重要。你可以阅读原始论文[2]，以及yake 的Python 包[3]关于它的信息。",
        pos_type="sentence",
        start=0,
        upper_en=True,
        tf_normal="yake",
        adjust=0,
        windows_size=5,
        stopwords=get_stopwords(
            "/Users/coma_white55/Downloads/Keyword_extract/yake/yake/StopwordsList/stopwords_zh.txt"
        ),
    )
    res = get_key_words(df, top=10, ascend=True, p=None)
    print(res)

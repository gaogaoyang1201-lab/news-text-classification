import jieba

text = "今天天气很好，我们学习文本分类。"

words = jieba.lcut(text)

print(words)
# LDA


```
import numpy as np

def compute_perplexity(lda_model, corpus):
    total_log_likelihood = 0
    total_num_words = 0

    # 对每个文档进行迭代
    for doc in corpus:
        # 获取文档的主题分布
        gamma, _ = lda_model.inference([doc])
        doc_topic_dist = gamma[0] / sum(gamma[0])

        # 计算文档的对数似然
        doc_log_likelihood = 0
        for word_id, freq in doc:
            
            word_prob = sum(prob for topic_id in range(lda_model.num_topics) for _, prob in lda_model.get_topic_terms(topicid=topic_id, topn=lda_model.num_terms))
            doc_log_likelihood += freq * np.log(word_prob)

        # 更新总的对数似然和词的数量
        total_log_likelihood += doc_log_likelihood
        total_num_words += sum(freq for _, freq in doc)

    # 计算困惑度
    perplexity = np.exp(-total_log_likelihood / total_num_words)

    return perplexity


from sklearn.model_selection import KFold
import numpy as np

def ten_fold_cv(num_topics, corpus, dictionary):
    # 初始化一个KFold对象
    kf = KFold(n_splits=10)

    # 初始化一个空列表来存储每一折的困惑度
    perplexities = []

    # 对每一折进行迭代
    for train_index, test_index in kf.split(corpus):
        # 划分训练集和测试集
        corpus_train = [corpus[i] for i in train_index]
        corpus_test = [corpus[i] for i in test_index]

        # 在训练集上训练LDA模型
        lda = LdaModel(corpus=corpus_train, id2word=dictionary, num_topics=num_topics)

        # 在测试集上计算困惑度，并将其添加到列表中
        # perplexities.append(lda.log_perplexity(corpus_test))
        perplexities.append(compute_perplexity(lda, corpus_test))

    # 计算十折交叉验证的平均困惑度
    return np.mean(perplexities), np.std(perplexities)


from gensim.models import LdaModel, CoherenceModel


perplex_dic = {}
coherence_dic = {}
for num_topics in range(2, 10):
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    # 计算困惑度
    perplex_dic[num_topics] = ten_fold_cv(num_topics, corpus, dictionary)
    print(num_topics, 'perplexity: ', perplex_dic[num_topics])
    # coherence_model_lda = CoherenceModel(model=lda, texts=high_freq_words, dictionary=dictionary, coherence='c_v')
    # coherence_dic[num_topics] = coherence_model_lda.get_coherence()
    # print(num_topics, 'coherence: ', coherence_dic[num_topics])

print(perplex_dic)
print(coherence_dic)


2 perplexity:  (0.4999999947753177, 1.0250355075788412e-08)
3 perplexity:  (0.3333333317410108, 8.298618570414164e-09)
4 perplexity:  (0.2499999951628192, 4.103560935333037e-09)
5 perplexity:  (0.1999999976819312, 3.0922105530728517e-09)
6 perplexity:  (0.16666666435155314, 2.162914601993435e-09)
7 perplexity:  (0.1428571420692047, 1.500098020159798e-09)
8 perplexity:  (0.12499999724238386, 2.25718780390517e-09)
9 perplexity:  (0.11111110921460605, 1.4465539286993876e-09)
{2: (0.4999999947753177, 1.0250355075788412e-08), 3: (0.3333333317410108, 8.298618570414164e-09), 4: (0.2499999951628192, 4.103560935333037e-09), 5: (0.1999999976819312, 3.0922105530728517e-09), 6: (0.16666666435155314, 2.162914601993435e-09), 7: (0.1428571420692047, 1.500098020159798e-09), 8: (0.12499999724238386, 2.25718780390517e-09), 9: (0.11111110921460605, 1.4465539286993876e-09)}
{}

# plot perplexity for different number of topics and save to file
import matplotlib.pyplot as plt
x = list(perplex_dic.keys())
y = list(perplex_dic.values())
plt.plot(x, y)
plt.xlabel("Number of topics")
plt.ylabel("Perplexity score")
plt.savefig('perplexity_score.png')
```
from jieba import analyse
#引入TextRank关键词抽取接口
textrank = analyse.textrank
#文本
text = "度：节点的度是指和该节点相关联的边的条数，又称关联度，表明了节点的影响能力。\
        特别地，对于有向图，节点的的度又分为入度和出度，\
        分别表示指向该节点的边的条数和从该节点出发的边的条数。这就好比越是重要城市，\
        与周边城市的铁路连接率越高。\
        接近中心性：每个结点到其它结点的最短路的平均长度。\
        对于一个结点而言，它距离其它结点越近，那么它的中心度越高。\
        比如一个城市的中心地带往往是出行方便，离各个公共设施比较近的。"
#对文本进行TextRank算法处理
keywords = textrank(text)
#按重要性输出抽取出的关键词
print("\nkeywords by textrank:")
for keyword in keywords:
    print(keyword)
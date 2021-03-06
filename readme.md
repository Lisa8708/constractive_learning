## 对比学习
主要包括文本表示相关的算法和文本相似、文本匹配模型

## 模型
- sentence_bert: 参考:https://www.sbert.net/docs/training/overview.html
- simcse: 参考苏神的文章:https://kexue.fm/archives/8348
    - 无监督的tf版本来源苏神的代码:https://github.com/bojone/SimCSE
    - 有监督的torch版本来源:
- simbert: 基于微软的UniLM的思想设计的融检索与生成为一体的任务，参考苏神的文章:https://spaces.ac.cn/archives/7427
- CoSENT: 一个优化cos值的新方案，参考苏神的文章:https://spaces.ac.cn/archives/8847

## 数据
- ATEC: https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC (重新划分了train、valid和test)
- BQ: http://icrc.hitsz.edu.cn/info/1037/1162.htm
- LCQMC: http://icrc.hitsz.edu.cn/Article/show/171.html
- PAWSX: https://arxiv.org/abs/1908.11828 (只保留了中文部分)
- STS-B: https://github.com/pluto-junzeng/CNSD

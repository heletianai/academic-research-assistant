# 项目1（RAG+Agent）Why 追问卡片

> 面试官听完项目介绍后最常追问的10个 Why。每题先看"面试直接说"，被追问再看"追问备忘"。

---

## 1. 为什么用这个 Embedding 模型？对比过哪些？

**面试直接说：**
用的 `text-embedding-3-small`，通过 OpenRouter 调用。选它是因为性价比高——1536维，MTEB 排名靠前，价格是 large 版的 1/6。语料是英文论文，OpenAI 的模型在英文上本身就强。而且 Embedding 只是粗排的一半，后面还有 CrossEncoder 精排兜底，所以粗排阶段不需要最好的模型，够用就行。

**追问备忘：**
- 写死在 `settings.py` 的 `EMBEDDING_MODEL = "openai/text-embedding-3-small"`
- **没有做过 Embedding 模型的 A/B 实验**，诚实说"对比实验主要做在检索策略上（4组），Embedding 模型没有换着跑过"
- 如果追问本地模型：可以换 BGE-M3 或 GTE 省 API 费，但当前规模没必要

---

## 2. 为什么 chunk size 选 512？试过其他的吗？

**面试直接说：**
512 字符是经典中间值。太小（128）一个 chunk 信息量不够回答问题；太大（1024）一个 chunk 混入多个主题，检索相关性下降。512 在学术论文场景大约覆盖 1-2 个段落，一个完整知识点。overlap 50 字符（约10%）防止切分边界截断关键信息。切分策略是段落优先——先按双换行分段，合并到不超过512，超长才强制滑窗。

**追问备忘：**
- **没有做过不同 chunk size 的对比实验**，诚实说"512 是参考业界常用值直接设的，如果要优化会用评估脚本跑 256/512/1024 的 MRR 对比"
- 过滤了长度 < 20 的碎片 chunk
- 超长段落 `_force_split` 用滑窗：每次取512，前进 512-50=462 字符

---

## 3. 检索不准的时候怎么处理的？

**面试直接说：**
用了 CRAG 机制。检索完不直接生成，LLM 先对每篇文档做三元评分：relevant、ambiguous、irrelevant。超过一半 relevant 直接生成；ambiguous 就改写 query 重新检索，最多3次；全 irrelevant 或重试耗尽，降级到 DuckDuckGo Web Search。实测：第一次检索 0/3 relevant → ambiguous → 改写 → 第二次 2/3 relevant → 成功生成。

**追问备忘：**
- 评分在 `grader.py`，路由在 `rag_subgraph.py` 的 `route_by_grade()`
- 改写会把前2篇文档的中间300字喂给 LLM 辅助改写
- Web Search 也失败就降级到 LLM 自身知识（三层兜底）
- 阈值50%是 CRAG 原论文思路，3篇文档意味着至少2篇 relevant 才直接生成

---

## 4. 为什么用 FAISS？和其他方案对比过吗？

**面试直接说：**
选 FAISS 是因为轻量，pip install 就能用，不需要额外部署服务。数据量小（6篇论文，几百个 chunk），IndexFlatIP 暴力搜索毫秒级就够了。如果要上百万级文档，会考虑 Milvus 或 Qdrant。

**追问备忘：**
- 用的 `IndexFlatIP`（内积）+ `normalize_L2`（归一化后内积=余弦相似度）
- 支持持久化：`faiss.index` + `documents.json`，第二次启动直接加载
- **没有对比过其他向量数据库**，诚实说
- FAISS 缺点：不支持动态增删（要重建索引）、不支持 metadata filtering、没有分布式能力

---

## 5. Reranker 用了吗？为什么用？

**面试直接说：**
用了 `cross-encoder/ms-marco-MiniLM-L-6-v2`，80MB 的轻量 Cross-Encoder，本地 CPU 推理。粗排用 Hybrid 取 top-10，Reranker 精排取 top-3。原因是 RRF 融合分数只反映排名，不反映 query 和 doc 的真正语义匹配度。Cross-Encoder 把 query 和 doc 拼在一起做交叉注意力，精度更高。实测加了 Reranker 后 MRR 从 0.87 提升到 1.0。

**追问备忘：**
- 10条候选精排约 0.3 秒（M系列 Mac）
- 如果追问"为什么不用 bge-reranker"：ms-marco-MiniLM 是经典模型，英文场景两者差距不大
- `TOP_K_RETRIEVAL = 10, TOP_K_RERANK = 3`

---

## 6. Prompt 怎么设计的？system/user prompt 怎么分的？

**面试直接说：**
7个 prompt 集中管理在 `prompts.py`。四个设计原则：格式引导（"分类结果："引导直接输出标签）、few-shot 覆盖边界 case、不同任务不同 temperature（分类0/生成0.1/改写0.3/闲聊0.7）、RAG 生成严格限制"只用上下文信息"防幻觉。

**追问备忘：**
- **大部分调用没有用 system/user 分离**，全放在 user message 里。诚实说"DeepSeek 通过 OpenRouter 调用，实测 system 和 user 效果差距不大。如果要优化应该把角色定位放 system、具体任务放 user"
- 只有 Web Search 兜底节点用了 system message

---

## 7. MRR 为什么是 1.0？测试集多大？

**面试直接说：**
测试集只有 10 个 case，命中判定是关键词匹配——文档内容包含 expected_keywords 中的任意一个就算命中。MRR=1.0 是因为测试集太小、问题和论文高度对齐，比如问"什么是 self-attention"对应的论文就是 Attention Is All You Need，top-1 必然命中。**这个 1.0 不能代表真实场景效果**，需要扩大到几百条、加入 hard negative 和跨文档问题才有说服力。

**追问备忘：**
- 面试时**主动暴露缺陷**："10 条 case + 关键词匹配，说明的不是系统多好，而是测试不够严格"
- 生成评估用 LLM-as-Judge 双维度：Faithfulness（防幻觉）+ Relevancy（防跑题）
- 局限：Judge 和 Generator 是同一个 DeepSeek，存在 self-bias

---

## 8. 遇到的最大的坑是什么？

**面试直接说：**
三个坑。最折腾的是 **OpenMP 冲突**——Mac 上 PyTorch 和 FAISS 链接不同版本 OpenMP，启动直接 crash，设 `KMP_DUPLICATE_LIB_OK=TRUE` 解决。第二个是 **.env 没被加载**，API key 为空，所有 LLM 调用静默失败，排查了半天。第三个是 **retriever 为 None**——论文目录为空时 retriever 初始化失败，下游 reranker 拿到空输入报错。加了判空逻辑，retriever 为 None 时返回空结果，CRAG 自动走 Web Search 兜底。

**追问备忘：**
- OpenMP 问题是 Mac 经典坑，Linux 生产环境不会有
- retriever 判空体现**防御性编程**——组件失败时优雅降级而不是崩溃

---

## 9. 如果重新做，你会改什么？

**面试直接说：**
三个方面。第一，**评估体系先建**——现在10条测试集太粗糙，应该先构建几百条人工标注的 ground truth，有靠谱评估才能驱动优化。第二，**chunk 策略改成语义切分**——现在按字符数切不考虑语义边界，应该按论文 section/paragraph 结构切。第三，**加 metadata filtering**——论文有年份、作者、会议等结构化属性，检索时可以先缩小范围再做语义搜索，现在完全没利用。

**追问备忘：**
- 代码里没有 semantic chunking，也没有 metadata filtering（FAISS IndexFlatIP 本身不支持）
- 如果追问"为什么当时没做"：时间有限，先跑通核心链路，这些是有评估基座后的迭代方向

---

## 10. 这个项目有什么明显的局限性？

**面试直接说：**
四个。第一，**数据量太小**，6篇论文不具备真实挑战性。第二，**评估不充分**，10条case+关键词匹配无法反映真实效果。第三，**没有工程化**——没有 API 鉴权、并发处理、监控告警、A/B 测试。第四，**LLM 调用成本高**——CRAG 每篇文档单独调 LLM 评分，一次问答可能调5-6次 API。

**追问备忘：**
- 如果追问"哪个最重要"：**评估体系**。没有靠谱的评估，所有优化都是盲打
- 中文支持弱——BM25 分词按字拆分没用 jieba
- 没有缓存机制，相同 query 重复调 API

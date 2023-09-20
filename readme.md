# CausalLM-Lit

集成了 Lightning 了 trl 框架，用来训练 Causal Language Model，实现监督微调与强化学习过程。

我们发现如果把模型结构确定为 Causal Language Model，即与目前大语言模型相同结构的情况下，对模型的改动都很小，大部分情况下只需要设计好数据的文本输入。因为，我们把主要的精力放到数据的处理上面，用于多种多样的 NLP 任务。这样的设计不仅适合 NLPer 新手，也适合那些经常涉及多种 NLP 任务的研究人员。

主要的优点包括：

1. 通用的模型架构，大部分的需要调整的部分都可通过命令行参数或者配置文件修改。
2. 支持全量微调与 PEFT 微调（目前只支持 LoRA），包括模型加载与保存。
3. 通用的数据处理方式，对于各种数据形式只需要较少配置就可以适配。 
4. 集成了强化学习框架 trl，实现完整的大语言模型的微调过程。

在上面优点的加持下，目前实现了下面的 NLP 任务：

1. 分类任务：使用 GPT2 和 LLaMA 实现新闻内容分类，GPT2 的分类准确率与 Roberta 类似，F1-score在 71% 左右，LLaMA（LoRA）的 F1-score 为 75%。
2. 抽取任务：使用 ChatGLM 实现文档级事件抽取任务。
3. 生成任务：使用 LLaMA 实现根据 Caption 实现 VQA 的任务，准确率接近 BLIP 模型。
4. 控制生成：使用强化学习过程，实现 GPT2 和 LLaMA 正面影评生成。

## 模块
下面介绍主要的可配置模型，通过这些模块的配置或者自定义，能完成大部分的 NLP 任务。

### 数据处理

数据处理包含指令形式的通用的数据，也可以自定义数据格式。
数据集的设置都是通过配置文件进行定义，例如新闻分类的配置文件为：

```yaml
datamodule:
  class_name: news.NewsDM #指定数据的类
  args:
    data_dir: data/news   #设定数据文件夹的位置

metrics:                  #声明是一个需要计算指标的数据集
  name: f1                #验证的指标为 f1，用来确定较优的模型保存参数以及 early stoping
  eval_calculate: true    #声明在验证的时候进行计算指标
```

#### 通用数据（指令数据格式）
通用的指令数据包含三个部分：指令、输入、输出，目前使用的是 Alpaca 的指令格式，即：
```python
"Below is an instruction that describes a task, paired with an input that provides further context. "
"Write a response that appropriately completes the request.\n\n"
"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
```
如果数据比较符合这种形式，那预处理数据转换成包含instruction、input、output的形式，即类似于：
```json
{
    "instruction": "Is this a reasoning question or a visual question?",
    "input": "Context: the two baseball players and ump are ready. Question: What player is ranked highest in this sport?",
    "output": "reasoning"
}
```
这样再通过覆盖参数（后面会详细介绍）的方法，就可以训练起来自己的数据。

#### 自定义数据类型
自定义数据包含两个部分：  
1. 数据导入
2. 数据处理

我们使用的 Lightning 的框架，它预设了几个完成整个优化过程的阶段（stage），例如优化（fit）、测试（test）、预测（predict），为了减少不必要的数据加载，可以通过判断所处的状态来加载不同的数据，例如：
```python
# 摘录正面影评生成数据: datas.imdb.PositiveIMDbDM
def setup(self, stage: str):
    if stage == pl.trainer.states.TrainerFn.FITTING:
        train_data = load_dataset(self.data_dir, split='train')
        val_data = load_dataset(self.data_dir, split='test')
        train_data = [self.convert(x) for x in train_data if x["label"]==1]
        ...

    elif stage == pl.trainer.states.TrainerFn.TESTING:
        test_data = load_dataset(self.data_dir, split='test')
        ...
```
另外建议在自定义数据的时候实现 convert 方法，用来将单条数据转换为输入输出形式，即：
```python
{
    "input": "...",  #语言模型的输入
    "output": "...", #期待模型的输出结果
    "origin": {...}  #可选部分，可以用来计算评价指标
}
```

### 模型定义
模型的定义与是通过配置文件，这样的好处是可以通过指定不同的配置文件就能完成基座模型的切换。一个完整的模型配置文件示例：
```yaml
pretrain:
  path: /vault/pretrains/huggingface/huggyllama/llama-7b           #预训练参数
  args:
    trust_remote_code: true                                        #初始化模型的参数

tokenizer:
  manual_add_eos_token: true                                       #分词器的一些特殊行为

config:
  use_cache: false                                                 #generation 时的参数

quant:                                                             #进行量化，下面是量化的参数
  bf16: true
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"

peft:                                                              #peft 微调，下面是 LoRA 的参数配置
  lora_alpha: 16
  lora_dropout: 0.05
  r: 8
  bias: none

optim:                                                             #优化器的配置
  lr: 1e-4
```


### 参数覆盖
大部分的修改都是通过参数修改的，影响参数有以下几个方面：

1. 命令行参数
2. 数据配置文件
3. 模型配置文件
4. 命令行参数覆盖

其中前三点比较常见或者已经介绍，下面介绍命令行参数覆盖。

命令行参数覆盖的目标是通地命令行来修改数据和模型的配置参数，这样的好处比较灵活，适合用来进行超搜索。  
例如，通用的指令数据形式往往只需要修改数据的存储路径，其格式为：
```shell
--D.datamodule.args.data_dir=...
```
这样就可以在不修改配置文件的情况下，修改数据的存储路径。
而对于修改模型的学习率这样可以进行超参搜索的配置，其格式为：
```shell
--M.optim.lr=...
```
可以看出，覆盖的时候需要指定是数据覆盖（`--D.`）还是模型覆盖（`--M.`）,后面的参与则为配置文件的 key 路径，中间用`.`分隔。


### 强化学习奖励模型

我们使用的是 trl 框架，需要奖励模型输出一批数据中（batch）每条数据的奖励值，每个奖励值是个单值 tensor 并组成一个列表。  
例如，生成正面影评的奖励函数为：
```python
class PositiveSentimentRewardModule():
    def __init__(self) -> None:
        self.sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 16
        }

        self.pipeline = pipeline("sentiment-analysis","/pretrains/huggingface/lvwerra/distilbert-imdb")
    
    def __call__(self, output):
        texts = []
        for prompt, pred, _ in output:
            pred = pred.strip()
            texts.append(prompt + " " + pred)
        pipe_outputs = self.pipeline(texts, **self.sent_kwargs)                  # 分类模型输出 confidence
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]  # 转化为奖励列表
        return rewards, {}
```

## TODO
- [ ] 强化学习分布式训练


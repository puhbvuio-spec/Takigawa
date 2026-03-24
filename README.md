# Community Feedback Bundle

## 标签库命名与存放位置
- 推荐文件名：`star_rail_feedback_labels.v3.0.0.json`
- 推荐存放目录：`label_libraries/`
- 在当前包中的完整路径：`label_libraries/star_rail_feedback_labels.v3.0.0.json`

脚本默认优先读取上面这一路径；也可以通过命令行 `--label-library` 传入其他版本标签库。

## 输入文件示例
- 示例输入：`data/inbox/sample_feedback_input.csv`

### 表头说明
| 字段名            | 是否必需 | 含义                                | 运行时是否使用 |
| ----------------- | -------- | ----------------------------------- | -------------- |
| `record_id`       | 建议提供 | 记录唯一 ID；不提供时脚本会自动生成 | 是             |
| `node_id`         | 是       | 观测节点 / 业务聚合节点             | 是             |
| `source_platform` | 是       | 来源平台，如 NGA / Weibo / Bilibili | 是             |
| `publish_time`    | 是       | 发表时间，建议 ISO 格式             | 是             |
| `post_title`      | 是       | 评论所属帖子标题；仅作补充语境      | 是             |
| `post_tag`        | 是       | 评论所属帖子标签；仅作补充语境      | 是             |
| `raw_text`        | 是       | 评论正文，标签判别的主依据          | 是             |
| `source_url`      | 否       | 源链接                              | 是             |
| `user_id`         | 否       | 用户 ID                             | 是             |
| `case_note`       | 否       | 仅用于示例讲解，不参与程序判别      | 否             |

### 关于 `post_title` 与 `post_tag` 的使用原则
- 评论正文 `raw_text` 是主判别依据。
- `post_title`、`post_tag`、`source_platform` 只能作为补充语境，帮助理解极短文本、代词化文本或信息不完整文本。
- 当正文已经足够完成判别时，必须以正文为准，不能让标题或标签反客为主。
- 示例文件故意加入了“标题标签与正文张冠李戴”的样本，用于验证清洗与稳健性，而不是模拟理想数据集。

### 示例文件中的案例备注
`sample_feedback_input.csv` 额外提供 `case_note` 列，专门说明每一行样本想验证的能力，例如：
- 正文即可完成多标签判别
- 垃圾文本清洗
- 近重复合并
- 标题 / 标签只能补充、不能主导
- 短文本补语境

这列不会投入程序运行；当前脚本会把它视为普通扩展字段并忽略。

## 运行方式
- 规则模式：双击 `run_rule_mode.bat`
- LLM 模式：双击 `启动.bat`

### LLM 模式的 API KEY 要求
当前仅支持 **通义千问 Coding Plan** 套餐的 API Key（以 `sk-sp-` 开头）。

- 获取地址：[百炼控制台](https://bailian.console.aliyun.com/)
- API 接口：`https://coding.dashscope.aliyuncs.com/v1/chat/completions`
- 默认模型：`qwen3.5-plus`

`启动.bat` 运行时会提示输入 API Key，并校验前缀必须为 `sk-sp-`，不符合则拒绝运行。

> **注意：** 其他类型的 DashScope API Key（如 `sk-` 开头的标准套餐密钥）未经测试，暂不提供支持。如确有需要，可在 `PipelineConfig` 中手动指定 `api_base_url`。

### qwen3 系列兼容说明
脚本已内置以下适配逻辑，确保与 qwen3 系列模型兼容：
- 请求中设置 `enable_thinking: false`，关闭深度思考模式以获得纯 JSON 输出
- 设置 `presence_penalty: 0.5`，减少重复输出
- 响应中自动清除 `<think>...</think>` 标签块后再解析 JSON
- 区分 429 限流与其他错误码，自动避让重试（最多 5 次）

## 输出目录
每次运行后会在 `data/work/job_xxx/output/` 下生成：
- `assets/`：机器资产 JSON
- `reports/`：面向操作者的报表
  - `statistics_rows.csv`：标签统计明细
  - `review_pool.csv`：人工审查候选池
  - `node_summary.csv`：节点汇总
  - `preprocess_overview.csv`：预处理概览
  - `dashboard.html`：可视化分析看板

### 可视化看板（dashboard.html）
管道运行结束后会自动生成一份 HTML 可视化看板，浏览器直接打开即可查看，无需额外安装依赖（图表通过 ECharts CDN 加载）。

看板包含以下模块：
| 模块               | 内容                                                        |
| ------------------ | ----------------------------------------------------------- |
| KPI 卡片           | 总声量、NSR 净情感值、情感结构条（正/中/负占比）、舆情判断  |
| 数据质量概览       | 清洗后总量、垃圾/无效数、近似重复数、模板内容数             |
| 负向 Pareto 图     | 负向标签集中度柱状图 + 累计占比折线                         |
| 治理归因饼图       | 按标签库一级维度自动归因（剧情叙事、角色设计、玩法与战斗…） |
| 正/负向标签 Top 10 | 水平条形图，标签名使用 `一级维度/二级维度` 可读格式         |
| 节点命中分布       | 各观测节点的标签命中量条形图                                |
| 来源平台分布       | 饼图展示各平台的数据占比                                    |
| 代表性原声         | 正/负向各最多 6 条原始评论，附标签标注、平台、时间          |

> 看板生成为非阻塞步骤：即使生成失败，也不会影响主流程的 JSON/CSV 产出。

## 示例成果展示
仓库提供了一份实际运行后的可视化看板：

- **[example_output/dashboard.html](example_output/dashboard.html)**

下载后用浏览器直接打开即可体验完整的分析看板效果（ECharts 图表通过 CDN 加载，需联网）。

## 依赖
仓库已提供 `requirements.txt`，可直接执行：

```bash
python -m pip install -r requirements.txt
```

核心依赖：
- `pandas`
- `aiohttp`
- `openpyxl`

如需启用本地 TF-IDF 检索增强，额外建议安装：
- `scikit-learn`
- `scipy`

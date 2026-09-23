# AI 象棋棋谱来源与纳入规则

核查日期：2026-09-23。这里的“AI 棋谱”是**完整、可逐步重放的象棋对局**。单个 FEN、网络权重、分析标签、国际象棋 PGN 和人类对局均不计入 AI 对局数。互联网不存在可验证的“全部 AI 象棋棋谱”索引；来源持续增补，每批数据单独记录许可、格式、哈希和校验结果。

| 来源 | 核查结果 | 本次处理 |
| --- | --- | --- |
| 本项目生成的 Pikafish 自对弈 | 使用[官方 Pikafish](https://github.com/official-pikafish/Pikafish)二进制，按种子选择开局并记录全局走法、终止状态；二进制不进入仓库。 | 纳入 `data/ai-games-160.jsonl`：160 局、17,667 步，其中 104 局自然终局、56 局达到 140 步上限。相应逐局面分析见 `data/teacher-games-160.jsonl`：16,387 个局面。两份文件由 `verify-games.js` 交叉校验。Pikafish 仅作为外部教师/评测者。 |
| [Pika Xiangqi Zero / Px0](https://www.kaggle.com/datasets/pikacat/px0data) | [Pikafish 官方 README](https://github.com/official-pikafish/Pikafish#acknowledgements)称训练数据为 ODbL；Kaggle 当前 API 却返回 `licenseName: Unknown`，约 11 GB，文件格式未核实。 | 保留为待核查来源；未把训练 chunk 当成完整棋谱，也未下载整个数据集。 |
| [Xiangqi-R1 完整对局数据](https://huggingface.co/datasets/hoduyquocbao/xiangqi-r1-master-dataset) | 数据卡称 Apache-2.0、约 519 MB，以多轮消息记录对局；本环境的文件 API 返回 401，尚未抽样验证合法性。 | 待取得可访问文件并做逐步重放后再纳入。 |
| [Xiangqi Gen6 / NNUE 自对弈数据](https://huggingface.co/datasets/hoduyquocbao/xiangqi-gen6-platinum-dataset) | 数据卡称 MIT，主要宣称局面和权重；卡片自报体量与页面体量不同，尚未核实是否有完整对局。 | 暂不计入棋谱；后续逐文件检查。 |
| [AI Agent Arcade](https://github.com/linxule/arcade) | MIT 项目含 AI 智能体的象棋对局归档；本次未取得并核验实际棋谱文件。 | 待导入器能识别其格式后按合法性与去重规则筛选。 |
| [CCPD](https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset) | CC BY 4.0 的人类大师棋谱；本项目已有 738 条可用开局线。 | 只用作开局多样性，不计作 AI 棋谱。 |

## 纳入规则

1. 每局保存起始 FEN、按 ICCS 坐标记法排列的全部走法、开局前缀、最终 FEN、胜负或截断原因。截断局不冒充胜负。
2. 逐步验证走法合法、最终局面与终止状态；对应的教师局面须和完整棋谱的同一步及实际走法相符。重复棋谱拒收。
3. 数据集哈希、引擎二进制哈希、开局库哈希和随机种子随数据保存。公开下载源还须保留来源 URL、许可与原文件哈希。
4. 训练划分按整局，并在不同分组间清除相同局面；未来增加开局家族隔离。强棋力标签优先使用搜索更深的教师分析，50 ms 的快速自对弈主要用于扩大状态覆盖，不作为可靠胜率标签。

复现本批数据（需自备官方 Pikafish 可执行文件与其 NNUE 文件）：

```sh
node teacher.js --pikafish /absolute/path/to/Pikafish --openings data/openings.json --opening-plies 8 --games 160 --maxplies 140 --movetime 50 --seed 20260924 --output data/teacher-games-160.jsonl --games-output data/ai-games-160.jsonl
node verify-data.js data/teacher-games-160.jsonl
node verify-games.js data/ai-games-160.jsonl data/teacher-games-160.jsonl
```

校验与合并统计见 [数据报告](reports/data-v2.json)。新旧两个教师文件合计 21,387 条分析记录，跨源去重后有 20,901 个不同局面。训练器现在可重复传入 `--data`，并把不同来源的同号对局隔离；当前固定种子划分为训练 14,221、验证 2,304、校准 2,202、测试 2,174 条，跨分组相同局面为 0。开局家族级隔离尚未实现，因此这些数字只证明逐局与逐局面的隔离。

长将判罚修复后，逐局重放将先前误标为重复和棋的 19 局改判为长将方负；没有棋局在新的终局点之后继续走子。原始教师分析只接收 FEN，不知道之前的重复历史，因此靠近循环终点的 `best` 标签不应被解释为遵守长将规则的最佳招。训练带历史输入的模型前，应排除这类局面或根据完整棋谱重标。

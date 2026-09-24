# AI 象棋棋谱来源与纳入规则

核查日期：2026-09-23。这里的“AI 棋谱”是**完整、可逐步重放的象棋对局**。单个 FEN、网络权重、分析标签、国际象棋 PGN 和人类对局均不计入 AI 对局数。互联网不存在可验证的“全部 AI 象棋棋谱”索引；来源持续增补，每批数据单独记录许可、格式、哈希和校验结果。

| 来源 | 核查结果 | 本次处理 |
| --- | --- | --- |
| 本项目生成的 Pikafish 自对弈 | 使用[官方 Pikafish](https://github.com/official-pikafish/Pikafish)二进制，按种子选择开局并记录全局走法、终止状态；二进制不进入仓库。 | 纳入 `data/ai-games-160.jsonl`：160 局、17,667 步，其中 104 局自然终局、56 局达到 140 步上限。相应逐局面分析见 `data/teacher-games-160.jsonl`：16,387 个局面。两份文件由 `verify-games.js` 交叉校验。Pikafish 仅作为外部教师/评测者。 |
| 本项目 Jev 模型自对弈 | 当前正式走法模型与自研分阶段搜索控制红黑双方；从已验证开局变化开始，固定随机种子记录探索选择。 | `data/selfplay-sample-games.jsonl` 含 4 局流程样例、304 个半回合；`data/selfplay-sample-positions.jsonl` 含开局后 272 个局面及完整模型概率、搜索分布和剪枝统计。2 局将死、1 局三次重复、1 局达到上限并标为截断。样例用于验证管线，不作为棋力结论。 |
| [Pika Xiangqi Zero / Px0](https://www.kaggle.com/datasets/pikacat/px0data) | [Pikafish 官方 README](https://github.com/official-pikafish/Pikafish#acknowledgements)称训练数据为 ODbL；Kaggle 当前 API 却返回 `licenseName: Unknown`，约 11 GB，文件格式未核实。 | 保留为待核查来源；未把训练 chunk 当成完整棋谱，也未下载整个数据集。 |
| [Xiangqi-R1 完整对局数据](https://huggingface.co/datasets/hoduyquocbao/xiangqi-r1-master-dataset) | 数据卡称 Apache-2.0、约 519 MB，以多轮消息记录对局；本环境的文件 API 返回 401，尚未抽样验证合法性。 | 待取得可访问文件并做逐步重放后再纳入。 |
| [Xiangqi Gen6 / NNUE 自对弈数据](https://huggingface.co/datasets/hoduyquocbao/xiangqi-gen6-platinum-dataset) | 数据卡称 MIT，主要宣称局面和权重；卡片自报体量与页面体量不同，尚未核实是否有完整对局。 | 暂不计入棋谱；后续逐文件检查。 |
| [AI Agent Arcade](https://github.com/linxule/arcade) | MIT 项目含 AI 智能体的象棋对局归档；本次未取得并核验实际棋谱文件。 | 待导入器能识别其格式后按合法性与去重规则筛选。 |
| [CCPD](https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset) | Yu-Han Tseng 与 Bo-Nian Chen 的 CC BY 4.0 棋谱库；本项目已有 738 条可用开局线。另核查 208 份「電腦對局」PGN。 | 新增 `data/ccpd-computer-games.jsonl`：196 局可重放、20,112 半回合，其中 155 局标为电脑对局竞赛、41 局标为人机赛。5 份重复、7 份未通过导入；101 局原文件结果为 `*`，保留为未知结果。人机赛单独标记，不当作纯 AI 自对弈。 |

CCPD 的「大师对局／以开局分类」另外提供 6,963 份 PGN。本项目核对前 24 个半回合，去掉 409 份重复，得到 6,554 局和 51,009 个开局局面，汇成对弈时使用的 [大师开局库](reports/master-opening-book.json)。这些是**人类大师棋谱**，不计入上表的 AI 棋谱数；ECCO C 类仅用于标识中炮对屏风马体系，库中走法的频次不是胜率。

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

CCPD 电脑对局的[导入报告](reports/ccpd-computer-import.json)记录原仓库提交、208 份源文件的整体哈希、每份失败或重复的原因；每局也保留分类、来源文件及其 SHA-256。原始导入文件只提供**走法和原 PGN 结果**。它有 18,059 个不同局面，其中 17,999 个未出现在已有两份教师文件中，且中残局占多数；`*` 结果不能充当和棋或胜负标签。复现导入（按报告中的源提交检出 CCPD）：

```sh
node ccpd-games.js --input /absolute/path/to/CCPD/Dataset/對局/電腦對局 --source-commit 368a47a947773dd8692c026e286dd19b6277b993 --compare-teacher data/teacher-openings-5000.jsonl --compare-teacher data/teacher-games-160.jsonl
node verify-ccpd-games.js data/ccpd-computer-games.jsonl
```

已从 154 局「電腦對局競賽」中均衡抽取 1,200 个新局面，按开局、中局、残局比例 25/40/35 分布，使用外部 Pikafish 每局面 500 ms、多候选重新分析。`data/teacher-ccpd-competition-1200.jsonl` 保留原棋谱、局号、步数与实际走法；它不是原棋谱的胜负标签。所有局面与原棋谱逐步对应，且与旧教师集没有重复。标注代码能在中断后通过 `--resume` 续跑；采样、深度和哈希见[教师数据报告](reports/teacher-ccpd-competition-1200.json)。复现：

```sh
node label-ccpd-games.js --pikafish /absolute/path/to/Pikafish --positions 1200 --movetime 500 --exclude-teacher data/teacher-openings-5000.jsonl --exclude-teacher data/teacher-games-160.jsonl --output data/teacher-ccpd-competition-1200.jsonl
node verify-data.js data/teacher-ccpd-competition-1200.jsonl
node verify-ccpd-teacher.js data/teacher-ccpd-competition-1200.jsonl data/ccpd-computer-games.jsonl data/teacher-openings-5000.jsonl data/teacher-games-160.jsonl
```

长将判罚修复后，逐局重放将先前误标为重复和棋的 19 局改判为长将方负；没有棋局在新的终局点之后继续走子。原始教师分析只接收 FEN，不知道之前的重复历史，因此靠近循环终点的 `best` 标签不应被解释为遵守长将规则的最佳招。训练带历史输入的模型前，应排除这类局面或根据完整棋谱重标。

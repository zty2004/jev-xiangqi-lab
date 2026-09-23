# 中国象棋 AI 实验室

目标是训练**自有的 Jev 式象棋决策模型**，无需 TypeSafe Jev API 或 Key。当前交付合法走法、自研限时搜索、局面到全部合法走法的概率模型、UCI 接口、本地网页、实战开局库和与外部 Pikafish 对弈的评测脚本。模型定义与训练路线见 [DESIGN.md](DESIGN.md)。

## 本地运行

要求 Node.js 20 或更新版本，无需安装 npm 依赖。

```sh
npm test
npm start
```

打开 `http://127.0.0.1:3000`。网页支持人机对弈、选择红黑方、思考时间、悔棋和 FEN 导入；选择黑方时棋盘会翻转，黑方显示在下方。服务器只监听本机地址。

网页的**教学模式**允许你用鼠标代走红黑双方，并随时选择红方或黑方在下。每步之后，自研引擎对当前局面分析，按设置标注前 1–5 招；数字和箭头对应推荐排名，旁边列出搜索评分。你可以选择任意合法走法，推荐不会自动落子。设置 `CHOICE_MODEL` 启动服务器后，推荐列表还会显示本地模型给各招的概率：

```sh
CHOICE_MODEL=/absolute/path/to/models/choice-openings-5000.pt npm start
```

## 后端接口

`node uci.js` 启动 UCI 棋引擎，支持 `uci`、`isready`、`ucinewgame`、`position startpos/fen ... moves ...`、`go movetime N`、`go depth N`、`go perft N` 和 `quit`。

网页服务提供 `GET /api/new`、`POST /api/state`、`POST /api/move`、`POST /api/ai`。`/api/move` 可提交中文棋谱走法，例如 `炮二平五`；返回值、网页走棋记录和引擎推荐均提供中文棋谱。鼠标落子、UCI 和训练文件内部仍采用 ICCS 坐标，以兼容 FEN 与引擎协议。

搜索采用迭代加深、Alpha-Beta、置换表、吃子优先的走法排序与吃子延伸。设置 `CHOICE_MODEL` 后，模型先给全部合法走法分配概率；搜索用概率安排根节点的搜索顺序，再用搜索分数决定最终走法。Pikafish 不参与正式选招，只用作数据教师和外部评测对手。

将军判断现直接检查将帅是否受到攻击。与上一版相比，三个固定局面的完整 4 层搜索在节点数、评分和首选走法相同的条件下，用时中位数缩短约 2.43–3.37 倍；1 秒限时下，开局和一个中局局面均多完成 1 层搜索。验证范围与原始测量见[搜索优化报告](reports/search-check-optimization.json)。这些速度结果尚不能证明对局胜率提高。

随后缓存了攻击检查所需的棋子符号；相同固定深度下，三个局面又快了约 1.24–1.31 倍，原始测量见[第二次搜索优化报告](reports/search-check-piece-cache.json)。

第一项加速配合当前正式模型的[每步 5 秒成对评测](benchmark-choice-5000-fastcheck-openings-5s.jsonl)仍对 Pikafish 得 0/2；这项评测使用的规则代码哈希已写入对局记录，与第二项微优化区分。

## 与 Pikafish 评测

下载 [Pikafish 官方发布包](https://github.com/official-pikafish/Pikafish/releases)，将可执行文件与 `pikafish.nnue` 放在同一目录。运行：

```sh
npm run benchmark -- --pikafish /absolute/path/to/Pikafish-MacOS-universal --games 2 --movetime 5000 --output benchmark-results.jsonl
```

评测会在同一台机器上依次给双方每步 5 秒，使用固定开局并交换红黑方。结果写入 JSONL，包含每局走法、结果、引擎文件哈希、开局和用时设置。正式比较棋力时应增大偶数对局数，固定引擎版本、硬件、线程数与开局集，并报告胜率及不确定性。

运行中若引擎或对手超时，评测脚本写入 `aborted` 记录并停止，**不会把运行故障算成对局负局**。`--start-game 1 --games 2` 可单独补跑第二局（从零编号），输出到新的文件。`node summarize-benchmark.js --input games.jsonl --output report.json` 重放棋谱、核对终局、生成中文记谱并将旧格式中的运行故障从成绩中剔除。

使用导入的开局库固定评测起点：

```sh
node benchmark.js --pikafish /absolute/path/to/Pikafish --openings data/openings.json --opening-plies 8 --games 20 --movetime 5000 --output benchmark-openings.jsonl
```

2026-09-23 的流程验证在相同引擎代码、相同 8 步开局和每步 5 秒下各跑了 2 局：纯搜索基线 0/2，本地选择模型引导搜索也是 0/2，均被 Pikafish 将死。双方对局走法有变化，但这些样本不足以证明模型提高棋力。记录见 `benchmark-baseline-openings-5s.jsonl` 与 `benchmark-choice-openings-5s.jsonl`。

5,000 局面训练后的新模型也在相同开局、每步 5 秒、红黑轮换的 2 局中得 0/2，记录见 `benchmark-choice-5000-openings-5s.jsonl`。目前没有证据表明实战棋力已提高；更不能据此声称接近或超过 Pikafish。

当前裁判实现支持将死、困毙、单方长将判负、普通重复判和及 60 回合无吃子/走兵；**长捉、将一捉等复杂判罚尚未实现**。因此当前自动对局结果是研发基准，不能作为赛事规则下的正式等级分。

## 本地选择模型训练

当前提供一条可复现的教师蒸馏原型。`teacher.js` 让外部 Pikafish 自对弈并保存局面、所有合法走法、最佳走法与最多 8 个候选评分。`train/choice_model.py` 训练一个卷积网络，对每个局面的全部合法走法输出概率。按整局拆分训练、选模型、校准和测试，剔除跨组重复局面；校准集只用于拟合温度，最终测试集只报告结果。模型输出的 `concentration` 是本项目定义的概率分布集中度，不是 TypeSafe Jev 的 confidence 公式，也不是最高招的概率。

选招并非只看下一手：搜索引擎使用逐层加深的多步搜索，并在叶节点继续检查吃子和将军变化。教学模式用 MultiPV 搜索前 1–5 招，展示每招的后续推演与实际完成深度；默认思考时间为 5 秒，可选 10 或 20 秒。模型目前只给根局面的合法走法提供先验，搜索树内部还没有逐节点模型评估，这是后续网络引导搜索的重点。界面的“深度”是搜索层数（半回合数），不是完整回合数。

重复局面不再一律判和：若一方在重复循环的每个己方回合都将军，第三次出现同一局面时判该长将方负；普通重复仍判和。终局判断与搜索评分共用此规则。此版本尚未覆盖世界象棋联合会规则中更复杂的长捉、将一捉等情况，参见[世界象棋规则第 20 条](https://www.wxf-xiangqi.org/images/wxf-rules/2018_World_XiangQi_Rules_English2018.pdf)。

另从 CCPD 的电脑相关棋谱中逐步校验并导入 196 局，保留电脑对局与人机赛分类；101 局原 PGN 未给胜负，不用于胜负监督。详情见[CCPD 导入报告](reports/ccpd-computer-import.json)。其中从 154 局电脑竞赛棋谱抽取 1,200 个不与旧教师集重复的局面，以 Pikafish 每局面 500 ms、多候选方式重新分析；全量重放校验与标签统计见[新教师数据报告](reports/teacher-ccpd-competition-1200.json)。这批标签尚未证明新模型的实战棋力。

AI 棋谱来源、纳入规则与 160 局新生成的完整自对弈棋谱见 [数据来源](DATA_SOURCES.md) 和 [数据报告](reports/data-v2.json)。新旧教师文件可以一起训练：

```sh
python3 train/choice_model.py train --data data/teacher-openings-5000.jsonl --data data/teacher-games-160.jsonl --opening-data data/opening-positions.jsonl --opening-samples 2000 --output models/choice-v2.pt --epochs 10 --batch 128 --channels 64 --blocks 4 --device cpu
```

`dash-dev-191-root` 的 GPU 0 训练实验使用 [GPU 固定脚本](train/run_gpu_21k.sh)，权重为 `models/choice-21k-gpu0.pt`，完整配置、哈希与评测见[训练报告](reports/choice-21k-gpu0.json)。在排除旧教师文件和开局库局面后的 1,973 个新测试局面上，新模型与教师最佳招一致率 20.48%，旧模型 19.16%；新模型 NLL 2.8645，旧模型 2.9169。这只说明教师模仿指标改善。新模型在相同 8 步开局、每步 5 秒、红黑轮换的 [2 局评测](benchmark-choice-21k-openings-5s.jsonl)中得 0/2，均被 Pikafish 将死；样本太少，不能据此估计稳定棋力。当前网页继续使用旧模型。

下一版实验让模型同时读取前两个盘面、半回合计数与当前局面的重复次数，并用有终局结果的对局训练胜/和/负头。两套 GPU 0 候选权重与训练配置见[历史模型报告](reports/history-wdl-gpu0.json)。混合数据版本在同一留出集的教师最佳招一致率为 20.42%，旧模型为 20.88%；胜负头的 NLL 为 0.781，差于训练类别频率基线的 0.765。在传入完整走棋历史后，该模型的[每步 5 秒成对评测](benchmark-choice-history-mixed-openings-5s.jsonl)对 Pikafish 得 0/2。两套新权重仍为实验候选，网页继续使用旧模型。

教师标签审计发现，约 19% 的局面中最终最佳招与保存的候选评分首位不同。把软标签中最终最佳招的显式权重从 0.3 提高到 0.7 后，[新候选模型](reports/choice-best-dominant-gpu0.json)在同一留出集的最佳招一致率从 20.88% 升至 22.17%，NLL 从 2.854 降至 2.786。按 22 局重采样的 NLL 改善区间为 0.038–0.098；最佳招一致率改善区间仍跨过零。它在[每步 5 秒成对评测](benchmark-choice-best-dominant-openings-5s.jsonl)中对 Pikafish 得 0/2，因此新权重暂未替换网页模型。

把 1,200 个重新分析的 CCPD 局面加入同一训练流程后，[GPU 0 新候选](reports/choice-ccpd1200-gpu0.json)在仅包含这批来源的 15 局、118 个留出局面上，[对比旧候选](reports/compare-ccpd1200-gpu0.json)的最佳招一致率由 29.66% 降至 28.81%，NLL 由 2.650 升至 2.737。局数有限，且尚未进行实战评测；这版权重保留作实验记录，不替换网页模型。

继续把新来源在训练采样中的权重提高 4 倍，也未改善同一 15 局测试：[加权模型报告](reports/choice-ccpd-weighted-gpu0.json)与[成对对比](reports/compare-ccpd-weighted-gpu0.json)显示，一致率从未加权版本的 28.81% 降至 23.73%，NLL 从 2.737 升至 2.800。加权权重不启用；当前优先改善开局选招与更可靠的数据划分。

```sh
mkdir -p data models
node teacher.js --pikafish /absolute/path/to/Pikafish-MacOS-universal --positions 1000 --movetime 100 --output data/teacher-1000.jsonl
python3 train/choice_model.py train --data data/teacher-1000.jsonl --opening-data data/opening-positions.jsonl --opening-samples 2000 --output models/choice-openings-3000.pt --epochs 8
CHOICE_MODEL=/absolute/path/to/models/choice-openings-3000.pt node uci.js
```

概率和数据隔离测试：`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_choice_model.py`。输出的 `top1` 是与限时 Pikafish 最佳招一致的比例，`nll`、多类别 `brier` 和 `ece10` 评估这个教师选招事件的概率质量；这些指标不等于棋力或胜率。

训练代码需要 PyTorch（依赖见 `requirements.txt`）。进一步的本机实验使用 5,000 个教师局面和最多 2,000 个开局局面，模型文件为 `models/choice-openings-5000.pt`，数据与指标见 [实验报告](reports/choice-openings-5000.json)。在剔除旧教师数据及全部开局库中出现过的局面后，新旧模型在同一批 599 个局面上与教师最佳招的一致率分别为 21.0% 和 16.5%。这只是监督学习指标，尚不足以说明实战棋力提高。新模型经校准后的测试集对数损失和 ECE 略高于校准前，报告同时保留两组数值。

复现本轮训练：

```sh
node teacher.js --pikafish /absolute/path/to/Pikafish --openings data/openings.json --opening-plies 8 --positions 5000 --movetime 100 --output data/teacher-openings-5000.jsonl
node verify-data.js data/teacher-openings-5000.jsonl
python3 train/choice_model.py train --data data/teacher-openings-5000.jsonl --opening-data data/opening-positions.jsonl --opening-samples 2000 --output models/choice-openings-5000.pt --epochs 10 --batch 128 --channels 64 --blocks 4 --device cpu
```

## 开局库

已将 [Chinese Chess Practical Dataset (CCPD)](https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset) 的 961 条开局 PGN 转换并逐步校验为坐标走法：738 条不同变化可用，193 条与已有变化重复，30 条未通过导入条件（其中 24 条不是标准初始局面）。得到 9,084 个不同开局局面。原始数据由 Yu-Han Tseng 与 Bo-Nian Chen（2026）提供，采用 CC BY 4.0；导出文件保留来源、许可、来源文件名和原始文件哈希。

实际对弈现使用同一数据集按开局分类的 6,963 份大师对局 PGN：去掉 409 份重复后，6,554 局前 24 个半回合全部合法，汇成 51,009 个局面、56,835 个合法候选招。每招保留大师对局出现次数、ECCO C 类（中炮对屏风马）出现次数和原棋谱示例。黑方遇到中炮时优先走 C 类的主流变化；目前“炮二平五”后稳定走“马８进７”，镜像开局稳定走“马２进３”。[ECCO 2004 分类说明](https://www.xqbase.com/ecco/ecco_intro.htm)只定义开局类别，不证明某招必然最优；本书的频次也不是胜率。来源、失败项和校验见[大师开局库报告](reports/master-opening-book.json)。

对弈时从至少 5 局大师对局支持的走法中，保留出现次数达到最多走法四分之一的前 3 招，再让本地选择模型排序并由自研搜索决定；无合格书招则正常搜索。最长使用到第 24 个半回合。中炮第一应手“马８进７”与后续屏风马体系的两种常见变化经[限时 Pikafish 分析](reports/opening-response-check.json)交叉检查。网页的走棋记录、推荐和后续推演显示中文记谱。UCI 通信和训练文件继续使用坐标协议。可用 `OPENING_BOOK=off` 关闭对弈开局库以做对照实验。

另从同一大师库导出 [1,970 个训练局面](data/master-opening-positions.jsonl)：仅保留至少 5 局支持的局面，目标走法概率取自正式对弈时启用的开局候选及其大师对局频次，包含中炮对屏风马的分类约束。这个文件是监督学习输入；仅生成文件不会改变当前模型权重或证明棋力提高。

GPU 0 上完成一组同配置对照：候选模型加入其中 1,500 个大师开局训练局面，对照模型只使用相同教师数据。排除训练数据、教师数据和旧开局数据后，247 个开局留出局面的常见招匹配率从对照的 26.72% 升至 32.39%，对大师走法分布的交叉熵从 2.768 降至 2.339；旧网页模型在同一留出集为 27.53% 和 2.710。另在训练集中查看中炮第一应手，候选模型首选从对照的“炮８平５”变为“马８进７”；该例本身不作为泛化证据。[训练报告](reports/choice-master-opening-gpu0.json)与[三模型对照](reports/compare-master-opening-gpu0.json)保留权重哈希、数据划分和指标。这证明候选模型学到了部分开局分布，但还没有实战胜率证据；网页继续使用旧模型，正式对弈的开局库仍负责约束前 24 个半回合。

首次每步 5 秒实战验证的[原始记录](benchmark-choice-master-opening-5s-interrupted.jsonl)中，候选执红一局在 86 个半回合后被将死；执黑一局在第 17 个半回合后因引擎响应超时而中断。中文棋谱与有效成绩见[重放报告](reports/benchmark-choice-master-opening-interrupted.json)；第二局的运行故障不计为负局，尚不能据此比较模型棋力。

重新导入或生成训练局面：

```sh
node openings.js --input /absolute/path/to/CCPD/Dataset/開局 --output data/openings.json --maxplies 24
node book-positions.js --openings data/openings.json --output data/opening-positions.jsonl
node teacher.js --pikafish /absolute/path/to/Pikafish --openings data/openings.json --opening-plies 8 --positions 1000 --movetime 100 --output data/teacher-openings.jsonl
node verify-data.js data/teacher-openings.jsonl
node master-opening-book.js --input /absolute/path/to/CCPD/Dataset/對局/大師對局/以開局分類 --source-commit 368a47a947773dd8692c026e286dd19b6277b993
node verify-master-opening-book.js
node master-book-positions.js --book data/master-opening-book.json --output data/master-opening-positions.jsonl
```

## 后续训练方向

训练分成两步：先用已许可的棋谱或自对弈局面采样，用 Pikafish 为候选招生成教师评分，监督训练本地走法模型；再通过自对弈强化学习优化胜率。模型始终输出当前局面的全部合法走法分布。模型概率用于指导自研搜索，最终由搜索分数选择走法。

Pikafish 仅作为教师数据生成器与外部评测对手，训练数据与评测对局必须分开，避免测试集泄漏。超过 Pikafish 是长期实验目标；是否实现以固定条件下的对局胜率为准。

规则节点数以[公开的中国象棋 perft 结果](https://talkchess.com/viewtopic.php?t=85880)校验；测试覆盖开局及多个战术局面。

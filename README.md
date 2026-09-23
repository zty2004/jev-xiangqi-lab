# 中国象棋 AI 实验室

目标是训练**自有的 Jev 式象棋决策模型**，无需 TypeSafe Jev API 或 Key。当前交付合法走法、自研限时搜索、局面到全部合法走法的概率模型、UCI 接口、本地网页、开局棋谱导入和与外部 Pikafish 对弈的评测脚本。模型定义与训练路线见 [DESIGN.md](DESIGN.md)。

## 本地运行

要求 Node.js 20 或更新版本，无需安装 npm 依赖。

```sh
npm test
npm start
```

打开 `http://127.0.0.1:3000`。网页支持人机对弈、选择红黑方、思考时间、悔棋和 FEN 导入。服务器只监听本机地址。

## 后端接口

`node uci.js` 启动 UCI 棋引擎，支持 `uci`、`isready`、`ucinewgame`、`position startpos/fen ... moves ...`、`go movetime N`、`go depth N`、`go perft N` 和 `quit`。

网页服务提供 `GET /api/new`、`POST /api/state`、`POST /api/move`、`POST /api/ai`。走法采用 ICCS 坐标，例如 `b2e2`；FEN 与 Pikafish 使用的常见中国象棋格式兼容。

搜索采用迭代加深、Alpha-Beta、置换表、吃子优先的走法排序与吃子延伸。设置 `CHOICE_MODEL` 后，模型先给全部合法走法分配概率；搜索用概率安排根节点的搜索顺序，再用搜索分数决定最终走法。Pikafish 不参与正式选招，只用作数据教师和外部评测对手。

## 与 Pikafish 评测

下载 [Pikafish 官方发布包](https://github.com/official-pikafish/Pikafish/releases)，将可执行文件与 `pikafish.nnue` 放在同一目录。运行：

```sh
npm run benchmark -- --pikafish /absolute/path/to/Pikafish-MacOS-universal --games 2 --movetime 5000 --output benchmark-results.jsonl
```

评测会在同一台机器上依次给双方每步 5 秒，使用固定开局并交换红黑方。结果写入 JSONL，包含每局走法、结果、引擎文件哈希、开局和用时设置。正式比较棋力时应增大偶数对局数，固定引擎版本、硬件、线程数与开局集，并报告胜率及不确定性。

使用导入的开局库固定评测起点：

```sh
node benchmark.js --pikafish /absolute/path/to/Pikafish --openings data/openings.json --opening-plies 8 --games 20 --movetime 5000 --output benchmark-openings.jsonl
```

2026-09-23 的流程验证在相同引擎代码、相同 8 步开局和每步 5 秒下各跑了 2 局：纯搜索基线 0/2，本地选择模型引导搜索也是 0/2，均被 Pikafish 将死。双方对局走法有变化，但这些样本不足以证明模型提高棋力。记录见 `benchmark-baseline-openings-5s.jsonl` 与 `benchmark-choice-openings-5s.jsonl`。

当前裁判实现支持将死、困毙、三次重复及 60 回合无吃子/走兵，但**尚未实现比赛规则中的长将、长捉判罚**。因此当前自动对局结果是研发基准，不能作为赛事规则下的正式等级分。

## 本地选择模型训练

当前提供一条可复现的教师蒸馏原型。`teacher.js` 让外部 Pikafish 自对弈并保存局面、所有合法走法、最佳走法与最多 8 个候选评分。`train/choice_model.py` 训练一个卷积网络，对每个局面的全部合法走法输出概率。按整局拆分训练和验证，并排除验证集中出现过的训练局面。

```sh
mkdir -p data models
node teacher.js --pikafish /absolute/path/to/Pikafish-MacOS-universal --positions 1000 --movetime 100 --output data/teacher-1000.jsonl
python3 train/choice_model.py train --data data/teacher-1000.jsonl --opening-data data/opening-positions.jsonl --opening-samples 2000 --output models/choice-openings-3000.pt --epochs 8
CHOICE_MODEL=/absolute/path/to/models/choice-openings-3000.pt node uci.js
```

训练代码需要 PyTorch（依赖见 `requirements.txt`）。当前 1,000 个教师局面和 2,000 个开局局面的模型只是流程验证，远不足以达到强引擎水平。

## 开局库

已将 [Chinese Chess Practical Dataset (CCPD)](https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset) 的 961 条开局 PGN 转换并逐步校验为坐标走法：738 条不同变化可用，193 条与已有变化重复，30 条未通过导入条件（其中 24 条不是标准初始局面）。得到 9,084 个不同开局局面。原始数据由 Yu-Han Tseng 与 Bo-Nian Chen（2026）提供，采用 CC BY 4.0；导出文件保留来源、许可、来源文件名和原始文件哈希。

重新导入或生成训练局面：

```sh
node openings.js --input /absolute/path/to/CCPD/Dataset/開局 --output data/openings.json --maxplies 24
node book-positions.js --openings data/openings.json --output data/opening-positions.jsonl
node teacher.js --pikafish /absolute/path/to/Pikafish --openings data/openings.json --opening-plies 8 --positions 1000 --movetime 100 --output data/teacher-openings.jsonl
```

## 后续训练方向

训练分成两步：先用已许可的棋谱或自对弈局面采样，用 Pikafish 为候选招生成教师评分，监督训练本地走法模型；再通过自对弈强化学习优化胜率。模型始终输出当前局面的全部合法走法分布。模型概率用于指导自研搜索，最终由搜索分数选择走法。

Pikafish 仅作为教师数据生成器与外部评测对手，训练数据与评测对局必须分开，避免测试集泄漏。超过 Pikafish 是长期实验目标；是否实现以固定条件下的对局胜率为准。

规则节点数以[公开的中国象棋 perft 结果](https://talkchess.com/viewtopic.php?t=85880)校验；测试覆盖开局及多个战术局面。

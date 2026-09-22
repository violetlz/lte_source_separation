# 新实验平台的项目组织与文件架构

日期：2026-09-22。状态：建议架构，尚未创建项目仓库、迁移代码或启动训练。

## 1. 建议：并列新建目录，独立维护项目

建议新项目命名为 `lte_source_separation`，Python 包名为 `rfsep`。在本地与现有 `source_separation` 目录并列，采用独立 Git 历史；未来发布到 GitHub 时使用独立仓库。当前无需为了讨论架构创建远程仓库。

旧仓库继续保存 RRC-QPSK/OFDM-BPSK 复现、LTE Oracle 逐 RE 实验和已有结果。新项目围绕已确认的 LTE 场景启发波形基准建立唯一主链路。旧代码按功能选择性迁入，并记录来源；新项目的运行环境不依赖旧目录。

本次核对远程目录树的提交仍为 `a39a2fc2312fc6b5e4cdf0d1962a92f711ce627f`，与此前审查一致。旧项目目前同时包含顶层 `model.py/dataset.py/learner.py`、多个实验入口、`source_separation/lte_oracle_*`、MATLAB 工程与历史结果；把新平台继续接入这些入口，容易混用旧配置、旧尺度和逐 RE 数据路径。

选择独立目录是对这次实验边界变化的工程判断，不代表旧 RGS 算法或所有代码都错误。新平台允许重复利用已验证的实现，避免继承不适用的默认假设。

## 2. 文件架构

下面是最终目标结构。标有“后续”的文件在对应研究阶段才实现，第一阶段不创建伪实现，不用返回固定值或空函数冒充支持。目录中省略各子包的常规 `__init__.py`。

```text
lte_source_separation/
    README.md
    AGENTS.md
    pyproject.toml
    requirements.lock
    .gitignore
    THIRD_PARTY.md

    docs/
        platform_design.md
        data_contract.md
        experiment_protocol.md
        migration_notes.md
        progress.md

    configs/
        platform/
            lte_v1.yaml
        training/
            target_diffusion.yaml
            interference_diffusion.yaml
            interference_fm.yaml          # 后续
        experiments/
            smoke.yaml
            noiseless_rgs.yaml
            middle_sir.yaml
            awgn.yaml                     # 后续固定噪声子集
            fm_comparison.yaml            # 后续

    src/
        rfsep/
            __init__.py
            __main__.py
            cli.py
            config.py
            contracts.py
            randomness.py
            observation.py

            waveforms/
                constellations.py
                ofdm.py
                sources.py
                receiver.py

            data/
                manifests.py
                datasets.py

            models/
                diffwave.py

            priors/
                interfaces.py
                diffusion.py
                gaussian.py
                flow_matching.py         # 后续

            training/
                trainer.py
                objectives.py
                checkpoints.py

            solvers/
                rgs.py
                initialization.py
                flow_solver.py           # 后续

            baselines/
                direct.py
                lmmse.py
                sic.py

            diagnostics/
                oracle_re.py
                prior_quality.py
                waveform_audit.py

            evaluation/
                runner.py
                metrics.py
                statistics.py
                profiling.py
                plotting.py

    tests/
        conftest.py
        test_waveforms.py
        test_power.py
        test_data_contract.py
        test_randomness.py
        test_score_conventions.py
        test_rgs.py
        test_metrics.py
        test_noiseless_pipeline.py
        fixtures/
            reference_manifest.json

    reference/
        matlab/
            generate_reference.m
            README.md

    reports/
        README.md
        platform_validation.md

    artifacts/                            # 本地运行产物，不提交大文件
        manifests/
        calibration/
        datasets/
        checkpoints/
        runs/

    notebooks/                            # 按分析需要增加，不作为正式入口
```

`requirements.lock` 在实际环境安装、核验后生成，不能现在凭空填写“已验证”的依赖版本。`notebooks/` 和后续 FM 文件不必在第一天全部创建。

## 3. 各层职责

| 层 | 关键文件 | 职责与边界 |
|---|---|---|
| 项目约定 | `README.md`、`AGENTS.md` | 安装和运行方式、固定研究范围、Codex 必须遵守的实验规则 |
| 平台配置 | `configs/platform/lte_v1.yaml` | 源分布、采样、CP、归一化、接收信息的唯一配置来源 |
| 配置读取 | `config.py` | 类型/范围检查、解析配置、计算哈希；训练/实验配置引用平台配置，禁止默默覆盖源分布 |
| 类型契约 | `contracts.py` | 定义观测、公开接收信息、评分真值和分离输出，隔离真值 |
| 随机流 | `randomness.py` | 按样本 ID 和用途生成种子，避免 batch 大小、顺序改变样本 |
| 信号生成 | `waveforms/` | 星座映射、OFDM/CP、目标与干扰源、连续生成与有效截取 |
| 接收与比特 | `waveforms/receiver.py` | 唯一的目标去 CP、FFT、频点抽取及比特判决实现 |
| 观测模型 | `observation.py` | 接受时域源，执行固定功率校准和时域混合；后续加入声明的 AWGN |
| 数据 | `data/` | 长记录划分、单源训练数据、测试观测/真值分离存储、manifest 重建 |
| 网络 | `models/diffwave.py` | 网络结构，不包含 SIR、混合、BER 或 LTE 生成逻辑 |
| 先验 | `priors/` | 把模型输出解释为 noise/score/velocity；处理坐标与概率路径转换 |
| 训练 | `training/` | 训练目标、优化器、验证、恢复；每次只训练一个独立源 |
| 推断 | `solvers/` | 使用观测与先验恢复波形，不知道真实源、比特、τ、φ |
| 传统方法 | `baselines/` | Direct、LMMSE、从观测估计参数的 SIC，与学习方法共用接收和评分 |
| 诊断 | `diagnostics/` | 解析逐 RE Oracle、先验质量及波形审计；明确与正式方法分开 |
| 评价 | `evaluation/` | 同样本比较、BER/NMSE/EVM、配对统计、统一计时与制图 |
| 独立参考 | `reference/matlab/` | 生成少量参考样本；主 Python 链路不要求 MATLAB 常驻 |
| 运行产物 | `artifacts/` | 自动生成，可移到服务器数据盘；通过配置指定根路径 |
| 可发布结果 | `reports/` | 验收报告和选择性保存的结果摘要，含来源运行 ID |

区分 `models` 与 `priors` 尤其重要：同一个网络骨干可以预测噪声或速度，概率意义由训练目标与先验适配器决定。以后引入 FM 时，不应复制一整套生成器、解调器和统计代码。

## 4. 四条主调用路径

| 路径 | 调用关系 |
|---|---|
| 单源训练 | 配置 → 独立源 dataset → 网络/训练目标 → 先验 checkpoint |
| 构造混合 | 两个独立源 → 时域 observation → 接收可见输入与独立真值 |
| 分离与评分 | 接收可见输入 → RGS/基线 → 目标 receiver → evaluator 读取真值计分 |
| 后续 FM | 复用原数据与 receiver，只更换先验适配器、推断器或概率路径 |

主推断接口只接受 `observation`、`receiver_info`、先验、预算和随机流。评价器另行持有 `truth`。不把两者塞进一个包含所有字段的大字典，再寄希望于推断代码“自觉不读”。

`receiver_info` 只包含方案允许公开的参数。真实干扰 τ、φ、调制分配图不能伪装成配置条件。Oracle 诊断入口可以读取这些字段，但其结果必须标记 Oracle。

不让 `solvers/rgs.py` 从旧仓库导入 `lte_oracle_dataset.py`，也不修改 `sys.path` 引用相邻旧工程。可复用函数迁入后要适配新契约，并记录来源提交。

## 5. 旧代码如何迁入

| 旧资产 | 新归属 | 迁入方式 |
|---|---|---|
| `model.py` 的 DiffWave 骨干 | `models/diffwave.py` | 选择性迁入；核对通道、长度、感受野和时间嵌入 |
| `learner.py` 中训练与恢复逻辑 | `training/` | 提取可复用逻辑，重接单源 dataset 与统一坐标 |
| 原始波形 RGS 的更新逻辑 | `solvers/rgs.py` | 对照论文核验，使用时域状态与新先验接口 |
| 星座映射、OFDM 基础代码 | `waveforms/` | 数值核对后迁入，消除旧固定参数与隐藏缩放 |
| 当前 LTE 解析 score | `diagnostics/oracle_re.py` | 仅用于解析 fixture/Oracle；不能替代主波形先验 |
| `ber_statistics.py` | `evaluation/statistics.py` | 核验记录级配对统计、样本对应和边界条件 |
| MATLAB 生成代码 | `reference/matlab/` | 提取少量参考生成与核验功能 |
| 旧配置、数据加载器和混合入口 | 留在旧仓库 | 根据新数据契约重写对应模块 |
| 旧 checkpoint 和结果 | 留在旧产物目录 | 新项目只引用来源；不伪装为新源分布上的资产 |

`THIRD_PARTY.md` 与 `migration_notes.md` 记录迁入模块的来源、提交和改动，并保留已有版权/许可声明。不要把“新建项目”理解为抹掉代码来源。

## 6. 第一阶段的实现范围

### 交付 A：生成与接收闭环

先实现配置、契约、随机流、波形、时域观测、Direct、接收器和必要测试。完成单源零误码、CP 检查、频点/尺度审计、Q_b 校准、划分与种子检查，输出 `platform_validation.md`。

这是新平台的第一个可运行版本，无需先启动两个长训练任务。初期可只保存重建 manifest 与少量审计样本。

### 交付 B：独立先验训练

迁入 DiffWave 与训练框架，实现目标/干扰两份训练配置、分等级去噪评估和恢复训练。训练样本始终来自已冻结的主源分布；先导 checkpoint 可以继续训练。

### 交付 C：无噪声 RGS 与公平评估

实现时域 RGS、LMMSE、解析诊断和正式 evaluator，加入初始化/预算对照。传统 SIC 可以在先导训练期间补齐，但正式方法优越性结论前须完成核对。所有预设 SIR 点运行并保留，不按测试表现删点。

### 交付 D：后续研究接口

在 C 完成后再实现 FM 先验、FM 融合方法和固定 AWGN 子集。当前只定义所需协议，不引入复杂插件系统、调度服务、网页看板或完整 LTE 协议栈。

## 7. 统一命令入口与产物

建议最终只维护一个 Python 命令入口，而非为每次试验增加独立脚本。以下是待实现接口示例，不是当前已可执行的命令：

```bash
python -m rfsep audit --config configs/platform/lte_v1.yaml
python -m rfsep calibrate --config configs/platform/lte_v1.yaml
python -m rfsep train --config configs/training/target_diffusion.yaml
python -m rfsep train --config configs/training/interference_diffusion.yaml
python -m rfsep evaluate --config configs/experiments/noiseless_rgs.yaml
python -m rfsep report --run <run_id>
```

训练/实验配置显式引用平台配置。每次运行写入 `resolved_config.yaml`，并记录平台配置哈希；校准和模型不匹配时直接报错，不自动采用其他实验默认参数。

每个 `artifacts/runs/<run_id>/` 至少包含：

| 文件 | 内容 |
|---|---|
| `resolved_config.yaml` | 实际生效的完整配置 |
| `run_manifest.json` | 代码提交/未提交改动标识、环境、种子、源版本、checkpoint 与校准哈希 |
| `per_example.csv` | 每例比特数、误码数、指标、耗时、推断种子与失败状态 |
| `summary.csv` | 各条件汇总和置信区间 |
| `figures/` | BER-SIR、BER-计算预算及诊断图 |
| `run.log` | 运行日志与错误 |

真实波形、模型权重、大量日志不提交 Git。配置、实现、必要的小型 fixture、文档和正式结果摘要纳入版本控制。测试 fixture 可由固定 manifest 生成；只有独立 MATLAB 参考确有必要时才提交少量参考数组。

## 8. 建议写入 AGENTS.md 的项目约定

1. 先读平台设计、数据契约和实验协议；默认固定研究边界。
2. 源分布配置唯一；改变其物理含义必须升版本，并说明哪些模型/校准需要重新验证。
3. 主链路只接受时域混合与时域分离；逐 RE 解析实验限于诊断。
4. 目标/干扰独立训练；训练不得混入测试记录或混合真值监督。
5. 接收可见信息与评分真值用不同结构；正式推断不得访问真值。
6. 复用统一解调器、功率定义和种子规则；不得为不同算法各写一套评分链路。
7. 不静默删去失败样本或不利 SIR；运行超时/NaN 应显式记入结果。
8. 昂贵训练按已确认预算执行；先通过链路验收及先导评估，再扩大已授权的计算任务。
9. 后续 FM/AWGN 功能未实现时明确报不支持，不返回假结果。
10. 每个阶段更新 `docs/progress.md`，说明已验证内容、剩余问题及下一项具体工作。

## 9. 给 Codex 的第一项工作

第一项任务应限定为：在新项目目录搭建交付 A，迁入平台设计文档，完成可复现的时域生成—混合—目标解调—评分闭环和对应验收报告。此时先不启动完整模型训练，也不实现 FM 方法。

后续沿用这个唯一主入口逐层接入训练与 RGS。这样每一步都有可检查的成果，两个长训练任务建立在已经验证的数据链路上。

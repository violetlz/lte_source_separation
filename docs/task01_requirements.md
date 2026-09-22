# Codex 执行任务 01：搭建 LTE 场景启发平台的生成与接收闭环

请直接执行下面的开发任务：检查工作环境、建立项目、编写代码、运行必要验证、修复问题，并交付可复现的验收报告。不要只回复设计思路、伪代码或待办清单。

本文件已包含完成任务所需的核心约定。若同时收到 `LTE_Inspired_Waveform_Separation_Platform_Design.md` 和 `LTE_Platform_Project_Architecture.md`，先阅读与本阶段有关的部分。附件没有提供时，按本文件继续工作，不要因此停工。本文件明确限定本次任务范围；后续平台功能不在本次实现范围内。

## 1. 背景、目标与本次范围

研究主线是：独立学习通信源分布，将生成先验嵌入通信接收算法，最终降低目标信号 BER。未来先用两个 Diffusion 波形先验与 α-RGS 建立基线，再研究 Flow Matching 的融合方式与概率路径。

平台名称：**LTE 场景启发的 5/20 MHz 异步 OFDM 波形源分离基准**。

当前任务仅完成“交付 A：生成与接收闭环”：

1. 独立新项目骨架、配置校验与数据契约。
2. 5 MHz 目标和 20 MHz 干扰两个独立源生成器。
3. 固定集合功率归一化、独立 SIR 校准和时域混合。
4. 唯一的目标接收/比特判决实现，以及 Direct 接收结果。
5. 单源零误码、CP、频点、尺度、随机流、数据划分和信息隔离的验证。
6. 可运行的命令入口、机器可读结果和 `reports/platform_validation.md`。

本次不实现或运行：Diffusion/FM 模型、模型训练、RGS、SIC、LMMSE、解析 score 分离、物理 AWGN、多径、CFO、多普勒、完整 LTE 同步/导频/控制/编码/HARQ/MIMO。不要加载旧 checkpoint，不需要 GPU。后续命令可以暂不提供；若暴露未支持命令，明确报错，不能返回占位结果。

**“闭环”指从比特生成时域波形，再从时域输入恢复目标比特并评分；本任务不声称已经完成两源分离。**

## 2. 工作目录和旧项目的处理

新项目目录名：`lte_source_separation`；Python 包名：`rfsep`。

- 用户指定了新项目路径时优先使用该路径。
- 当前已处于 `lte_source_separation` 项目根目录时，就在该目录继续。
- 当前处于旧 `source_separation` 仓库内时，先确认其 Git 根目录，再在其父目录下建立并列的 `lte_source_separation`；不要在旧项目内部嵌套新项目。
- 当前是其他工作目录时，选择其中独立的 `lte_source_separation` 子目录；先确认没有与已有项目冲突。
- 路径已存在时检查内容，保留用户已有改动，不覆盖或删除未知文件。仅有影响实现的真实冲突/权限阻塞才需要向用户说明。
- 新项目采用独立本地 Git 管理；已有 Git 仓库则沿用。不要修改旧仓库，不创建远程仓库，不执行 push。没有生成提交时如实记录工作树状态或内容摘要。
- 旧仓库参考地址：`https://github.com/violetlz/source_separation.git`。此前审查提交：`a39a2fc2312fc6b5e4cdf0d1962a92f711ce627f`。可只读参考星座、OFDM 和审计逻辑；记录实际参考的提交及迁入内容。
- 旧仓库或网络无法访问，不阻塞本任务；依据本文件独立实现。禁止用 `sys.path` 或相对导入依赖旁边的旧工程。

遵守当前环境中实际适用的系统、开发者和项目指令。常规实现选择自行完成；遇到阻塞时保留已完成成果并准确说明，不绕过权限或伪造验证。

## 3. 实现环境与本阶段文件结构

优先使用 Python + NumPy；可使用 SciPy、PyYAML、Matplotlib、pytest。沿用可用的合适 Python 环境，必要时在新目录建立隔离环境，避免改动旧项目环境。使用 `pyproject.toml` 支持可编辑安装，实际验证后记录依赖版本；不要凭空宣称某套依赖已验证。

至少实现下列文件。允许为了清晰调整相邻小模块，但要说明；不要增加通用插件平台、服务端、数据库或网页界面。

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
        platform/lte_v1.yaml
        experiments/smoke.yaml
        experiments/platform_validation.yaml
    src/rfsep/
        __init__.py
        __main__.py
        cli.py
        config.py
        contracts.py
        randomness.py
        observation.py
        waveforms/
            __init__.py
            constellations.py
            ofdm.py
            sources.py
            receiver.py
        data/
            __init__.py
            manifests.py
            datasets.py
        baselines/
            __init__.py
            direct.py
        diagnostics/
            __init__.py
            waveform_audit.py
        evaluation/
            __init__.py
            runner.py
            metrics.py
            plotting.py
    tests/
        conftest.py
        test_waveforms.py
        test_power.py
        test_data_contract.py
        test_randomness.py
        test_metrics.py
        test_noiseless_pipeline.py
        fixtures/reference_manifest.json
    reference/matlab/
        README.md
        generate_reference.m               # 仅在采用 MATLAB 交叉核验时实现
    reports/
        platform_validation.md
    artifacts/
        manifests/
        calibration/
        datasets/
        runs/
```

本次不需要创建模型、先验、训练、RGS/FM 求解器的空壳文件。`data/datasets.py` 当前可提供 NumPy 样本迭代器，不要仅为未来训练引入 PyTorch 依赖。

原始数据和大量运行产物放 `artifacts/` 并加入 Git 忽略规则；少量必要 fixture、配置、实现、文档和验收报告纳入版本控制。复用代码保留已有声明并记录来源。不得为依赖代码自行臆造许可。

## 4. 固定物理边界与数值参数

这是合成的 LTE numerology 数据波形，不是完整标准 LTE 物理帧。接收器观察完整 20 MHz 干扰所在的宽带 IQ；不模拟只留下 5 MHz 的前端滤波与抽取。

| 参数 | 目标 | 干扰 |
|---|---:|---:|
| 标称带宽 | 5 MHz | 20 MHz |
| RB 数 | 25 | 100 |
| 每 RB 子载波数 | 12 | 12 |
| 活动子载波数 K | 300 | 1200 |
| 子载波间隔 | 15000 Hz | 15000 Hz |
| 公共 FFT/IFFT 长度 N | 2048 | 2048 |
| 公共采样率 Fs | 30720000 Hz | 30720000 Hz |
| 每子帧 OFDM 符号数 | 14 | 14 |
| 每时隙 CP 样点数 | `[160,144,144,144,144,144,144]` | 相同 |
| 每子帧 CP 序列 | 上述列表重复两次 | 相同 |
| 子帧/观测窗长 L | 30720 个复数采样，即 1 ms | 相同 |
| 额外窗函数 | 关闭 | 关闭 |
| 中心频率偏移、残余 CFO | 0 | 0 |
| 传播 | 单通道、平坦恒定 | 相同 |

目标活动频点按以下有符号整数升序排列：`[-150,...,-1,1,...,150]`。

干扰活动频点：`[-600,...,-1,1,...,600]`。

DC 为零，其余未活动的 IFFT bins 为零。FFT 自然索引使用 `k % 2048`，文档明确 `fftshift` 只用于展示或显式转换。

固定内部网格形状为 `[num_symbols, num_active_subcarriers]`；目标比特按 `[symbol, active_subcarrier, bit_in_symbol]` 的 C-order 展平。MATLAB 参考通常使用 `[subcarrier,symbol]`，比较时必须显式对齐，不能因为转置两边同时写错而误判通过。

验证配置关系：`Fs == N * delta_f`；`L == 14*N + sum(cp_lengths)`；活动频点合法、唯一、无 DC；目标资源位于干扰活动资源中。配置出现未知字段、冲突值或本阶段未支持的非零物理效应时清楚报错，不能静默忽略。

## 5. 星座映射与两个独立源

### 5.1 映射约定

采用常规 LTE Gray 标记、单位集合平均符号能量。以下 `b0,b1,...` 是一个符号内依次取出的 0/1 比特；做 `1-2*b` 运算前转换成有符号数值，防止 uint8 溢出。

```text
QPSK:
    I = 1 - 2*b0
    Q = 1 - 2*b1
    symbol = (I + j*Q) / sqrt(2)

16QAM:
    I = (1 - 2*b0) * (2 - (1 - 2*b2))
    Q = (1 - 2*b1) * (2 - (1 - 2*b3))
    symbol = (I + j*Q) / sqrt(10)

64QAM:
    I = (1 - 2*b0) * (4 - (1 - 2*b2) * (2 - (1 - 2*b4)))
    Q = (1 - 2*b1) * (4 - (1 - 2*b3) * (2 - (1 - 2*b5)))
    symbol = (I + j*Q) / sqrt(42)
```

实现对应的解映射。允许用小型星座表做确定性最近邻判决，但要定义等距 tie-breaking。穷举 4/16/64 种标签验证映射与逆映射，并用独立参考锚点检查比特含义；仅验证自身往返不足以证明 LTE 标签正确。

参考锚点：QPSK 的 `00` 对应 `(1+j)/sqrt(2)`；16QAM 的 `0000` 对应 `(1+j)/sqrt(10)`；64QAM 的 `000000` 对应 `(3+3j)/sqrt(42)`。这些是常规映射，不实现 MUST 等扩展模式。

### 5.2 目标源

- 每个独立目标记录产生随机数据，QPSK，300 × 14 个 RE。
- 形成对齐目标子帧起点的 1 ms 波形，包含全部真实 CP。
- 目标载波相位在接收参考系内为零；目标时间、相位、幅度参考已知。
- 每例目标真值为 8400 个未编码比特。
- 为落实长记录划分，可按源记录 ID 生成连续 3 个子帧，并固定取中间对齐的子帧；这一内部组织不改变上述目标边缘分布。
- 不对目标做未知随机相位增强或任意时间 crop。

### 5.3 干扰源

1. 生成连续 3 个子帧，长度 `3*L=92160`，各子帧数据独立。
2. 将按频率排序的 1200 个活动频点分成四个连续的 300 子载波块，即四个 25 RB 分配块。
3. 每个块、每个子帧独立且等概率抽取 `QPSK/16QAM/64QAM`；在该块的 14 个符号内使用同一调制。
4. 各块使用独立数据，星座均为单位集合平均符号能量；四块共享 OFDM 时钟和一个全局载波相位。
5. 抽取整数 `tau`，均匀分布在闭区间 `[0,30719]`；抽取 `phi ~ Uniform[0,2*pi)`。
6. 从长波形有效截取：`b = exp(1j*phi) * b_long[L+tau : 2*L+tau]`。

不能用 `roll`、循环补齐或重复一个子帧代替连续生成。测试 `tau=0` 与 `tau=L-1` 两个边界；强制 tau/phi/modulation 的接口仅用于单元测试与诊断，不改变默认随机主分布。

该块级规则是固定合成资源分配近似，不是完整 LTE 调度器。保存相应调制图、tau、phi 和 crop 元数据到评分/审计真值，不能交给主接收器。

## 6. 功率约定和时域混合

### 6.1 固定集合归一化

使用酉 FFT/IFFT：`norm="ortho"`。K 个活动频点、单位平均能量符号的时域有效符号，统一乘固定源系数：

\[
c_K=\sqrt{N/K}.
\]

然后插入 CP，或在整个含 CP 波形上乘同一固定系数；两者应一致。目标/干扰的集合平均时域样点功率均为 1，单条记录的实际功率允许自然波动。

**禁止每条记录 RMS 归一化；禁止按真实样本重新求 κ；禁止按单个调制块的实测能量重新缩放。**

complex128 是数值审计默认精度；支持 complex64 存储和接收验证，并记录实际 FFT 计算精度。不能仅在文件上标 complex64 而隐瞒全流程始终使用更高精度。

实现未来网络坐标转换工具并验证往返：`R(x)=sqrt(2)*stack(real(x),imag(x))`，其逆为 `(I+jQ)/sqrt(2)`。本次不实现 score。

### 6.2 唯一目标接收算子 D

D 接受长度 L 的复数时域数组，执行：按目标 CP 序列去 CP → 每符号酉 FFT → 抽取目标 300 个活动频点 → 除以 `c_300`。

输出 `[14,300]` 目标 RE，随后统一 QPSK 判决和比特展平。Direct 接收只对观测 y 调用这一实现。接收代码不得从 `native_grid`、目标比特或真实 s 直接构造估计。

### 6.3 独立校准与固定 κ

主横轴为目标解调带内 SIR，记作 `sir_re_db`。定义

\[
Q_s=E\|Ds\|^2/4200=1,
\qquad Q_b=E\|Db\|^2/4200.
\]

用独立 `calibration` 记录估计 Q_b；正式校准默认 4096 个独立干扰长记录，每个记录取一个随机 crop，流式计算均值及标准误，不保存整个校准数据集。用目标样本验证 Q_s=1，但不把有限样本噪动当作逐例调整依据。

对每个给定 `gamma=sir_re_db`，使用全条件共享的固定系数：

\[
\kappa=\sqrt{1/(Q_b\,10^{\gamma/10})}.
\]

无噪声时域混合：`y = s + kappa*b`。

另存名义总功率 SIR：`sir_total_db = -20*log10(kappa)`。在该源分布中 Q_b 应接近 0.25，两种 SIR 约差 6.02 dB；该数值仅作 sanity check，不能硬编码 Q_b=0.25 冒充校准。

保存 `calibration.json`：样本数、种子方案、源版本/配置哈希、Q_s 验证结果、Q_b、标准误、数据精度、记录 ID/manifest 的摘要及校准身份。评价时校验其与平台配置一致；缺失或不匹配时明确失败，不能用测试集自动补校准。

记录每例真实带内功率和实际 SIR，但这些真值统计仅进入评价产物。集合 SIR 用总能量比计算，不能把每例 dB 的平均值当成同一统计量。

本阶段只支持 `noise: none` / `noise_variance: 0`。有限物理 SNR 或非零噪声配置应报“不支持”；无需提前实现噪声生成器。

## 7. 数据、随机流与信息隔离

### 7.1 数据结构

定义清楚的、可序列化的数据类型，至少区分：

- `SourceRecord`：源 ID、长记录 ID、规范化波形及生成/审计信息，仅生成和数据层持有。
- `Observation`：不透明 example ID 与接收到的 y。
- `ReceiverInfo`：已声明公开的接收参数、κ、零噪声条件、SIR 定义/校准标识；不能含随机生成密钥、干净源、比特、真实 tau/phi 或每例调制图。
- `GroundTruth`：真实 s/b、目标比特、调制图、tau/phi、参考网格与完整来源记录，用于评分和明确的审计。
- `ReceiverOutput`：目标 RE/判决比特，以及需要时的目标波形估计；不依赖 GroundTruth。

观测与真值分开存储、分开加载。正式接收入口只接收 `Observation` 和 `ReceiverInfo`。不要使用一个含所有字段的 metadata 字典后再靠代码约定隐藏字段。

可重建 manifest 包含源 ID、生成种子、配置哈希、记录所属 split 和 crop 身份，但 manifest 由数据/评价编排层持有，不透传到检测器。

### 7.2 随机性与 split

- 使用局部随机数生成器，不依赖全局 `np.random.seed()` 状态。
- 根种子建议固定为 `20260922`；按用途区分目标数据、干扰数据、调制分配、tau 和 phi 随机流。
- 用稳定哈希/明确的 SeedSequence 方案派生子种子；不能使用跨进程随机化的 Python 内置 `hash()`。
- 长记录具有唯一身份，先归属 `train/validation/test/calibration/audit`，再生成 crop。同一个底层长记录不能跨 split；不同 split 不是只换目录名称。
- 同一长记录的所有 crop 必须留在同一 split，并可追溯 parent record。
- 测试样本对访问顺序、batch 大小、分块处理和进程重启保持一致。
- 更换干扰随机流不得改变目标比特；更换 SIR 不得改变 s/b；SIR 只决定 κ。
- 正式对照对所有 SIR 复用同一组目标/干扰样本，便于后续配对比较。

本任务不需要先生成大型训练数据集。实现可复现的生成与划分机制，保存少量示例、验收集的 manifest 和逐例结果即可。

## 8. 命令入口与运行流程

提供实际可用的统一 CLI，至少支持下面语义；参数拼写可以小幅调整，但 README 必须给出已实际运行的准确命令。

```bash
python -m rfsep audit --config configs/platform/lte_v1.yaml
python -m rfsep calibrate --config configs/platform/lte_v1.yaml --num-records 4096
python -m rfsep evaluate --config configs/experiments/smoke.yaml
python -m rfsep evaluate --config configs/experiments/platform_validation.yaml
python -m rfsep report --run <run_id>
python -m pytest
```

`smoke.yaml`：可先使用 16 或 32 个独立样本快速检查，但必须显式引用有效校准；若使用小样本临时校准，标明 provisional，不能作为正式验收校准。

正式平台验收：256 个独立测试混合，固定 SIR_RE 为 `[-12,-9,-6,-3,0,3,6,9,12]` dB。每点 256 × 8400 个目标比特，所有条件跑完，包含：

1. **Clean target**：D(s) 与真实目标比特比较，必须零误码。
2. **Direct mixture**：D(y) 与真实目标比特比较，记录实际 BER；不要求它在干扰下零误码，也不要求曲线单调。
3. **Oracle subtraction audit**：仅审计代码使用真实 b 计算 `y-kappa*b` 后接收，必须零误码。明确这是链路检查，不是源分离算法成绩，不能把它画成 RGS/FM 或实际接收器性能。

用流式或适当小批处理控制内存，不为 9 个 SIR 各存一套重复波形。无需运行昂贵训练或生成几十 GB 数据。

## 9. 必须运行的验收与反例测试

以下检查用于捕获实质错误，不要求堆砌只验证实现细节的测试。核心项失败时先定位和修复，不能删测试或放宽到失去意义。

### A. 星座与资源映射

- 穷举 4/16/64 标签的映射/逆映射，所有标签通过；检查独立锚点及星座全表平均能量为 1。
- 核查活动频点数量、DC 空载、负/正边缘与频率次序。
- 用独立单音或直接复指数和核对若干实际 N=2048 的时域采样，覆盖正负边缘频点、CP 边界和尺度；独立检查不得调用被测调制器重建期望值。
- 活动/空载 bin 的检查在已对齐的单 OFDM 有效符号 FFT 上做。不要要求含 CP、跨边界截取的整段 PSD 带外严格为零。

### B. 波形往返与 CP

- 正常 1 ms 波形总长 30720，CP 总长 2048。
- 验证每个符号 CP 等于该符号有效部分尾部；两个时隙的首符号都是 160 点。
- 干扰 CP 审计在截取前的连续源或截取后可定位的完整干扰符号上进行，使用其真实边界仅作审计。不能拿目标 CP 边界要求异步干扰也满足同样的重复位置。
- 目标单源恢复全部 8400 比特，BER=0。
- 在未随机 crop/旋转的干扰参考帧中，用真实调制图进行审计解调，验证三种调制均正确；这不是主接收器入口。
- float64/complex128 相对网格重建误差建议 <1e-10；complex64 路径建议 <1e-5，记录实际误差和计算精度。
- **反例：只把 CP 样点清零/破坏，保留有效符号不变，CP 审计必须失败。**去 CP 后 FFT 往返仍可能通过，这不能替代 CP 审计。
- tau=0 与 tau=L-1 结果必须等于正确的长记录切片；样本不足时明确报错，不循环补齐。

### C. 归一化与 SIR

- 目标/干扰集合时域平均功率接近 1，同时展示单例功率存在正常波动。
- 对集合均值记录记录级标准误；可采用 `abs(mean_power-1) <= max(0.01,5*SE)` 作为工程检查，不能据单条样本要求精确功率 1。
- 校准 Q_b 来自独立记录，接近 0.25；展示估计、标准误与校准身份，不人为强制为理论近似值。
- 独立测试集合实测 SIR_RE 与设定的误差建议不超过 0.25 dB，超限需查原因；每例 SIR 波动允许存在。
- 验证不同样本在同一条件使用完全相同 κ；不能从 GroundTruth 求每例 κ。
- 人为制造配置/源版本与校准哈希不匹配，必须明确拒绝。

### D. 时域闭环与信息边界

- 验证 `y=s+kappa*b`，形状、dtype 正确，数值残差处于相应精度误差内。
- 混合函数接受时域数组，不接受 native_grid 替代波形；接收器能在真值网格不加载的情况下完整运行。
- ReceiverInfo 中无隐藏真值字段；删除评分真值不影响 Direct 接收输出，评分步骤可以明确因缺真值而不能执行。
- 修改 GroundTruth 的目标比特或干扰参数只能影响审计/评分，不能改变已经给定观测下的检测输出。
- Oracle subtraction 仅在审计路径使用真值，方法名称和报告明确区分。

### E. 随机性与数据划分

- 同一 ID/配置/种子在打乱顺序、不同 batch 大小以及新进程中生成相同结果，比较波形内容而非可能含时间戳的压缩文件字节。
- train/validation/test/calibration/audit 底层长记录集合互斥；人为尝试同一长记录进入多个 split 时能够检测。
- 两个源的随机流独立；改变 B 的数据/相位种子不改变 s。
- 更换 SIR 只改变 κ 和 y，不改变干净 s/b。

### F. 评分与完整运行

- 用手工构造的少量错比特检查误码计数，禁止因 reshape/broadcast 导致错误统计。
- 256 例 × 9 个 SIR 条件全部有结果；每点 Direct 分母为 2,150,400 比特，记录实际分母。
- Clean target 和 Oracle subtraction 审计零误码；Direct 在干扰下的实际错误如实报告。
- 失败、NaN 或无效样本不能静默排除；运行失败必须显式标记，不能输出伪造的零 BER。

不要求本阶段进行 BER 优越性的置信区间推断或训练随机种子比较；保存逐记录错误数，为后续配对统计留出基础。

## 10. 独立参考核验

基础参数与星座映射参考：ETSI TS 136 211 V15.11.0，§6.2.3、§6.12 和 §7.1.2–7.1.4：

https://www.etsi.org/deliver/etsi_ts/136200_136299/136211/15.11.00_60/ts_136211v151100p.pdf

如果当前环境有可用 MATLAB + LTE Toolbox，采用少量确定性网格调用 `lteOFDMModulate` 交叉核验，显式指定正常 CP、Nfft=2048、windowing=0。读取返回的采样与 CP 信息，明确工具与酉 IFFT 的固定尺度关系；不能通过每例最佳拟合归一化掩盖错误。

参考接口：

https://www.mathworks.com/help/lte/ref/lteofdmmodulate.html

MATLAB 不可用时不安装大型软件、不阻塞任务；使用独立复指数求和/固定星座锚点/解析 CP 与采样关系完成数学交叉核验，在报告中明确“MATLAB 交叉核验未运行”。独立参考应与主实现的关键计算路径不同，不能两个函数调用同一个错误实现后称为交叉验证。

## 11. 输出文件与报告内容

每个运行目录 `artifacts/runs/<run_id>/` 至少包含：

```text
resolved_config.yaml
run_manifest.json
validation_checks.json
per_example.csv
summary.csv
run.log
figures/
```

其中 `run_manifest.json` 记录代码状态、Python/依赖版本、平台哈希、校准哈希、种子方案、样本 manifest 摘要、精度、硬件和耗时。当前未使用模型，模型字段写明不适用，不能填旧 checkpoint。

`per_example.csv` 至少保存：run ID、example ID、方法/审计类型、SIR_RE 设定、κ、实际带内/总功率比、错误数、比特数、状态和耗时。实际功率来自评分层，不供检测器使用。

至少提供两类诊断图：

1. 同一例目标、干扰和混合的频谱/PSD，频率单位正确，说明两种标称带宽与实际活动资源尺度。有限窗口 OFDM 允许旁瓣和泄漏。
2. CP 与对应符号尾部的对照/误差图，或等价的易读数值展示。

可额外保存 Direct BER-SIR 图，必须标明它仅是平台基线，未运行分离方法。

`reports/platform_validation.md` 必须包含：

- 实际新项目路径、运行环境、时间、代码状态与复现命令。
- 平台参数、比特/频点顺序、采样/CP、归一化、公开信息与简化边界。
- 目标/干扰分布和长记录截取方式。
- 测试命令、通过/失败/未运行数量，关键误差的实际数值，反例测试是否能检错。
- Q_b 校准值、标准误、样本数、功率与 SIR 的实测对照。
- 9 个 SIR 上 Direct 的实际错误数、比特数与 BER，Clean/Oracle 审计结果。
- MATLAB/独立参考核验实际运行了什么；未运行项明确写未运行。
- 配置、校准、结果和图的可定位路径。
- 已知限制：未编码、无导频/控制、无物理噪声、宽带观测；未训练模型、未运行 RGS/FM。
- 当前结论只限于链路是否正确、可复现，不能声称已分离两源或证明生成方法有效。

README 给出从安装到重现验收的准确命令。`docs/progress.md` 更新本次完成内容、剩余问题和下一阶段入口。若附件齐全，保留设计原文并指明本次实现范围；否则以本文件整理阶段文档，不虚构未提供附件内容。

## 12. 完成标准和最终回复

完成标准是：实际创建了可运行项目；核心验收通过；正式独立校准、256 例全部 SIR 条件和报告生成实际执行；用户按 README 可以重现。

先制定简短实施计划，然后连续推进。不要在完成骨架、写完测试或给出命令后就结束。核心检查发现问题时在授权范围内修复，再重跑受影响的验证；充分验证后停止无目的扩大测试。

若环境存在真实阻塞，尽可能完成其余独立工作，在报告中标注 blocked/未运行并说明准确原因。不能把环境未验证项写成 PASS，也不能为了完成任务偷偷缩减正式样本数或源分布。

最终回复简要给出：

1. 新项目实际路径与主要实现内容。
2. 运行过的验证、实际结果和未完成项。
3. `platform_validation.md` 及结果目录位置。
4. 下一步是否具备接入独立源先验训练的基础；若不具备，列出具体阻塞。

现在开始执行本任务。

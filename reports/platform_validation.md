# LTE 场景启发平台：生成与接收闭环验收报告

生成时间：2026-09-22T14:57:58.294537+00:00。本报告只证明生成—混合—目标解调—评分链路正确且可复现，不声称已经完成两源分离，也不证明任何生成先验有效。

## 项目、环境与复现

- 项目路径：`/data3/workspace/lab/lte_source_separation`
- 正式运行：`/data3/workspace/lab/lte_source_separation/artifacts/runs/20260922T145746Z_platform_validation`
- Python：`3.13.11 | packaged by Anaconda, Inc. | (main, Dec 10 2025, 21:28:48) [GCC 14.3.0]`；NumPy `2.5.3`；PyYAML `6.0.3`；Matplotlib `3.11.2`
- 代码状态：commit=无初始提交，branch=main；工作树状态：`?? .gitignore
?? AGENTS.md
?? README.md
?? THIRD_PARTY.md
?? artifacts/
?? configs/
?? docs/
?? pyproject.toml
?? reference/
?? reports/
?? requirements.lock
?? src/
?? tests/`
- 平台配置 SHA-256：`18b2a1bc425f217e36c9e67f9fd64f13701f45c83148ec0ccba42205f8115cb5`
- 校准身份：`aae240e407c2d42820842d847b278fe5278d29696cc9e3afc80d2cf483acc444`
- 样本 manifest 摘要：`c271b93400e2c975dcbf3599efc2168bfac03c31cd83aa53d2e5504d5269d180`

从项目根目录复现：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/python -m rfsep audit --config configs/platform/lte_v1.yaml
.venv/bin/python -m rfsep calibrate --config configs/platform/lte_v1.yaml --num-records 4096
.venv/bin/python -m rfsep evaluate --config configs/experiments/platform_validation.yaml
.venv/bin/python -m rfsep report --run 20260922T145746Z_platform_validation
```

## 固定平台边界

目标是 5 MHz、25 RB、300 个活动子载波的 QPSK；干扰是 20 MHz、100 RB、1200 个活动子载波，四个连续 300 子载波块在每子帧独立等概率选择 QPSK/16QAM/64QAM。两者共用 30.72 MHz 采样率、2048 点酉 FFT/IFFT、14 个 OFDM 符号和正常 CP `[160,144×6]` 重复两次。目标位为 `[symbol, active_subcarrier, bit_in_symbol]` 的 C-order；自然 FFT bin 为 `k % 2048`，`fftshift` 不参与主链路。

每个源先连续生成 3 个 1 ms 子帧。目标固定取中间对齐子帧；干扰抽取 `tau∈[0,30719]` 和全局相位 `phi`，取 `exp(j phi) * b_long[L+tau:2L+tau]`，没有 roll、循环补齐或单帧复制。固定集合归一化为 `c_K=sqrt(2048/K)`，禁止逐记录 RMS 归一化。

接收器公开信息只含平台/校准身份、名义 SIR、固定 κ 和零噪声声明；真实源、目标位、tau、phi、调制图只在独立 GroundTruth 中。Direct 只接收 Observation 与 ReceiverInfo。当前是合成 numerology 数据波形，不是完整 LTE 帧；接收端观察完整 20 MHz 宽带 IQ。

## 测试与反例检查

`.venv/bin/python -m pytest --junitxml=artifacts/runs/pytest_delivery_a.xml`：17 passed / 0 failed / 0 skipped（未运行 0）。 正式运行内嵌波形审计为 14 通过 / 0 失败，整体 `PASS`。关键数值：complex128 网格相对误差 `1.942e-16`，complex64 路径 `8.674e-08`，原始 CP 最大误差 `0.000e+00`。未裁剪干扰参考帧覆盖 `16QAM, 64QAM, QPSK` 且 CP/调制审计通过。把首个 CP 清零的反例产生 `2.175e+00` 最大误差，同时去 CP 后网格误差仍为 `1.942e-16`，证明 CP 审计能够捕获仅 FFT 往返看不出的破坏。人为 split 冲突已被拒绝：`True`。

正式运行状态：`PASSED`；验证总门：`PASS`。256×9 条件全部保留，每点 Direct 分母为 2,150,400 bit。Clean 总错误 `0`；Oracle subtraction audit 总错误 `0`。Oracle 项仅是使用真值干扰的链路审计，不是接收算法成绩。

## 独立校准与功率

校准 split 与 test split 独立，记录数 `4096`，provisional=`false`。`Q_b=0.2496880567`，记录级标准误 `3.575e-05`；`Q_s=1.0000000000`，标准误 `2.797e-17`。目标含 CP 时域平均功率 `0.99994403`，干扰为 `1.00004831`。κ 由全条件共享的校准值一次确定，没有使用 test 真值逐例重算。正式测试集合最大 SIR_RE 偏差 `0.0018` dB（门限 0.25 dB）。

## Direct 基线结果

| SIR_RE (dB) | 名义总功率 SIR (dB) | 实测集合 SIR_RE (dB) | Direct 错误数 | 比特数 | BER |
|---:|---:|---:|---:|---:|---:|
| -12 | -18.0260 | -11.9982 | 884718 | 2150400 | 0.4114202 |
| -9 | -15.0260 | -8.9982 | 807774 | 2150400 | 0.37563895 |
| -6 | -12.0260 | -5.9982 | 701542 | 2150400 | 0.32623791 |
| -3 | -9.0260 | -2.9982 | 557497 | 2150400 | 0.2592527 |
| 0 | -6.0260 | 0.0018 | 378224 | 2150400 | 0.17588542 |
| 3 | -3.0260 | 3.0018 | 171111 | 2150400 | 0.079571708 |
| 6 | -0.0260 | 6.0018 | 35472 | 2150400 | 0.016495536 |
| 9 | 2.9740 | 9.0018 | 1681 | 2150400 | 0.00078171503 |
| 12 | 5.9740 | 12.0018 | 0 | 2150400 | 0（约 95% 上界 1.4e-06） |

Direct 曲线是未做分离的观测基线，非 RGS/FM 性能。所有失败、NaN 与状态均进入机器可读输出；本次没有静默删样本。

## 独立参考与产物

已实际运行 MATLAB 25.1.0.2943329 (R2025a) / LTE Toolbox 25.1；按解析固定关系 `MATLAB = Python × sqrt(K)/N` 换算（未做逐例最佳拟合），目标波形相对误差为 `3.192e-16`，干扰波形为 `3.342e-16`。

Git 中保留的轻量正式结果快照位于 `reports/formal_validation/`，包括汇总 CSV、验证门、校准摘要、MATLAB 核验、运行 manifest 以及三张诊断图。以下 `artifacts/` 路径是本机的完整可复现运行产物；逐例 CSV 和原始 MATLAB fixture 不提交 Git。

- 校准：`/data3/workspace/lab/lte_source_separation/artifacts/calibration/lte_v1_calibration.json`
- resolved config：`/data3/workspace/lab/lte_source_separation/artifacts/runs/20260922T145746Z_platform_validation/resolved_config.yaml`
- run manifest：`/data3/workspace/lab/lte_source_separation/artifacts/runs/20260922T145746Z_platform_validation/run_manifest.json`
- validation checks：`/data3/workspace/lab/lte_source_separation/artifacts/runs/20260922T145746Z_platform_validation/validation_checks.json`
- 逐例结果：`/data3/workspace/lab/lte_source_separation/artifacts/runs/20260922T145746Z_platform_validation/per_example.csv`
- 汇总：`/data3/workspace/lab/lte_source_separation/artifacts/runs/20260922T145746Z_platform_validation/summary.csv`
- 频谱图：`/data3/workspace/lab/lte_source_separation/artifacts/runs/20260922T145746Z_platform_validation/figures/waveform_spectrum.png`
- CP 图：`/data3/workspace/lab/lte_source_separation/artifacts/runs/20260922T145746Z_platform_validation/figures/cp_comparison.png`
- Direct BER 图：`/data3/workspace/lab/lte_source_separation/artifacts/runs/20260922T145746Z_platform_validation/figures/direct_ber_vs_sir.png`

## 已知限制与下一阶段

当前未编码、无导频/控制/HARQ/MIMO，无物理 AWGN、多径、CFO、多普勒或前端 5 MHz 滤波；未训练模型，未实现或运行 Diffusion、RGS、FM、SIC 或 LMMSE。交付 A 的数据契约、可复现源生成、独立校准、统一目标接收器和评分产物已经形成，下一阶段可以在不改变这些物理与信息边界的前提下接入两个独立源先验训练。

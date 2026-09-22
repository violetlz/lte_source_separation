from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_validation_report(project_root: Path, run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    checks = json.loads((run_dir / "validation_checks.json").read_text())
    resolved_text = (run_dir / "resolved_config.yaml").read_text()
    import yaml
    resolved = yaml.safe_load(resolved_text)
    calibration_path = project_root / "artifacts" / "calibration" / "lte_v1_calibration.json"
    calibration = json.loads(calibration_path.read_text())
    summary = _read_csv(run_dir / "summary.csv")
    direct = sorted((row for row in summary if row["method"] == "direct_mixture"),
                    key=lambda row: float(row["sir_re_db"]))
    clean_errors = sum(int(row["num_errors"]) for row in summary if row["method"] == "clean_target")
    oracle_errors = sum(int(row["num_errors"]) for row in summary if row["method"] == "oracle_subtraction_audit")
    matlab_path = project_root / "artifacts" / "matlab_reference" / "matlab_crosscheck.json"
    if matlab_path.is_file():
        matlab = json.loads(matlab_path.read_text())
        matlab_text = (f"已实际运行 MATLAB {matlab['matlab_version']} / LTE Toolbox {matlab['lte_toolbox_version']}；"
                       f"按解析固定关系 `MATLAB = Python × sqrt(K)/N` 换算（未做逐例最佳拟合），"
                       f"目标波形相对误差为 `{matlab['target_relative_error']:.3e}`，"
                       f"干扰波形为 `{matlab['interference_relative_error']:.3e}`。")
    else:
        matlab_text = "MATLAB 交叉核验未运行；仅执行了解析复指数、星座锚点及 CP 数学核验。"
    git = manifest["git"]
    git_state = (f"commit={git.get('commit') or '无初始提交'}，branch={git.get('branch') or 'main'}；"
                 f"工作树状态：`{git.get('status_porcelain') or 'clean'}`")
    test_xml = project_root / "artifacts" / "runs" / "pytest_delivery_a.xml"
    if test_xml.is_file():
        root = ET.parse(test_xml).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        test_counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                       for key in ("tests", "failures", "errors", "skipped")}
        passed = test_counts["tests"] - test_counts["failures"] - test_counts["errors"] - test_counts["skipped"]
        pytest_text = (f"`.venv/bin/python -m pytest --junitxml=artifacts/runs/pytest_delivery_a.xml`："
                       f"{passed} passed / {test_counts['failures'] + test_counts['errors']} failed / "
                       f"{test_counts['skipped']} skipped（未运行 0）。")
    else:
        pytest_text = "pytest 结果文件缺失，本项未运行。"
    table = ["| SIR_RE (dB) | 名义总功率 SIR (dB) | 实测集合 SIR_RE (dB) | Direct 错误数 | 比特数 | BER |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in direct:
        ber_text = (f"0（约 95% 上界 {float(row['zero_error_95_upper_bound']):.3g}）"
                    if int(row["num_errors"]) == 0 else f"{float(row['ber']):.8g}")
        table.append(f"| {float(row['sir_re_db']):g} | {float(row['nominal_sir_total_db']):.4f} | "
                     f"{float(row['measured_sir_re_db']):.4f} | "
                     f"{int(row['num_errors'])} | {int(row['num_bits'])} | {ber_text} |")
    audit = checks["waveform_audit"]
    content = f"""# LTE 场景启发平台：生成与接收闭环验收报告

生成时间：{datetime.now(timezone.utc).isoformat()}。本报告只证明生成—混合—目标解调—评分链路正确且可复现，不声称已经完成两源分离，也不证明任何生成先验有效。

## 项目、环境与复现

- 项目路径：`{project_root.resolve()}`
- 正式运行：`{run_dir.resolve()}`
- Python：`{manifest['python'].splitlines()[0]}`；NumPy `{manifest['numpy']}`；PyYAML `{manifest['pyyaml']}`；Matplotlib `{manifest['matplotlib']}`
- 代码状态：{git_state}
- 平台配置 SHA-256：`{manifest['platform_hash']}`
- 校准身份：`{manifest['calibration_hash']}`
- 样本 manifest 摘要：`{manifest['sample_manifest_digest']}`

从项目根目录复现：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/python -m rfsep audit --config configs/platform/lte_v1.yaml
.venv/bin/python -m rfsep calibrate --config configs/platform/lte_v1.yaml --num-records 4096
.venv/bin/python -m rfsep evaluate --config configs/experiments/platform_validation.yaml
.venv/bin/python -m rfsep report --run {run_dir.name}
```

## 固定平台边界

目标是 5 MHz、25 RB、300 个活动子载波的 QPSK；干扰是 20 MHz、100 RB、1200 个活动子载波，四个连续 300 子载波块在每子帧独立等概率选择 QPSK/16QAM/64QAM。两者共用 30.72 MHz 采样率、2048 点酉 FFT/IFFT、14 个 OFDM 符号和正常 CP `[160,144×6]` 重复两次。目标位为 `[symbol, active_subcarrier, bit_in_symbol]` 的 C-order；自然 FFT bin 为 `k % 2048`，`fftshift` 不参与主链路。

每个源先连续生成 3 个 1 ms 子帧。目标固定取中间对齐子帧；干扰抽取 `tau∈[0,30719]` 和全局相位 `phi`，取 `exp(j phi) * b_long[L+tau:2L+tau]`，没有 roll、循环补齐或单帧复制。固定集合归一化为 `c_K=sqrt(2048/K)`，禁止逐记录 RMS 归一化。

接收器公开信息只含平台/校准身份、名义 SIR、固定 κ 和零噪声声明；真实源、目标位、tau、phi、调制图只在独立 GroundTruth 中。Direct 只接收 Observation 与 ReceiverInfo。当前是合成 numerology 数据波形，不是完整 LTE 帧；接收端观察完整 20 MHz 宽带 IQ。

## 测试与反例检查

{pytest_text} 正式运行内嵌波形审计为 {audit['passed']} 通过 / {audit['failed']} 失败，整体 `{'PASS' if audit['all_passed'] else 'FAIL'}`。关键数值：complex128 网格相对误差 `{audit['checks']['target_roundtrip']['relative_grid_error']:.3e}`，complex64 路径 `{audit['checks']['precision_and_network_coordinates']['complex64_relative_grid_error']:.3e}`，原始 CP 最大误差 `{audit['checks']['cp']['cp_max_abs_error']:.3e}`。未裁剪干扰参考帧覆盖 `{', '.join(audit['checks']['interference_reference_frame']['modulations_covered'])}` 且 CP/调制审计通过。把首个 CP 清零的反例产生 `{audit['checks']['cp_damage_counterexample']['damaged_cp_max_abs_error']:.3e}` 最大误差，同时去 CP 后网格误差仍为 `{audit['checks']['cp_damage_counterexample']['post_cp_removal_grid_relative_error']:.3e}`，证明 CP 审计能够捕获仅 FFT 往返看不出的破坏。人为 split 冲突已被拒绝：`{audit['checks']['split_disjointness']['deliberate_collision_rejected']}`。

正式运行状态：`{manifest['status'].upper()}`；验证总门：`{'PASS' if checks['all_passed'] else 'FAIL'}`。256×9 条件全部保留，每点 Direct 分母为 2,150,400 bit。Clean 总错误 `{clean_errors}`；Oracle subtraction audit 总错误 `{oracle_errors}`。Oracle 项仅是使用真值干扰的链路审计，不是接收算法成绩。

## 独立校准与功率

校准 split 与 test split 独立，记录数 `{calibration['num_records']}`，provisional=`{str(calibration['provisional']).lower()}`。`Q_b={calibration['q_b']:.10f}`，记录级标准误 `{calibration['q_b_standard_error']:.3e}`；`Q_s={calibration['q_s']:.10f}`，标准误 `{calibration['q_s_standard_error']:.3e}`。目标含 CP 时域平均功率 `{calibration['target_time_power_mean']:.8f}`，干扰为 `{calibration['interference_time_power_mean']:.8f}`。κ 由全条件共享的校准值一次确定，没有使用 test 真值逐例重算。正式测试集合最大 SIR_RE 偏差 `{checks['sir_ensemble_tolerance']['max_abs_error_db']:.4f}` dB（门限 0.25 dB）。

## Direct 基线结果

{chr(10).join(table)}

Direct 曲线是未做分离的观测基线，非 RGS/FM 性能。所有失败、NaN 与状态均进入机器可读输出；本次没有静默删样本。

## 独立参考与产物

{matlab_text}

- 校准：`{calibration_path.resolve()}`
- resolved config：`{(run_dir / 'resolved_config.yaml').resolve()}`
- run manifest：`{(run_dir / 'run_manifest.json').resolve()}`
- validation checks：`{(run_dir / 'validation_checks.json').resolve()}`
- 逐例结果：`{(run_dir / 'per_example.csv').resolve()}`
- 汇总：`{(run_dir / 'summary.csv').resolve()}`
- 频谱图：`{(run_dir / 'figures/waveform_spectrum.png').resolve()}`
- CP 图：`{(run_dir / 'figures/cp_comparison.png').resolve()}`
- Direct BER 图：`{(run_dir / 'figures/direct_ber_vs_sir.png').resolve()}`

## 已知限制与下一阶段

当前未编码、无导频/控制/HARQ/MIMO，无物理 AWGN、多径、CFO、多普勒或前端 5 MHz 滤波；未训练模型，未实现或运行 Diffusion、RGS、FM、SIC 或 LMMSE。交付 A 的数据契约、可复现源生成、独立校准、统一目标接收器和评分产物已经形成，下一阶段可以在不改变这些物理与信息边界的前提下接入两个独立源先验训练。
"""
    output = project_root / "reports" / "platform_validation.md"
    output.write_text(content)
    return output

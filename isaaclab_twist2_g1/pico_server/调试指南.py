#!/usr/bin/env python
"""
SONIC推理调试指南

根据日志输出诊断问题
"""

print("="*80)
print("SONIC推理调试指南")
print("="*80)

print("""
根据您的日志输出分析：

【观察到的现象】
1. ✓ SMPL数据已接收: _smpl_joints_buf有有效数据
2. ✗ 输出仍是默认值: sonic_targets = SONIC_DEFAULT_POS
3. ⚠ ZMQ警告: "ZMQPoller: no data available"

【可能的原因】

原因1: _smpl_data_valid标志未设置 (最可能)
----------------------------------------------
症状: 虽然SMPL数据存在，但_smpl_data_valid=False
原因: 首次接收数据时设置了标志，但后续没有新数据时标志可能被重置

解决方案:
  查看日志中是否有:
  - "[ZMQ] SMPL data marked as VALID"
  - "[SONIC] _smpl_data_valid=True"

  如果没有，说明标志设置有问题。


原因2: Encoder/Decoder未加载
----------------------------------------------
症状: 模型文件路径错误或加载失败
原因: encoder_path或decoder_path不正确

解决方案:
  查看日志中是否有:
  - "[SonicActionProvider] loaded model_encoder.onnx"
  - "[SonicActionProvider] loaded model_decoder.onnx"
  - "Successful load sonic model"

  如果没有，检查模型路径:
  echo $SONIC_ENCODER_PATH
  echo $SONIC_DECODER_PATH


原因3: Encoder推理失败
----------------------------------------------
症状: 输入维度不匹配或数据格式错误
原因: encoder输入不是1762维或数据类型错误

解决方案:
  查看日志中是否有:
  - "[SONIC] Encoder input shape: (1, 1762)"
  - "[SONIC] ✓ Encoder output latent shape: (1, 64)"

  如果看到错误，检查:
  - encoder输入维度是否正确
  - 数据类型是否为float32


原因4: ZMQ数据流中断
----------------------------------------------
症状: "ZMQPoller: no data available" 持续出现
原因: Pico服务器停止发送数据或连接断开

解决方案:
  1. 检查Pico服务器是否还在运行
  2. 查看Pico服务器日志是否有FPS输出
  3. 重启Pico服务器


【调试步骤】

步骤1: 检查日志输出
-------------------
运行系统后，查找以下关键日志:

必须看到的日志:
  ✓ "[SonicActionProvider] loaded model_encoder.onnx"
  ✓ "[SonicActionProvider] loaded model_decoder.onnx"
  ✓ "Successful load sonic model"
  ✓ "[ZMQ] SMPL data marked as VALID"
  ✓ "[SONIC] _smpl_data_valid=True"
  ✓ "[SONIC] Encoder input shape: (1, 1762)"
  ✓ "[SONIC] ✓ Encoder output latent shape: (1, 64)"
  ✓ "[SONIC] ✓ Decoder output shape: (1, 29)"

如果缺少任何一个，说明该步骤有问题。


步骤2: 检查_smpl_data_valid标志
-------------------------------
在日志中搜索:
  "[SONIC] _smpl_data_valid="

如果显示False，说明:
  - SMPL数据未接收
  - 或数据全为0
  - 或标志设置逻辑有问题


步骤3: 检查encoder推理
---------------------
在日志中搜索:
  "[SONIC] Running encoder inference..."

如果没有这行，说明没有进入推理逻辑。
如果有这行但没有"✓ Encoder output"，说明推理失败。


步骤4: 检查数据有效性
--------------------
在日志中查看:
  "[SONIC] SMPL joints sum: XXX"

如果sum接近0，说明SMPL数据无效。
如果sum > 1.0，说明SMPL数据有效。


【快速修复】

如果_smpl_data_valid始终为False:
--------------------------------
可能是因为ZMQ没有新数据，但历史数据是有效的。

临时解决方案:
  在_run_gear_sonic()开头添加:

  # 如果历史缓冲区有有效数据，强制设置标志
  if np.abs(self._smpl_joints_buf).sum() > 1.0:
      self._smpl_data_valid = True


如果encoder推理失败:
-------------------
检查输入维度:
  print(f"Encoder input shape: {encoder_input.shape}")
  print(f"Expected: (1, 1762)")

如果不匹配，检查每个观察值的维度。


【联系支持】

如果以上步骤都无法解决问题，请提供:
1. 完整的日志输出（从启动到出错）
2. encoder/decoder模型路径
3. observation_config.yaml内容
4. Pico服务器是否正常运行

""")

print("="*80)
print("现在请运行系统并查看日志输出")
print("="*80)
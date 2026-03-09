#!/usr/bin/env python
"""
检查SONIC Encoder模型的实际输入要求

使用方法:
    python check_sonic_encoder.py /path/to/model_encoder.onnx
"""

import sys
import os
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("错误: 未安装onnxruntime")
    print("安装命令: pip install onnxruntime")
    sys.exit(1)

def check_encoder(encoder_path):
    """检查encoder模型的输入输出"""
    if not os.path.isfile(encoder_path):
        print(f"错误: 文件不存在: {encoder_path}")
        return

    print("="*80)
    print(f"检查Encoder模型: {encoder_path}")
    print("="*80)

    try:
        session = ort.InferenceSession(encoder_path)

        print("\n【Encoder输入】")
        print("-"*80)
        total_dims = 0
        for i, inp in enumerate(session.get_inputs()):
            print(f"\n输入 [{i}]: {inp.name}")
            print(f"  形状: {inp.shape}")
            print(f"  类型: {inp.type}")

            # 计算维度
            if len(inp.shape) >= 2:
                dims = 1
                for d in inp.shape[1:]:  # 跳过batch维度
                    if isinstance(d, int):
                        dims *= d
                    elif isinstance(d, str):
                        print(f"  注意: 维度 '{d}' 是动态的")
                        dims = "动态"
                        break
                if isinstance(dims, int):
                    total_dims += dims
                    print(f"  特征维度: {dims}")

        if isinstance(total_dims, int):
            print(f"\n总特征维度: {total_dims}")

        print("\n【Encoder输出】")
        print("-"*80)
        for i, out in enumerate(session.get_outputs()):
            print(f"\n输出 [{i}]: {out.name}")
            print(f"  形状: {out.shape}")
            print(f"  类型: {out.type}")

        print("\n" + "="*80)
        print("【分析】")
        print("="*80)

        # 分析输入形状
        inputs = session.get_inputs()

        if len(inputs) == 1:
            print("\n✓ 单输入模式")
            inp = inputs[0]
            if len(inp.shape) == 2:
                print(f"  形状: (batch, features)")
                if isinstance(inp.shape[1], int):
                    feat_dim = inp.shape[1]
                    print(f"  特征维度: {feat_dim}")

                    # 根据文档推断
                    print(f"\n  根据SMPL模式文档，期望维度: 844")
                    print(f"    - encoder_mode_4: 4")
                    print(f"    - smpl_joints_10frame_step1: 720 (10×24×3)")
                    print(f"    - smpl_anchor_orientation_10frame_step1: 60 (10×6)")
                    print(f"    - motion_joint_positions_wrists_10frame_step1: 60 (10×6)")

                    if feat_dim == 844:
                        print(f"\n  ✓ 维度匹配！这是SMPL模式的encoder")
                    else:
                        print(f"\n  ✗ 维度不匹配 (实际: {feat_dim}, 期望: 844)")
                        print(f"     可能是其他模式或配置")

        elif len(inputs) == 3:
            print("\n✓ 三输入模式")
            for i, inp in enumerate(inputs):
                print(f"\n  输入{i}: {inp.name}")
                print(f"    形状: {inp.shape}")

                if len(inp.shape) >= 2:
                    # 推断可能的含义
                    if len(inp.shape) == 4:  # (batch, frames, joints, coords)
                        print(f"    → 可能是: SMPL joints (batch, 10, 24, 3)")
                    elif len(inp.shape) == 3:
                        if inp.shape[-1] == 6:
                            print(f"    → 可能是: 6D rotation (batch, 10, 6)")
                        elif inp.shape[-1] == 29:
                            print(f"    → 可能是: joint positions (batch, 10, 29)")
                        elif inp.shape[-1] == 3:
                            print(f"    → 可能是: 3D positions (batch, 10, 3)")
                    elif len(inp.shape) == 2:
                        if inp.shape[-1] == 4:
                            print(f"    → 可能是: encoder mode (batch, 4)")
                        elif inp.shape[-1] == 720:
                            print(f"    → 可能是: flattened SMPL joints (batch, 720)")
                        elif inp.shape[-1] == 60:
                            print(f"    → 可能是: flattened 6D rotation 或 wrist positions (batch, 60)")

        elif len(inputs) == 4:
            print("\n✓ 四输入模式")
            print("  可能对应:")
            print("    - 输入0: encoder_mode (batch, 4)")
            print("    - 输入1: smpl_joints (batch, 720)")
            print("    - 输入2: anchor_orientation (batch, 60)")
            print("    - 输入3: wrist_positions (batch, 60)")

        else:
            print(f"\n? 未知输入模式 ({len(inputs)}个输入)")

        print("\n" + "="*80)
        print("【建议】")
        print("="*80)

        if len(inputs) == 1:
            print("\n如果是单输入模式，需要将所有特征拼接成一个向量:")
            print("""
encoder_input = np.concatenate([
    encoder_mode,           # (4,)
    smpl_joints_flat,       # (720,)
    anchor_orient_flat,     # (60,)
    wrist_pos_flat,         # (60,)
])[np.newaxis]  # (1, 844)

enc_inputs = {
    session.get_inputs()[0].name: encoder_input
}
""")

        elif len(inputs) == 3:
            print("\n如果是三输入模式，需要分别准备:")
            print("""
enc_inputs = {
    session.get_inputs()[0].name: smpl_joints,      # (1, 10, 24, 3) 或 (1, 720)
    session.get_inputs()[1].name: anchor_orient,    # (1, 10, 6) 或 (1, 60)
    session.get_inputs()[2].name: joint_pos_hist,   # (1, 10, 29) 或其他
}
""")

        print("\n运行测试推理以验证:")
        print("""
# 创建随机测试数据
test_inputs = {}
for inp in session.get_inputs():
    shape = [1 if isinstance(d, str) else d for d in inp.shape]
    test_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

# 尝试推理
try:
    outputs = session.run(None, test_inputs)
    print("✓ 推理成功！")
    for i, out in enumerate(outputs):
        print(f"  输出{i}形状: {out.shape}")
except Exception as e:
    print(f"✗ 推理失败: {e}")
""")

    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python check_sonic_encoder.py <encoder_path>")
        print("\n示例:")
        print("  python check_sonic_encoder.py /path/to/model_encoder.onnx")
        print("  python check_sonic_encoder.py ../GR00T-WholeBodyControl/ckpt/release/model_encoder.onnx")
        sys.exit(1)

    encoder_path = sys.argv[1]
    check_encoder(encoder_path)
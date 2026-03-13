#!/usr/bin/env python
"""
检查 SONIC Encoder 模型的输入要求
"""

import sys
import os

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed")
    print("Install with: pip install onnxruntime")
    sys.exit(1)

def check_encoder(encoder_path):
    """检查encoder模型的输入形状"""
    if not os.path.isfile(encoder_path):
        print(f"Error: File not found: {encoder_path}")
        return

    print("="*70)
    print(f"Checking Encoder: {encoder_path}")
    print("="*70)

    try:
        session = ort.InferenceSession(encoder_path)

        print("\nEncoder Inputs:")
        for i, inp in enumerate(session.get_inputs()):
            print(f"  [{i}] {inp.name}")
            print(f"      shape: {inp.shape}")
            print(f"      type: {inp.type}")

        print("\nEncoder Outputs:")
        for i, out in enumerate(session.get_outputs()):
            print(f"  [{i}] {out.name}")
            print(f"      shape: {out.shape}")
            print(f"      type: {out.type}")

        print("\n" + "="*70)

        # 分析输入形状
        print("\n分析:")
        inputs = session.get_inputs()
        if len(inputs) >= 2:
            inp1_shape = inputs[1].shape
            print(f"  第二个输入 ({inputs[1].name}) 的形状: {inp1_shape}")

            if len(inp1_shape) == 3:
                print(f"    → 这是 (batch, history, features) 格式")
                print(f"    → features={inp1_shape[2]}")
                if inp1_shape[2] == 4:
                    print(f"    → ✓ 只需要单个四元数 (4,)，当前实现正确！")
                elif inp1_shape[2] == 96:
                    print(f"    → ✗ 需要24个四元数 (24*4=96)，当前实现错误！")
            elif len(inp1_shape) == 4:
                print(f"    → 这是 (batch, history, bodies, features) 格式")
                print(f"    → bodies={inp1_shape[2]}, features={inp1_shape[3]}")
                if inp1_shape[2] == 24 and inp1_shape[3] == 4:
                    print(f"    → ✗ 需要24个身体部位的四元数，当前实现错误！")

    except Exception as e:
        print(f"Error loading encoder: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_encoder_inputs.py <encoder_path>")
        print("\nExample:")
        print("  python check_encoder_inputs.py /path/to/model_encoder.onnx")
        sys.exit(1)

    encoder_path = sys.argv[1]
    check_encoder(encoder_path)
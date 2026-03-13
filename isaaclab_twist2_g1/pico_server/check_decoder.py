#!/usr/bin/env python
"""检查SONIC Decoder模型的输入要求"""
import sys
import onnxruntime as ort

def check(path):
    sess = ort.InferenceSession(path)
    print("Decoder Inputs:")
    for i, inp in enumerate(sess.get_inputs()):
        print(f"  [{i}] {inp.name}: shape={inp.shape}, type={inp.type}")
    print("Decoder Outputs:")
    for i, out in enumerate(sess.get_outputs()):
        print(f"  [{i}] {out.name}: shape={out.shape}, type={out.type}")

if __name__ == "__main__":
    check(sys.argv[1])
#!/usr/bin/env python3
"""Prepare an exported ONNX graph for C code generation.

Steps (these are the two fixes that were needed to get emx-onnx-cgen to accept the model):
  1. onnxsim.simplify   -- constant-fold and remove redundant nodes
  2. drop the `training_mode` attribute from BatchNormalization nodes
     (PyTorch exports it; the C generator rejects it)

Usage:
    python clean_onnx.py rawtfnet_classifier.onnx rawtfnet_classifier_clean.onnx
"""
import sys

import onnx
from onnxsim import simplify


def main(src, dst):
    model = onnx.load(src)
    model, ok = simplify(model)
    if not ok:
        sys.exit("onnxsim could not validate the simplified model")

    stripped = 0
    for node in model.graph.node:
        if node.op_type == "BatchNormalization":
            keep = [a for a in node.attribute if a.name != "training_mode"]
            if len(keep) != len(node.attribute):
                del node.attribute[:]
                node.attribute.extend(keep)
                stripped += 1

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"wrote {dst}  (simplified; training_mode removed from {stripped} BatchNorm nodes)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

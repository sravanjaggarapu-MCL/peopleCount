"""
export_openvino.py
==================

Converts the PyTorch weights (yolov8n.pt) into OpenVINO IR - the .xml graph
plus the .bin weights that the project actually runs on.

Run it once:

    python export_openvino.py              # at config.INFERENCE_SIZE
    python export_openvino.py --imgsz 320  # to match `main.py --imgsz 320`

It writes config.OPENVINO_MODEL_DIR (yolov8n_openvino_model/), containing:

    yolov8n.xml      the network topology
    yolov8n.bin      the weights
    metadata.yaml    class names, task, stride - what ultralytics reads back

Why bother converting at all
----------------------------
The .pt file runs through PyTorch, which is a training framework doing
inference as a side job. OpenVINO is Intel's inference-only runtime: it folds
constants, fuses layers and emits code tuned for this CPU's instruction set.
Same weights, and the same boxes - the counts were compared frame by frame
against the .pt and matched exactly - at 32 ms/frame instead of 55 ms on the
development machine.

That margin depends on config.OPENVINO_DEVICE and config.TORCH_THREADS, both
of which are documented where they are defined. Change them carelessly and
the IR becomes slower than the .pt it replaced.

Why the export is pinned to one image size
------------------------------------------
IR is compiled for a fixed input shape, so the exported graph hard-codes
config.INFERENCE_SIZE. Change that setting and this script must be re-run;
the detector checks for the mismatch and tells you so rather than quietly
running at the wrong size.
"""

import argparse
import shutil
import sys
from pathlib import Path

import config


def _patch_openvino_namespace():
    """Make ultralytics' exporter work on OpenVINO 2026.x.

    ultralytics 8.3.28 calls `openvino.runtime.save_model`. The
    `openvino.runtime` namespace was deprecated in 2025 and deleted in 2026,
    where the same function lives at the top level as `openvino.save_model`.
    Pointing the old name at the module itself satisfies the call without
    downgrading OpenVINO or editing anything inside site-packages.

    The guard means this quietly does nothing once ultralytics is updated.
    """
    import openvino as ov

    if not hasattr(ov, "runtime"):
        ov.runtime = ov


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert yolov8n.pt into the OpenVINO .xml/.bin the project runs on."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.INFERENCE_SIZE,
        help="square input size to compile for; must match the size the project "
             "runs at, i.e. config.INFERENCE_SIZE or main.py --imgsz "
             f"(default: {config.INFERENCE_SIZE})",
    )
    args = parser.parse_args()
    if args.imgsz % 32:
        print(f"ERROR: imgsz must be a multiple of 32 (the network stride); got {args.imgsz}.")
        return 1

    pt_path = Path(config.PYTORCH_MODEL_PATH)
    if not pt_path.exists():
        print(f"ERROR: {pt_path} not found - nothing to convert.")
        return 1

    _patch_openvino_namespace()
    from ultralytics import YOLO

    out_dir = Path(config.OPENVINO_MODEL_DIR)
    if out_dir.exists():
        print(f"Removing previous export at {out_dir} ...")
        shutil.rmtree(out_dir)

    print(f"Converting {pt_path} to OpenVINO IR at imgsz={args.imgsz} ...")
    # half=False keeps the weights FP32. FP16 halves the file but gains
    # nothing on a CPU, which has no native FP16 arithmetic and would just
    # convert back up on every layer.
    exported = YOLO(str(pt_path)).export(
        format="openvino",
        imgsz=args.imgsz,
        half=False,
    )

    # export() names the directory after the weights file, which is already
    # where we want it; this is just a safety net if ultralytics ever changes
    # that convention.
    exported = Path(exported)
    if exported.resolve() != out_dir.resolve():
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.move(str(exported), str(out_dir))

    xml = Path(config.MODEL_PATH)
    if not xml.exists():
        print(f"ERROR: export finished but {xml} is missing.")
        return 1

    print(f"\nDone. The project will now load {xml} (imgsz={args.imgsz}).")
    if args.imgsz != config.INFERENCE_SIZE:
        print(
            f"NOTE: config.INFERENCE_SIZE is {config.INFERENCE_SIZE}, so run the "
            f"project with  --imgsz {args.imgsz}  or the detector will stop and "
            f"tell you the two disagree."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

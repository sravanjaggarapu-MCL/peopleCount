"""
detector.py
===========

The YOLO model, in both of its modes:

    detect()  - find people in one frame, with no memory of any other frame
    track()   - find people AND give each a stable ID across frames (ByteTrack)

Why tracking is a separate thing from detection
-----------------------------------------------
Detection is memoryless: each frame is analysed from scratch, so nothing
connects the box in frame 12 to the box in frame 11. That is fine for a
headcount, but useless for entry/exit counting - to say "this person crossed
the line downwards" you must first be able to say "this person" across time.

How ByteTrack supplies that
---------------------------
Each frame it predicts where every existing track should have moved (a Kalman
filter assuming roughly constant velocity), then matches those predictions
against the new detections by box overlap (IoU).

Its namesake trick is a two-round match. Most trackers discard low-confidence
detections outright; ByteTrack does not:

    Round 1 - match existing tracks against HIGH confidence detections.
    Round 2 - take the tracks still unmatched, try them against the LOW
              confidence leftovers.

A person half-hidden behind someone else produces exactly such a weak
detection. Round 2 is what keeps their track alive through the occlusion
instead of killing it and reissuing a fresh ID - and every ID change like that
would become a double count once we are counting crossings.
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import yaml
from ultralytics import YOLO

import config


@contextlib.contextmanager
def _pin_openvino_device(device: str):
    """Force OpenVINO to compile for `device` instead of letting it choose.

    ultralytics 8.3 hard-codes device_name="AUTO" when it compiles the IR, and
    offers no argument to override it. AUTO is not a device: it starts the
    model on the CPU, compiles for every other device it can find in the
    background, then silently migrates to whichever it rates highest. On this
    machine that is the integrated GPU - and an Intel UHD iGPU runs this
    network at ~162 ms/frame against the CPU's ~29 ms. The migration happens
    after a few seconds, so the symptom is video that starts at full speed and
    then collapses, with nothing in the logs to say why.

    Rather than edit site-packages or pin an ultralytics version, we swap the
    one method for the duration of the load and put it straight back. The
    patch is active only while OUR model compiles; anything else using
    OpenVINO in this process is untouched.
    """
    original = ov.Core.compile_model

    def compile_on_device(self, model, device_name=None, config=None, **kwargs):
        return original(self, model, device_name=device, config=config, **kwargs)

    ov.Core.compile_model = compile_on_device
    try:
        yield
    finally:
        ov.Core.compile_model = original


@dataclass
class Detection:
    """One person found in one frame.

    A small plain-data class rather than raw tensors, so the rest of the code
    never has to know what shape ultralytics returns things in. If we swap YOLO
    for something else later, this is the only boundary that has to change.
    """

    x1: int          # left edge, pixels
    y1: int          # top edge (y grows DOWNWARDS in image coordinates)
    x2: int          # right edge
    y2: int          # bottom edge
    confidence: float
    track_id: int | None = None   # None in detect mode; an integer in track mode

    @property
    def foot_point(self) -> tuple[int, int]:
        """Bottom-centre of the box: where the person meets the floor.

        Entry/exit counting will test THIS point against the counting line
        rather than the box centre. On a bullet camera mounted above head
        height the body leans into the frame, so the centroid reaches the line
        while the person is still short of the doorway. The feet are where the
        person actually is.
        """
        return (self.x1 + self.x2) // 2, self.y2

    @property
    def head_point(self) -> tuple[int, int]:
        """Top-centre of the box: the crown of the head.

        The alternative to the feet, and the better one when the feet are not
        reliably visible - in a crowd the lower body is occluded first, and a
        bottom edge guessed from behind somebody else's shoulders wanders far
        more than a top edge does. See config.TRACK_POINT.
        """
        return (self.x1 + self.x2) // 2, self.y1

    @property
    def reference_point(self) -> tuple[int, int]:
        """The point this run counts, draws and trails people by.

        Read from config at call time rather than captured once, so the
        start-up prompt can settle it without anything here being rebuilt.
        Everything that cares about WHERE a person is goes through this, so
        the dot on screen and the point tested against the line can never
        drift apart.
        """
        return self.head_point if config.TRACK_POINT == "head" else self.foot_point

    @property
    def centroid(self) -> tuple[int, int]:
        """Middle of the box. Handy for drawing; not what we count on."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


class PersonDetector:
    """Wraps the YOLO model and turns its output into Detection objects."""

    def __init__(self, model_path: str | None = None, inference_size: int | None = None):
        self.model_path = model_path or config.MODEL_PATH
        self.inference_size = inference_size or config.INFERENCE_SIZE

        self._check_export()

        # The weights are pretrained on COCO - we are NOT training anything.
        # task= is passed explicitly because an OpenVINO directory, unlike a
        # .pt checkpoint, carries no record of what the network was built for;
        # without it ultralytics prints a warning and guesses.
        print(f"Loading {config.MODEL_XML} ...")
        with _pin_openvino_device(config.OPENVINO_DEVICE):
            self.model = YOLO(self.model_path, task=config.MODEL_TASK)
            # Compilation is lazy: ultralytics does not touch OpenVINO until
            # the first inference, so the warm-up has to happen inside the
            # patch or the device pin would miss it entirely.
            self._warm_up()

        # Deliberately last. Setting it any earlier accomplishes nothing:
        # ultralytics calls select_device() while building its predictor,
        # which resets torch's pool to the machine's core count, and the
        # predictor is not built until the first inference above.
        self._free_cores_for_openvino()
        print(f"Model ready (imgsz={self.inference_size}).")

    @staticmethod
    def _free_cores_for_openvino():
        """Stop PyTorch's idle threads from starving the OpenVINO ones.

        Torch is still imported - ultralytics uses it for NMS and the tracker -
        and on import it sizes an intra-op thread pool to the whole machine.
        Those threads do not sleep when idle, they spin-wait, so on a 4-core
        laptop they sit on every core burning cycles while OpenVINO is trying
        to use them. The two runtimes fight, and OpenVINO loses badly:

            torch threads = 7  ->  194 ms per frame
            torch threads = 4  ->  138 ms
            torch threads = 2  ->   55 ms
            torch threads = 1  ->   32 ms   (matches the model benchmarked alone)

        Nothing is given up by shrinking the pool. The network itself runs
        inside OpenVINO, which manages its own threads; what is left for torch
        is NMS on a handful of boxes, measured at ~1.6 ms and unchanged here.
        """
        if config.TORCH_THREADS:
            torch.set_num_threads(config.TORCH_THREADS)

    def _check_export(self):
        """Fail early, and with instructions, if the IR model is unusable.

        Two things can be wrong, and OpenVINO reports neither of them in a way
        that points at the fix:

        Missing - the IR is generated rather than downloaded, so a fresh clone
        has the .pt but not the .xml. Left alone, ultralytics would read the
        absent directory as a model NAME and try to fetch it from the
        internet, failing seconds later with an unrelated message.

        Wrong size - IR is compiled for one fixed input shape. Feed it any
        other and inference dies deep inside the runtime with "Failed to set
        tensor" and a C++ source location, nothing about imgsz. metadata.yaml,
        written beside the .xml at export time, records the shape it was built
        for, so we can compare before the first frame instead.
        """
        if not Path(config.MODEL_XML).exists():
            raise FileNotFoundError(
                f"{config.MODEL_XML} not found.\n"
                f"Build the OpenVINO model first:\n\n"
                f"    python export_openvino.py\n"
            )

        meta = Path(config.OPENVINO_MODEL_DIR) / "metadata.yaml"
        if not meta.exists():
            return  # nothing to compare against; let ultralytics proceed

        # Stored as [height, width]; our exports are square.
        imgsz = yaml.safe_load(meta.read_text()).get("imgsz")
        exported = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
        if exported and exported != self.inference_size:
            raise ValueError(
                f"{config.MODEL_XML} was exported for imgsz={exported}, but "
                f"this run wants imgsz={self.inference_size}.\n"
                f"Either re-export at that size:\n\n"
                f"    python export_openvino.py --imgsz {self.inference_size}\n\n"
                f"or run at the size already exported (--imgsz {exported}).\n"
            )

    def _warm_up(self):
        """Run one throwaway inference on a blank image.

        The first predict() call is dramatically slower than the rest - it
        allocates buffers, picks algorithms and fills caches on the way
        through, and under OpenVINO it is also where the graph is compiled for
        this particular CPU. Doing it now, before the window opens, keeps that
        stall out of the display loop where it looks like the video has
        frozen.
        """
        blank = np.zeros((self.inference_size, self.inference_size, 3), dtype=np.uint8)
        self.model.predict(blank, imgsz=self.inference_size, verbose=False)

    def detect(self, frame, conf: float | None = None) -> list[Detection]:
        """Find people in one frame. No IDs, no memory of previous frames."""
        results = self.model.predict(
            frame,
            classes=[config.PERSON_CLASS_ID],   # people only - see config.py
            conf=conf if conf is not None else config.DETECT_CONF_THRESHOLD,
            imgsz=self.inference_size,
            verbose=False,                      # silence per-frame console spam
        )
        # predict() returns one result per input image; we passed a single
        # frame, so results[0] is ours.
        return self._to_detections(results[0].boxes, with_ids=False)

    def track(self, frame, conf: float | None = None) -> list[Detection]:
        """Find people and give each a stable ID that persists across frames."""
        results = self.model.track(
            frame,
            # persist=True is the critical argument: it tells ultralytics this
            # frame continues the same video as the last call, so tracker state
            # carries over. Omit it and the tracker is rebuilt every frame -
            # everyone would be "ID 1", newly born, forever.
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=[config.PERSON_CLASS_ID],
            conf=conf if conf is not None else config.TRACK_CONF_THRESHOLD,
            imgsz=self.inference_size,
            verbose=False,
        )
        return self._to_detections(results[0].boxes, with_ids=True)

    @staticmethod
    def _to_detections(boxes, with_ids: bool) -> list[Detection]:
        """Convert ultralytics' Boxes object into our plain Detection list."""
        if boxes is None or len(boxes) == 0:
            return []

        # boxes.id is None when the tracker has no confirmed tracks yet - an
        # empty scene, or the first frames while tracks are still being
        # confirmed. Indexing it unguarded would crash the loop.
        has_ids = with_ids and boxes.id is not None

        # Pull the parallel arrays off the tensor in one go; cheaper than
        # touching each box individually. .xyxy is already in ORIGINAL FRAME
        # pixels - ultralytics undoes the imgsz letterboxing for us, so there
        # is no rescaling to do here.
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.int().cpu().numpy() if has_ids else [None] * len(xyxy)

        return [
            Detection(int(x1), int(y1), int(x2), int(y2),
                      float(conf), int(tid) if tid is not None else None)
            for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids)
        ]

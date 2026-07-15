import asyncio
import sys
import threading
import time

import depthai as dai
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn

FPS = 15
RGB_W, RGB_H = 640, 400
DEPTH_W, DEPTH_H = 640, 400

latest_frames: dict[str, dict] = {}
frame_lock = threading.Lock()
frame_ready = threading.Event()

_JET = np.array([
    [0, 0, 128],
    [0, 0, 255],
    [0, 128, 255],
    [0, 255, 255],
    [128, 255, 128],
    [255, 255, 0],
    [255, 128, 0],
    [255, 0, 0],
    [128, 0, 0],
], dtype=np.uint8)


def jet_colormap(gray: np.ndarray) -> np.ndarray:
    idx = (gray.astype(np.float32) / 255 * (_JET.shape[0] - 1)).astype(np.uint8)
    return _JET[idx]


def camera_loop(mxid: str, pipeline: dai.Pipeline, q_enc, q_depth):
    label = mxid[-6:]
    rgb_seq = 0
    depth_seq = 0
    try:
        while pipeline.isRunning():
            got_any = False

            if q_enc.has():
                got_any = True
                enc_frame = q_enc.get()
                rgb_seq += 1
                rgb_jpeg = bytes(enc_frame.getData())
                with frame_lock:
                    d = latest_frames.get(mxid)
                    if d is None:
                        d = {'label': label}
                        latest_frames[mxid] = d
                    d['rgb'] = rgb_jpeg
                    d['rgb_seq'] = rgb_seq
                frame_ready.set()

            if q_depth.has():
                got_any = True
                depth_frame = q_depth.get()
                depth_raw = depth_frame.getFrame()

                valid = depth_raw < 65535
                if valid.any():
                    max_val = np.percentile(depth_raw[valid], 95) or 1
                    norm = np.clip(
                        depth_raw.astype(np.float32) / max_val * 255, 0, 255
                    ).astype(np.uint8)
                    heatmap = cv2.rotate(jet_colormap(norm), cv2.ROTATE_180)
                else:
                    heatmap = np.zeros((DEPTH_H, DEPTH_W, 3), dtype=np.uint8)

                _, depth_buf = cv2.imencode('.jpg', heatmap, [cv2.IMWRITE_JPEG_QUALITY, 70])
                depth_jpeg = depth_buf.tobytes()
                depth_seq += 1

                with frame_lock:
                    d = latest_frames.get(mxid)
                    if d is None:
                        d = {'label': label}
                        latest_frames[mxid] = d
                    d.update(
                        depth=depth_jpeg, depth_seq=depth_seq,
                        min_d=int(depth_raw[valid].min()) if valid.any() else 0,
                        max_d=int(depth_raw[valid].max()) if valid.any() else 0,
                        avg_d=int(depth_raw[valid].mean()) if valid.any() else 0,
                    )
                frame_ready.set()

            if not got_any:
                time.sleep(0.001)
    except Exception as e:
        print(f"[{label}] Error: {e}")
    finally:
        with frame_lock:
            latest_frames.pop(mxid, None)
        pipeline.stop()
        pipeline.wait()


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    prev_rgb_seq: dict[str, int] = {}
    prev_depth_seq: dict[str, int] = {}
    try:
        while True:
            with frame_lock:
                data = {k: dict(v) for k, v in latest_frames.items()}

            has_new = any(
                d.get('rgb_seq', 0) != prev_rgb_seq.get(mxid, 0) or
                d.get('depth_seq', 0) != prev_depth_seq.get(mxid, 0)
                for mxid, d in data.items()
            )

            if has_new:
                meta = {}
                for mxid, d in data.items():
                    meta[mxid] = {
                        'label': d['label'],
                        'rgb_seq': d.get('rgb_seq', 0),
                        'depth_seq': d.get('depth_seq', 0),
                        'min_d': d.get('min_d', 0),
                        'avg_d': d.get('avg_d', 0),
                        'max_d': d.get('max_d', 0),
                    }
                await ws.send_json(meta)

                for mxid, d in data.items():
                    label_bytes = d['label'].encode()
                    rgb_seq = d.get('rgb_seq', 0)
                    if rgb_seq != prev_rgb_seq.get(mxid, 0):
                        prev_rgb_seq[mxid] = rgb_seq
                        await ws.send_bytes(label_bytes + b'\x00' + d['rgb'])
                    depth_seq = d.get('depth_seq', 0)
                    if depth_seq != prev_depth_seq.get(mxid, 0):
                        prev_depth_seq[mxid] = depth_seq
                        await ws.send_bytes(label_bytes + b'\x01' + d['depth'])

            await asyncio.sleep(1.0 / FPS)
    except WebSocketDisconnect:
        pass


def main():
    device_infos = dai.Device.getAllAvailableDevices()
    if not device_infos:
        print("No Oak-D devices found")
        sys.exit(1)

    print(f"Found {len(device_infos)} device(s):")
    threads_data = []
    for info in device_infos:
        mxid = info.getDeviceId()
        print(f"  {mxid}")

        try:
            device = dai.Device(dai.DeviceInfo(mxid))
        except RuntimeError as e:
            print(f"  Failed to open {mxid}: {e}")
            continue

        pipeline = dai.Pipeline(defaultDevice=device)

        cam_rgb = pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A,
            sensorFps=FPS,
        )
        cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

        rgb_raw = cam_rgb.requestOutput(
            size=(RGB_W, RGB_H),
            type=dai.ImgFrame.Type.NV12,
            fps=FPS,
        )

        encoder = pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.MJPEG)
        encoder.setQuality(80)
        rgb_raw.link(encoder.input)

        left_cam = pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_B,
            sensorFps=FPS,
        )
        right_cam = pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_C,
            sensorFps=FPS,
        )

        left_out = left_cam.requestOutput(
            size=(640, 400),
            type=dai.ImgFrame.Type.GRAY8,
            fps=FPS,
        )
        right_out = right_cam.requestOutput(
            size=(640, 400),
            type=dai.ImgFrame.Type.GRAY8,
            fps=FPS,
        )

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(DEPTH_W, DEPTH_H)

        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        cfg = stereo.initialConfig
        cfg.costMatching.confidenceThreshold = 55
        cfg.postProcessing.speckleFilter.enable = True
        cfg.postProcessing.speckleFilter.speckleRange = 200
        cfg.postProcessing.spatialFilter.enable = True
        cfg.postProcessing.spatialFilter.holeFillingRadius = 2
        cfg.postProcessing.spatialFilter.numIterations = 1
        cfg.postProcessing.thresholdFilter.minRange = 300
        cfg.postProcessing.thresholdFilter.maxRange = 20000

        left_out.link(stereo.left)
        right_out.link(stereo.right)

        q_enc = encoder.bitstream.createOutputQueue(maxSize=1, blocking=False)
        q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()

        threads_data.append((mxid, pipeline, q_enc, q_depth))

    if not threads_data:
        print("No devices could be opened")
        sys.exit(1)

    for mxid, pipeline, q_enc, q_depth in threads_data:
        t = threading.Thread(target=camera_loop, args=(mxid, pipeline, q_enc, q_depth), daemon=True)
        t.start()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()

import asyncio
import base64
import io
import threading

import depthai as dai
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

FPS = 5
DEPTH_W, DEPTH_H = 640, 400

latest_frames = {}
frame_lock = threading.Lock()

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


def array_to_b64(arr: np.ndarray, quality: int = 80) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def camera_loop():
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A,
        sensorFps=FPS
    )
    
    rgb_output = cam_rgb.requestOutput(
        size=(1920, 1080), 
        type=dai.ImgFrame.Type.BGR888p, 
        fps=FPS
    )

    left_cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_B,
        sensorFps=FPS
    )
    right_cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_C,
        sensorFps=FPS
    )

    left_out = left_cam.requestOutput(
        size=(640, 400), 
        type=dai.ImgFrame.Type.GRAY8, 
        fps=FPS
    )
    right_out = right_cam.requestOutput(
        size=(640, 400), 
        type=dai.ImgFrame.Type.GRAY8, 
        fps=FPS
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(DEPTH_W, DEPTH_H)
    
    left_out.link(stereo.left)
    right_out.link(stereo.right)

    q_rgb = rgb_output.createOutputQueue(maxSize=1, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

    pipeline.start()

    with pipeline:
        while pipeline.isRunning():
            if q_rgb.has() and q_depth.has():
                rgb_frame = q_rgb.get()
                depth_frame = q_depth.get()

                rgb = rgb_frame.getCvFrame()[:, :, ::-1]
                depth_raw = depth_frame.getFrame()

                valid = depth_raw < 65535
                if valid.any():
                    max_val = np.percentile(depth_raw[valid], 95)
                    norm = np.clip(
                        depth_raw.astype(np.float32) / max_val * 255, 0, 255
                    ).astype(np.uint8)
                    heatmap = jet_colormap(norm)
                else:
                    heatmap = np.zeros((DEPTH_H, DEPTH_W, 3), dtype=np.uint8)

                rgb_b64 = array_to_b64(rgb)
                depth_b64 = array_to_b64(heatmap)

                with frame_lock:
                    latest_frames.update(
                        rgb=rgb_b64,
                        depth=depth_b64,
                        min_d=int(depth_raw[valid].min()) if valid.any() else 0,
                        max_d=int(depth_raw[valid].max()) if valid.any() else 0,
                        avg_d=int(depth_raw[valid].mean()) if valid.any() else 0,
                    )


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    prev = {}
    try:
        while True:
            with frame_lock:
                data = latest_frames.copy()
            if data and data != prev:
                await ws.send_json(data)
                prev = data
            await asyncio.sleep(1.0 / FPS)
    except WebSocketDisconnect:
        pass


def main():
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
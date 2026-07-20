import asyncio
import sys
import threading
import time

import depthai as dai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn

FPS = 15
RGB_W, RGB_H = 640, 400
DEPTH_W, DEPTH_H = 640, 400

latest_frames: dict[str, dict] = {}
frame_lock = threading.Lock()
frame_ready = threading.Event()


def camera_loop(mxid: str, pipeline: dai.Pipeline, q_rgb, q_depth, q_meta):
    label = mxid[-6:]
    rgb_seq = 0
    depth_seq = 0
    try:
        while pipeline.isRunning():
            got_any = False

            if q_rgb.has():
                got_any = True
                enc_frame = q_rgb.get()
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
                depth_enc = q_depth.get()
                depth_jpeg = bytes(depth_enc.getData())
                depth_seq += 1
                with frame_lock:
                    d = latest_frames.get(mxid)
                    if d is None:
                        d = {'label': label}
                        latest_frames[mxid] = d
                    d['depth'] = depth_jpeg
                    d['depth_seq'] = depth_seq
                frame_ready.set()

            if q_meta.has():
                got_any = True
                meta_frame = q_meta.get()
                spatial_data = meta_frame.getSpatialLocations()
                if spatial_data:
                    sd = spatial_data[0]
                    with frame_lock:
                        d = latest_frames.get(mxid)
                        if d is None:
                            d = {'label': label}
                            latest_frames[mxid] = d
                        d['min_d'] = int(sd.depthMin)
                        d['max_d'] = int(sd.depthMax)
                        d['avg_d'] = int(sd.depthAverage)

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

        # --- RGB pipeline (unchanged) ---
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

        rgb_encoder = pipeline.create(dai.node.VideoEncoder)
        rgb_encoder.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.MJPEG)
        rgb_encoder.setQuality(80)
        rgb_raw.link(rgb_encoder.input)

        # --- Stereo pipeline ---
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
        stereo.setSubpixel(False)

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

        # --- Disparity → ImageManip (rotate) → VideoEncoder (MJPEG) ---
        disp_manip = pipeline.create(dai.node.ImageManip)
        disp_manip.initialConfig.setRotationDeg(180)
        stereo.disparity.link(disp_manip.inputImage)

        depth_encoder = pipeline.create(dai.node.VideoEncoder)
        depth_encoder.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.MJPEG)
        depth_encoder.setQuality(80)
        disp_manip.out.link(depth_encoder.input)

        # --- Depth → SpatialLocationCalculator → XLinkOut ---
        spat_calc = pipeline.create(dai.node.SpatialLocationCalculator)
        spat_calc.setWaitForConfigInput(False)
        cfg_data = dai.SpatialLocationCalculatorConfigData()
        cfg_data.roi = dai.Rect(dai.Point2f(0, 0), dai.Point2f(1, 1))
        cfg_data.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
        spat_calc.initialConfig.addROI(cfg_data)
        stereo.depth.link(spat_calc.inputDepth)

        spat_calc_out = pipeline.create(dai.node.XLinkOut)
        spat_calc_out.setStreamName("spatialData")
        spat_calc.spatialData.link(spat_calc_out.input)

        # --- Output queues ---
        q_rgb = rgb_encoder.bitstream.createOutputQueue(maxSize=1, blocking=False)
        q_depth = depth_encoder.bitstream.createOutputQueue(maxSize=1, blocking=False)
        q_meta = spat_calc_out.output.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        threads_data.append((mxid, pipeline, q_rgb, q_depth, q_meta))

    if not threads_data:
        print("No devices could be opened")
        sys.exit(1)

    for mxid, pipeline, q_rgb, q_depth, q_meta in threads_data:
        t = threading.Thread(target=camera_loop, args=(mxid, pipeline, q_rgb, q_depth, q_meta), daemon=True)
        t.start()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()


        # --- Depth → SpatialLocationCalculator (Metrics) ---
        spat_calc = pipeline.create(dai.node.SpatialLocationCalculator)
        cfg_data = dai.SpatialLocationCalculatorConfigData()
        cfg_data.roi = dai.Rect(dai.Point2f(0, 0), dai.Point2f(1, 1))
        cfg_data.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
        spat_calc.initialConfig.addROI(cfg_data)
        stereo.depth.link(spat_calc.inputDepth)

        # --- Output queues ---
        q_rgb = rgb_encoder.bitstream.createOutputQueue(maxSize=1, blocking=False)
        q_depth = stereo.disparity.createOutputQueue(maxSize=1, blocking=False)
        q_meta = spat_calc.out.createOutputQueue(maxSize=1, blocking=False)

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
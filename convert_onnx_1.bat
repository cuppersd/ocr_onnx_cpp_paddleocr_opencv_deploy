paddle2onnx  --model_dir ./ch_PP-OCRv4_rec_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx --opset_version 10 --input_shape_dict="{'x':[-1,3,-1,-1]}" --enable_onnx_checker True
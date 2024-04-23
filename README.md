# ocr_onnx_cpp_paddleocr_opencv_deploy
## 将paddleocr模型转化onnx
* `paddle2onnx  --model_dir ./inference/ch_ppocr_mobile_v2.0_cls_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./inferencecls_onnx/model.onnx --opset_version 10 --input_shape_dict="{'x':[-1,3,-1,-1]}" --enable_onnx_checker True`
* 输入是动态尺寸，需要将其转化为固定尺寸。
* `python -m paddle2onnx.optimize --input_model model.onnx --output_model new_model.onnx --input_shape_dict "{'x':[1,3,32,320]}"`




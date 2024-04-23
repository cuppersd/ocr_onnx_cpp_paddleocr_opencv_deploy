import math

import cv2
print(cv2.__version__)
import onnxruntime
import numpy as np
use_dnn = True
small_rec_file = "new_model.onnx"
if use_dnn:
    onet_rec_session = cv2.dnn.readNetFromONNX(small_rec_file)
    print(onet_rec_session.__dir__())
else:
    onet_rec_session = onnxruntime.InferenceSession(small_rec_file)



## 根据推理结果解码识别结果
class process_pred(object):
    def __init__(self, character_dict_path=None, character_type='ch', use_space_char=False):
        self.character_str = ''
        with open(character_dict_path, 'rb') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip('\n').strip('\r\n')
                self.character_str += line
        if use_space_char:
            self.character_str += ' '
        dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        print(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                print(len(self.character))
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, label=None):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label


postprocess_op = process_pred('ppocr_keys_v1.txt', 'ch', True)


def resize_norm_img(img, max_wh_ratio):
    imgC, imgH, imgW = [int(v) for v in "3, 48, 100".split(",")]
    assert imgC == img.shape[2]
    imgW = int((32 * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH),cv2.INTER_LINEAR)

    resized_image = resized_image.astype('float32')
    # print(resized_image)
    resized_image = resized_image.transpose((2, 0, 1)) / 255

    resized_image -= 0.5
    resized_image /= 0.5
    print("resized_image.shape")
    # print(resized_image.shape)

    # padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    # padding_im[:, :, 0:resized_w] = resized_image
    padding_im = cv2.copyMakeBorder(resized_image, 0, 0, 0,imgW - w, cv2.BORDER_CONSTANT,value=[0,0,0] )
    print(resized_image.shape)
    return padding_im


def get_img_res(onnx_model, img, process_op):
    h, w = img.shape[:2]
    print(h, w)
    img = resize_norm_img(img, w * 1.0 / h)
    img = img[np.newaxis, :]
    print(img.shape)
    # print(img)
    if use_dnn:
        onnx_model.setInput(img)  # 设置模型输入
        outs = onnx_model.forward()  # 推理出结果
    else:
        inputs = {onnx_model.get_inputs()[0].name: img}
        outs = onnx_model.run(None, inputs)
        outs = outs[0]
    print(outs.shape)
    print(outs[0][0][0])
    result = process_op(outs)
    return result


pic = cv2.imread(r"tt.png")

# pic=cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
res = get_img_res(onet_rec_session, pic, postprocess_op)

print(res)
import random
import time
from PIL import Image
import torch
import numpy as np
from .ootd_utils import get_mask_location, hole_fill, refine_mask  # 引入工具函数
from .openpose.run_openpose import OpenPose
from .humanparsing.run_parsing import Parsing
import os
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms import ToTensor

class OOTDiffusion:
    def __init__(self, root: str, model_type: str = "hd"):
        """初始化 OOTDiffusion 类，设定设备和模型路径"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model_type = model_type
        self.repo_root = root

        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 模型路径（使用相对路径拼接绝对路径）
        body_model_path = os.path.join(current_dir, "models", "body_pose_model.pth")
        atr_model_path = os.path.join(current_dir, "models", "parsing_atr.onnx")
        lip_model_path = os.path.join(current_dir, "models", "parsing_lip.onnx")

        # 初始化模型
        self.parsing_model = Parsing(atr_model_path, lip_model_path)
        self.openpose_model = OpenPose(body_model_path, device=self.device)

    def preprocess_image(self, image_path):
        # (1,H,W,3) -> (3,H,W)
        image_path = image_path.squeeze(0)
        image_path = image_path.permute((2, 0, 1))
        image_path = to_pil_image(image_path)
        image = image_path
    
        return image

    def predict_keypoints_and_parse(self, model_image):
        """使用预训练模型预测关键点和人体解析结果"""
        # 获取原图尺寸
        original_width, original_height = model_image.size

        # 调整图像大小为 OpenPose 模型所需的尺寸
        input_image = model_image.resize((384, 512))  # OpenPose 模型输入为 512x384

        # 调用解析模型
        parse_result, face_mask = self.parsing_model(input_image)

        # 保存解析结果图
        if parse_result.mode != 'RGB':
                parse_result_RGB = parse_result.convert('RGB')
        parse_result_RGB.save(os.path.join('output', 'parse_result_RGB.jpg'))

        # 调用 OpenPose 模型
        keypoints = self.openpose_model(input_image)

        # 打印出解析结果和关键点信息
        print("Keypoints:", keypoints)
        print("Parse result:", parse_result)   

        return parse_result, keypoints, original_width, original_height

    def generate_mask(self, category, model_type, parse_result, keypoints, image_path1, original_width, original_height):
        """根据解析结果和关键点生成蒙版"""
        mask, mask_gray = get_mask_location(model_type, category, parse_result, keypoints)
        image_path1 = image_path1.resize((original_width, original_height), Image.NEAREST)
        mask = mask.resize((original_width, original_height), Image.NEAREST)
        mask_gray = mask_gray.resize((original_width, original_height), Image.NEAREST)
        masked_vton_img = Image.composite(mask_gray, image_path1, mask)

 	# 将图片转换为 Tensor
        # pil(H,W,3) -> tensor(H,W,3)
        masked_vton_img = masked_vton_img.convert("RGB")
        masked_vton_img = to_tensor(masked_vton_img)
        masked_vton_img = masked_vton_img.permute((1, 2, 0)).unsqueeze(0)
        mask = mask.convert("RGB")
        # 转换为 Tensor
        mask = ToTensor()(mask)  # 输出形状为 (C, H, W)
        mask = mask.permute((0, 1, 2)).squeeze(0)
        mask_first = mask[0]
        # 添加 Batch 维度
        mask_first = mask_first.unsqueeze(0)  

        return mask_first, masked_vton_img

    def __call__(self, image_path, category="upper_body"):
        """主方法，用于处理图像并生成结果"""
        model_image = self.preprocess_image(image_path)

        # 预测关键点和人体解析结果
        parse_result, keypoints, original_width, original_height = self.predict_keypoints_and_parse(model_image)

        # 生成蒙版
        mask, masked_vton_img = self.generate_mask(category, self.model_type, parse_result, keypoints, model_image, original_width, original_height)

        return mask, masked_vton_img

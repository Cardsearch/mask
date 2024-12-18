# __init__.py
from .inference_ootd import OOTDiffusion
from PIL import Image
import numpy as np

__all__ = [
    "preprocess_image",
    "predict_keypoints_and_parse",
    "generate_mask",
    "apply_mask_to_image",
    "get_mask_location",
    "hole_fill",
    "refine_mask"
]

class MaskGenerator:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图片
                "category": (["upper_body", "lower_body", "dresses"], {"default": "upper_body"}),  # 分割类别
                "model_type": (["hd", "dc"], {"default": "hd"}),  # 模型类型
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")  # 输出两张图片
    RETURN_NAMES = ("mask_image", "mask")  # 输出名称
    FUNCTION = "generate_mask"
    CATEGORY = "Image Processing"

    def generate_mask(self, image, category, model_type):
        # 初始化 OOTDiffusion 实例
        ootd = OOTDiffusion(image, model_type)  # 模型类型可以是 "hd" 或 "dc"
        # 调用主方法进行处理
        mask, masked_vton_img = ootd(
            image_path=image,  # 传入图像路径
            category=category
        )

        return masked_vton_img, mask

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "MaskGenerator": MaskGenerator,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskGenerator": "Mask Generator",
}

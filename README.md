# ComfyUI Mask Plugin

## 介绍
这是一个用于 ComfyUI 的插件，提供图像分割和蒙版生成功能。

## 安装
1. 安装依赖：
    ```
    pip install -r requirements.txt
    ```
2. 将插件目录添加到 ComfyUI 插件目录中。

## 使用
- 在 ComfyUI 中使用 `Mask Generator` 节点来生成图像的蒙版。
- 节点支持的类别包括：
  - `head`
  - `upper_body`
  - `lower_body`

## 文件结构
- `mask_generator.py`: 定义 `MaskGeneratorNode` 节点。
- `mask_utils.py`: 提供图像分割工具函数。
- `utils.py`: 包含常用工具函数和基础 `Node` 类。

from gradio_client import Client, handle_file
import shutil
import os
import argparse
import cv2
from tqdm import tqdm
from PIL import Image

# 初始化Qwen-Image-Edit API客户端
client = Client("http://10.59.67.2:5012/")

def find_video_files(root_dir):
    """查找目录下所有视频文件"""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    video_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                video_files.append(os.path.join(dirpath, filename))
    return video_files

def extract_first_frame(video_path, output_dir):
    """从视频中提取首帧并保存"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return None
        
        # 读取首帧
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"错误：无法读取视频首帧 {video_path}")
            return None
        
        # 保存首帧（原始尺寸，增强时统一转为720p）
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_path = os.path.join(output_dir, f"{base_name}_first_frame.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path
    except Exception as e:
        print(f"提取首帧失败 {video_path}：{str(e)}")
        return None

def generate_augmented_frame(client, image_path, prompt, prompt_id, output_path, target_width=1280, target_height=720):
    """调用API生成增强首帧（默认输出720p）"""
    try:
        # 固定输出720p（1280x720），无需获取原始尺寸
        width, height = target_width, target_height
        
        # 调用Qwen-Image-Edit API
        result = client.predict(
            image1=handle_file(image_path),
            image2=None,
            image3=None,
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            true_guidance_scale=1,
            num_inference_steps=4,
            rewrite_prompt=False,
            height=height,
            width=width,
            api_name="/infer"
        )
        src_path = result[0]

        # 保存增强结果
        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        dst_path = os.path.join(output_path, f"{name}_aug_prompt{prompt_id}{ext}")
        shutil.move(src_path, dst_path)
        return True
    except FileNotFoundError:
        print(f"错误：源图像不存在 {image_path}")
    except Exception as e:
        print(f"增强失败 {image_path}：{str(e)}")
    return False

def process_videos(source, output_root, target_width=1280, target_height=720):
    """处理视频：提取首帧→生成720p增强首帧"""
    # 核心Prompt列表：针对异常攀高场景设计（覆盖位置/环境/姿态/干扰等维度）
    prompt_list = [
        # 攀爬位置+基础姿态
        "工人在无安全防护的脚手架外侧攀爬，单手握持横杆，身体悬空，工业车间背景，正常光线，无遮挡",
        "工人攀爬裸露的金属管道支架（直径约30cm），双脚交替向上，携带扳手工具，化工厂背景，轻微粉尘",
        "工人在设备顶部边缘（宽度<50cm）攀爬，身体前倾保持平衡，制造业车间背景，有机械噪音提示（视觉无显示）",
        # 环境干扰+视角
        "摄像头俯拍：工人攀爬锈蚀的铁架，画面有轻微监控噪点，逆光导致工人轮廓偏暗，背景有堆放的建材",
        "摄像头侧拍：工人攀爬室外脚手架，弱光环境（模拟黄昏），被部分防护网遮挡腿部，带轻微镜头畸变",
        "摄像头45°斜拍：工人攀爬垂直梯子（无安全笼），画面有雨滴特效（模拟雨天作业），远处有厂房烟囱",
        # 姿态变异+遮挡
        "工人攀爬时脚下打滑（单脚悬空），双手紧握攀爬物，被上方掉落的绳索轻微遮挡肩部，正常光线",
        "工人半蹲在狭窄平台边缘攀爬，身体重心偏移，被工具箱部分遮挡腰部，强光照射（镜头有光晕）",
        "工人背向摄像头攀爬斜梯，双手抓握两侧栏杆，背部被管道轻微遮挡，画面边缘轻微模糊（模拟监控焦距）",
        # 极端场景补充
        "工人未系安全绳攀爬高约5米的脚手架，大风导致身体摇晃，背景有其他作业人员（非攀爬状态），正常光线",
        "夜间作业：工人借助头灯攀爬金属架，光线集中在手部，周围环境昏暗，画面有监控红外夜视效果（黑白噪点）",
        "工人攀爬倾斜的传送带支架，脚下有油污（视觉可见），身体向一侧倾斜保持平衡，被部分设备遮挡脚部"
    ]

    # 确定处理对象（视频目录或单个视频文件）
    if os.path.isdir(source):
        video_files = find_video_files(source)
        print(f"发现 {len(video_files)} 个视频文件，开始处理...")
    elif os.path.isfile(source) and source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        video_files = [source]
        print(f"开始处理单个视频：{source}")
    else:
        print(f"错误：无效的视频源 {source}")
        return

    # 定义输出目录结构
    original_frames_dir = os.path.join(output_root, "original_first_frames")  # 原始首帧（保留原尺寸）
    augmented_frames_dir = os.path.join(output_root, "augmented_first_frames_720p")  # 720p增强首帧

    # 批量处理视频
    for video_path in tqdm(video_files, desc="视频处理进度"):
        # 提取首帧
        frame_path = extract_first_frame(video_path, original_frames_dir)
        if not frame_path:
            continue  # 提取失败则跳过
        
        # 生成720p增强首帧
        for prompt_id, prompt in enumerate(prompt_list):
            generate_augmented_frame(
                client,
                frame_path,
                prompt,
                prompt_id,
                augmented_frames_dir,
                target_width,
                target_height
            )

def main():
    parser = argparse.ArgumentParser(description='异常攀高视频首帧提取与增强工具（默认输出720p）')
    parser.add_argument('--source', default="", help='视频源（单个视频路径或视频目录）')
    parser.add_argument('--output', default="", help='输出根目录（会自动创建原始首帧和720p增强首帧子目录）')
    parser.add_argument('--width', type=int, default=1280, help='增强图片宽度（默认1280，720p标准）')
    parser.add_argument('--height', type=int, default=720, help='增强图片高度（默认720，720p标准）')
    
    args = parser.parse_args()

    # 执行处理流程（固定输出指定尺寸，默认720p）
    process_videos(
        args.source,
        args.output,
        target_width=args.width,
        target_height=args.height
    )

if __name__ == "__main__":
    main()

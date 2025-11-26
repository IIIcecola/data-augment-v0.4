from gradio_client import Client, handle_file
import shutil
import os
import argparse
import cv2
import random
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
            return None, None
        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"错误：无法读取视频首帧 {video_path}")
            return None, None
        
        # 获取帧尺寸
        height, width = frame.shape[:2]
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_path = os.path.join(output_dir, f"{base_name}_first_frame.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path, (width, height)
    except Exception as e:
        print(f"提取首帧失败 {video_path}：{str(e)}")
        return None, None

def extract_last_frame(video_path, output_dir):
    """从视频中提取尾帧并保存"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return None, None
        
        # 获取视频总帧数并定位到最后一帧
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"错误：无法读取视频尾帧 {video_path}")
            return None, None
        
        # 获取帧尺寸
        height, width = frame.shape[:2]
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_path = os.path.join(output_dir, f"{base_name}_last_frame.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path, (width, height)
    except Exception as e:
        print(f"提取尾帧失败 {video_path}：{str(e)}")
        return None, None

def generate_augmented_frame(client, image_path, prompt, prompt_id, output_path, target_width=1280, target_height=720):
    """调用API生成增强帧（默认输出720p）"""
    try:
        width, height = target_width, target_height
        
        result = client.predict(
            image1=handle_file(image_path),
            image2=None,
            image3=None,
            prompt=prompt,
            seed=random.randint(0, 10000),  # 增加随机种子提升多样性
            randomize_seed=True,
            true_guidance_scale=1,
            num_inference_steps=4,
            rewrite_prompt=False,
            height=height,
            width=width,
            api_name="/infer"
        )
        src_path = result[0]

        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        # 命名规则：原帧名_aug_prompt{id}（确保首尾帧同prompt_id可配对）
        dst_path = os.path.join(output_path, f"{name}_aug_prompt{prompt_id}{ext}")
        shutil.move(src_path, dst_path)
        return True, (width, height)
    except FileNotFoundError:
        print(f"错误：源图像不存在 {image_path}")
    except Exception as e:
        print(f"增强失败 {image_path}：{str(e)}")
    return False, None

def generate_prompts(target_count=None):
    """生成多样化的工人攀爬增强Prompt（复用person_fall2的生成逻辑）"""
    # 位置多样性
    positions = [
        "图像左侧区域", "图像右侧区域", "图像中央区域",
        "图像前景", "图像中景", "图像背景"
    ]
    
    # 衣着多样性
    clothes = [
        "蓝色工装服", "红色安全背心", "黄色反光马甲",
        "灰色工作服", "黑色夹克", "绿色劳保服",
        "深蓝色连体工装", "橙色安全服", "白色衬衫+深色裤子",
        "迷彩工作服", "棕色工装裤+蓝色上衣", "藏青色工作套装"
    ]
    
    # 体型多样性
    body_types = [
        "体型中等的", "体型偏瘦的", "体型偏胖的",
        "身材高大的", "身材矮小的", "体型健壮的",
        "体型匀称的"
    ]
    
    # 年龄多样性
    ages = [
        "20-30岁的年轻人", "30-40岁的中年人",
        "40-50岁的中年人", "50-60岁的老年人"
    ]
    
    # 光线条件
    light_conditions = [
        "正常光线条件下", "弱光环境中", "逆光条件下",
        "侧光照射下", "强光环境（带轻微光晕）", "昏暗环境（可辨识细节）"
    ]
    
    # 监控特效
    monitor_effects = [
        "带有轻微监控噪点", "边缘轻微模糊（模拟监控焦距）",
        "轻微偏色（模拟监控摄像头特性）", "低分辨率质感（模拟监控画面）"
    ]

    # 性别
    genders = [
        "男性", "女性"
    ]

    # 攀爬物类型
    climbing_objects = [
        "金属梯子", "脚手架", "管道", "铁塔", "电线杆", "平台护栏"
    ]

    # 组合生成prompt
    prompts = []
    for pos in positions:
        for cloth in clothes:
            for body in body_types:
                for age in ages:
                    for light in light_conditions:
                        for effect in monitor_effects:
                            for gender in genders:
                                for obj in climbing_objects:
                                    prompt = (f"在{pos}有一名{age}，{gender}，{body}工人，穿着{cloth}，正在攀爬{obj}。"
                                             f"攀爬动作保持不变，{light}，{effect}，其他场景元素不变，"
                                             "符合工业监控场景视角，自然融入背景，无明显合成痕迹。")
                                    prompts.append(prompt)
    if target_count is not None and target_count > 0:
        # 确保目标数量不超过总数量
        target_count = min(target_count, len(prompts))
        return random.sample(prompts, target_count)
    return prompts

def process_videos(source, output_root, frame_type="both", target_width=None, target_height=None, target_prompt_count=None):
    """处理视频：提取指定帧→生成匹配的增强帧"""
    # 生成多样化prompt
    prompt_list = generate_prompts()
    print(f"已生成 {len(prompt_list)} 种不同的增强Prompt")

    # 确定处理对象
    if os.path.isdir(source):
        video_files = find_video_files(source)
        print(f"发现 {len(video_files)} 个视频文件，开始处理...")
    elif os.path.isfile(source) and source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        video_files = [source]
        print(f"开始处理单个视频：{source}")
    else:
        print(f"错误：无效的视频源 {source}")
        return

    # 输出目录结构
    original_first_dir = os.path.join(output_root, "original_first_frames")
    original_last_dir = os.path.join(output_root, "original_last_frames")
    augmented_first_dir = os.path.join(output_root, "augmented_first_frames")
    augmented_last_dir = os.path.join(output_root, "augmented_last_frames")

    # 批量处理视频
    for video_path in tqdm(video_files, desc="视频处理进度"):
        frames_to_process = []
        first_frame = None
        last_frame = None
        first_size = None
        last_size = None

        # 根据帧类型参数提取对应帧
        if frame_type in ["first", "both"]:
            first_frame, first_size = extract_first_frame(video_path, original_first_dir)
            if first_frame:
                frames_to_process.append(("first", first_frame, first_size, augmented_first_dir))
        
        if frame_type in ["last", "both"]:
            last_frame, last_size = extract_last_frame(video_path, original_last_dir)
            if last_frame:
                frames_to_process.append(("last", last_frame, last_size, augmented_last_dir))

        # 检查是否有可处理的帧
        if not frames_to_process:
            print(f"跳过视频 {video_path}（无有效帧可处理）")
            continue

        # 验证首尾帧尺寸（如果都需要处理）
        if frame_type == "both" and first_size != last_size:
            print(f"警告：视频 {video_path} 首尾帧尺寸不一致（首帧：{first_size}，尾帧：{last_size}），可能影响配对效果")

        # 用相同的prompt和ID增强指定帧（确保配对一致性）
        for prompt_id, prompt in enumerate(prompt_list):
            # 记录首帧增强尺寸用于尾帧匹配
            first_aug_size = None
            
            for frame_info in frames_to_process:
                frame_type, frame_path, frame_size, aug_dir = frame_info
                
                # 增强首帧时记录尺寸
                if frame_type == "first":
                    success, first_aug_size = generate_augmented_frame(
                        client, frame_path, prompt, prompt_id, aug_dir, target_width, target_height
                    )
                # 增强尾帧时复用首帧尺寸（如果存在）
                else:
                    if first_aug_size:
                        generate_augmented_frame(
                            client, frame_path, prompt, prompt_id, aug_dir,
                            target_width=first_aug_size[0], target_height=first_aug_size[1]
                        )
                    else:
                        generate_augmented_frame(
                            client, frame_path, prompt, prompt_id, aug_dir, target_width, target_height
                        )

def main():
    parser = argparse.ArgumentParser(description='异常攀高视频帧提取与匹配增强工具')
    parser.add_argument('--source', required=True, help='视频源（单个视频路径或视频目录）')
    parser.add_argument('--output', required=True, help='输出根目录（自动创建子目录存储原始/增强帧）')
    parser.add_argument('--frame-type', choices=['first', 'last', 'both'], default='both',
                      help='指定生成的帧类型：first（仅首帧）、last（仅尾帧）、both（两者都生成，默认）')
    parser.add_argument('--width', type=int, default=1280, help='增强图片宽度（默认1280），None则使用原始尺寸')
    parser.add_argument('--height', type=int, default=720, help='增强图片高度（默认720）')
    parser.add_argument('--prompt-count', type=int, default=None, help='指定生成的Prompt数量，None则生成所有可能的组合')
    
    args = parser.parse_args()

    process_videos(
        args.source,
        args.output,
        frame_type=args.frame_type,
        target_width=args.width,
        target_height=args.height,
        target_prompt_count=args.prompt_count
    )

if __name__ == "__main__":
    main()

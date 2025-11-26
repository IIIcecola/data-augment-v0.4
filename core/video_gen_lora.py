from gradio_client import Client, handle_file
import os
import argparse
import cv2
import random
from tqdm import tqdm
import traceback

# 支持的视频格式
SUPPORTED_VIDEO_FORMATS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")

def extract_first_frame(video_path, output_dir):
    """提取视频首帧并保存"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return None
        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"错误：无法读取视频首帧 {video_path}")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_path = os.path.join(output_dir, f"{base_name}_first_frame.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path
    except Exception as e:
        print(f"提取首帧失败 {video_path}：{str(e)}")
        return None

def generate_prompts(target_count):
    """生成仅改变性别、穿着和年龄的prompt列表"""
    # 基础动作描述（固定部分）
    base_action = "STANDHIGH, 一名工人在当前位置抓着货架边缘，双手用力拉拽，双脚交替，踩着货架侧面向上攀爬，最终成功攀爬并站稳在货架上。"
    
    # 可变属性
    genders = ["女性", "男性"]
    clothes = [
        "白色衬衫黑色长裤", "蓝色工装服", "红色安全背心+深色长裤",
        "黄色反光马甲+灰色工装裤", "灰色长袖工作服", "黑色耐磨夹克+卡其裤"
    ]
    ages = ["20到30岁的年轻人", "30到40岁的中年人", "40到50岁的中年人"]
    
    # 生成组合并去重
    prompts = []
    combinations = set()
    while len(prompts) < target_count:
        gender = random.choice(genders)
        cloth = random.choice(clothes)
        age = random.choice(ages)
        
        # 避免重复组合
        combo_key = f"{gender}_{cloth}_{age}"
        if combo_key in combinations:
            continue
        combinations.add(combo_key)
        
        # 构建完整prompt
        prompt = f"{base_action} 修改工人为{gender}，穿着{cloth}，{age}。"
        prompts.append(prompt)
    
    return prompts

def generate_video(client, img_path, video_prompt, output_dir, width, height):
    """调用API生成视频"""
    try:
        # 加载lora
        client.predict(
          local_high_LoRA_paths="path/to/LoRA",
          local_low_LoRA_paths="",
          api_name="/update_local_LoRA_path"
        )
        
        result = client.predict(
            prompt=video_prompt,
            negative_prompt='',
            seed=-1,
            steps=4,
            input_image=handle_file(img_path),
            end_image=None,
            mode_selector="图生视频", 
            fps_slider=24,
            input_video=None,
            prompt_refiner=False,
            lora_selector=["上传本地LoRA"],
            height=height,
            width=width,
            frame_num=75,
            api_name="/generate_video"
        )

        # 解析API返回的视频路径
        video_temp_path = result.get("video")
        if not video_temp_path or not os.path.exists(video_temp_path):
            print(f"警告：API未返回有效视频路径 {img_path}")
            return False

        # 构建输出视频路径
        img_dir, img_name = os.path.split(img_path)
        img_base_name = os.path.splitext(img_name)[0]
        # 加入prompt哈希值避免重名
        prompt_hash = hash(video_prompt) % 10000
        output_video_name = f"{img_base_name}_prompt_{prompt_hash}.mp4"
        output_video_path = os.path.join(output_dir, output_video_name)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 保存视频
        with open(video_temp_path, "rb") as f_in, open(output_video_path, "wb") as f_out:
            f_out.write(f_in.read())

        print(f"成功生成视频：{output_video_path}")
        return True

    except Exception as e:
        print(f"\n错误：生成视频失败 {img_path}")
        print(f"错误详情：{str(e)}")
        traceback.print_exc()
        return False

def get_all_video_files(input_path):
    """获取所有支持的视频文件"""
    video_files = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith(SUPPORTED_VIDEO_FORMATS):
            video_files.append(input_path)
        else:
            print(f"警告：{input_path} 不是支持的视频格式，跳过")
    elif os.path.isdir(input_path):
        print(f"正在扫描目录 {input_path} 下的视频...")
        for dirpath, _, filenames in os.walk(input_path):
            for filename in filenames:
                if filename.lower().endswith(SUPPORTED_VIDEO_FORMATS):
                    video_files.append(os.path.join(dirpath, filename))
        print(f"共发现 {len(video_files)} 个视频文件")
    else:
        raise ValueError(f"错误：输入路径 {input_path} 不是文件或目录")
    return video_files

def main():
    parser = argparse.ArgumentParser(description='基于视频首帧生成多样化人物视频工具')
    parser.add_argument('input', help='输入视频源（单张视频路径或视频目录）')
    parser.add_argument('output', help='视频输出目录')
    parser.add_argument('--api-url', required=True, help='视频生成API地址')
    parser.add_argument('--width', type=int, default=480, help='生成视频宽度')
    parser.add_argument('--height', type=int, default=832, help='生成视频高度')
    parser.add_argument('--prompt-count', type=int, required=True, help='生成的prompt数量（决定视频多样性）')

    args = parser.parse_args()

    # 初始化API客户端
    print(f"连接API：{args.api_url}")
    try:
      client = Client("http://api-base")
    except Exception as e:
        print(f"错误：无法连接API {args.api_url}")
        traceback.print_exc()
        return

    # 获取所有视频文件
    try:
        video_files = get_all_video_files(args.input)
        if not video_files:
            print("错误：未找到任何支持的视频文件")
            return
    except Exception as e:
        print(f"错误：获取视频文件失败")
        traceback.print_exc()
        return

    # 创建首帧保存目录
    first_frames_dir = os.path.join(args.output, "extracted_first_frames")
    print(f"首帧将保存至：{first_frames_dir}")

    # 生成prompt列表
    print(f"生成 {args.prompt_count} 个多样化prompt...")
    prompt_list = generate_prompts(args.prompt_count)

    # 批量处理视频
    print(f"\n开始处理（共 {len(video_files)} 个视频，每个视频生成 {args.prompt_count} 个变体）...")
    for video_path in tqdm(video_files, desc="视频处理进度"):
        # 提取首帧
        first_frame_path = extract_first_frame(video_path, first_frames_dir)
        if not first_frame_path:
            print(f"跳过视频 {video_path}（首帧提取失败）")
            continue

        # 为每个prompt生成视频
        for prompt in prompt_list:
            generate_video(
                client=client,
                img_path=first_frame_path,
                video_prompt=prompt,
                output_dir=args.output,
                width=args.width,
                height=args.height
            )

    print("\n" + "="*50)
    print(f"批量处理完成！")
    print(f"总处理视频：{len(video_files)} 个")
    print(f"生成视频总数：{len(video_files) * args.prompt_count} 个")
    print(f"首帧保存目录：{first_frames_dir}")
    print(f"视频输出目录：{os.path.abspath(args.output)}")
    print("="*50)

if __name__ == "__main__":
    main()

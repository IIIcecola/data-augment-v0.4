from gradio_client import Client, handle_file
import os
import argparse
import random
from tqdm import tqdm
from PIL import Image

# 初始化API客户端（根据实际API地址调整）
client = Client("http://10.59.67.2:5012/")

def find_background_images(root_dir):
    """查找所有背景图片文件"""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    background_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                background_files.append(os.path.join(dirpath, filename))
    return background_files

def get_image_size(image_path):
    """获取图片尺寸"""
    with Image.open(image_path) as img:
        return img.size

def generate_fall_image(client, background_path, prompt, output_path, index):
    """生成单张倒地人员图像"""
    try:
        # 获取背景图尺寸并保持一致
        width, height = get_image_size(background_path)
        
        # 调用API生成图像
        result = client.predict(
            image1=handle_file(background_path),
            image2=None,
            image3=None,
            prompt=prompt,
            seed=random.randint(0, 10000),  # 随机种子增加多样性
            randomize_seed=True,
            true_guidance_scale=1.2,
            num_inference_steps=5,
            rewrite_prompt=False,
            height=height,
            width=width,
            api_name="/infer"
        )
        src_path = result[0]

        # 保存结果
        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(background_path))[0]
        dst_path = os.path.join(output_path, f"{base_name}_fall_{index}.jpg")
        with open(src_path, "rb") as f_in, open(dst_path, "wb") as f_out:
            f_out.write(f_in.read())
        return True
    except Exception as e:
        print(f"处理失败 {background_path}: {str(e)}")
        return False

def generate_prompts():
    """生成多样化的倒地人员描述Prompt"""
    # 位置多样性
    positions = [
        "图像左侧区域", "图像右侧区域", "图像中央区域",
        "图像左上角", "图像右上角", "图像左下角", "图像右下角",
        "图像前景偏左", "图像前景偏右", "图像中景左侧",
        "图像中景右侧", "图像背景左侧", "图像背景右侧"
    ]
    
    # 朝向多样性
    orientations = [
        "面部朝上平躺", "面部朝下俯卧", "左侧身侧卧",
        "右侧身侧卧", "蜷缩身体俯卧", "半坐半躺姿态",
        "膝盖弯曲仰卧", "四肢伸展仰卧", "单膝跪地前倾倒地"
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
        "体型匀称的", "体型单薄的"
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

    # 强化比例约束：明确监控远距离视角的人物大小（核心优化）
    scale_constraint = "人物尺寸缩小，符合监控摄像头5-10米远距离拍摄比例，占图像总面积的5%-15%，" \
                       "避免近景特写效果，保持监控场景的远距离视角真实感"
    # 组合生成prompt
    prompts = []
    for pos in positions:
        for orient in orientations:
            for cloth in clothes:
                for body in body_types:
                    for age in ages:
                        for light in light_conditions:
                            for effect in monitor_effects:
                                for gender in genders:
                                    
                                  prompt = (f"在{pos}添加一名{age}，{gender}, {body}人，穿着{cloth}，呈{orient}状态。"
                                           f"{light}，{effect}，{scale_constraint}，符合工业监控场景视角，人物比例与监控场景匹配，"
                                           "自然融入背景，无明显合成痕迹，画面真实感强。")
                                  prompts.append(prompt)
    return prompts

def process_backgrounds(background_dir, output_dir, num_per_background=None):
    """处理背景图生成倒地人员图像"""
    # 获取所有背景图
    background_files = find_background_images(background_dir)
    if not background_files:
        print("错误：未找到任何背景图片")
        return
    
    # 生成所有可能的prompt组合
    all_prompts = generate_prompts()
    print(f"已生成 {len(all_prompts)} 种不同的Prompt组合")
    
    # 如果指定了每张背景图生成的数量，随机选择对应数量的prompt
    total_generated = 0
    target_count = 5000  # 目标生成总数
    
    for bg_path in tqdm(background_files, desc="处理背景图"):
        # 计算还需要生成的数量
        remaining = target_count - total_generated
        if remaining <= 0:
            break
            
        # 确定当前背景图需要生成的数量
        if num_per_background:
            current_num = min(num_per_background, remaining)
        else:
            # 平均分配剩余数量
            current_num = max(1, remaining // (len(background_files) - background_files.index(bg_path)))
            
        # 随机选择prompt
        selected_prompts = random.sample(all_prompts, current_num)
        
        # 生成图像
        for i, prompt in enumerate(selected_prompts):
            success = generate_fall_image(
                client, 
                bg_path, 
                prompt, 
                output_dir, 
                f"{background_files.index(bg_path)}_{i}"
            )
            if success:
                total_generated += 1
                
        print(f"背景图 {os.path.basename(bg_path)} 已生成 {current_num} 张图像，累计生成 {total_generated}/{target_count}")
    
    print(f"生成完成，共生成 {total_generated} 张倒地人员图像")

def main():
    parser = argparse.ArgumentParser(description='基于监控背景图生成多样化人员倒地图像工具')
    parser.add_argument('background_dir', help='监控背景图目录')
    parser.add_argument('output_dir', help='生成图像输出目录')
    parser.add_argument('--num-per-bg', type=int, help='每张背景图生成的图像数量（不指定则自动分配以达到目标数量）')
    
    args = parser.parse_args()
    
    process_backgrounds(
        args.background_dir,
        args.output_dir,
        args.num_per_bg
    )

if __name__ == "__main__":
    main()

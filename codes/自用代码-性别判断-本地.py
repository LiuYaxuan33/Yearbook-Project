import os
import time
import json
from deepface import DeepFace

def analyze_genders_deepface(image_folder):
    # 获取并排序图片文件
    valid_extensions = ('png', 'jpg', 'jpeg')
    image_files = sorted([f for f in os.listdir(image_folder) 
                        if f.lower().endswith(valid_extensions)])
    
    total_images = len(image_files)
    if total_images == 0:
        print("⚠️ 目标文件夹中没有发现图片文件！")
        return

    print(f"🔍 在目标文件夹中检测到 {total_images} 张图片")
    print("=" * 50)

    # 初始化数据存储
    processed_count = 0
    success_count = 0
    failed_files = []
    results_list = []
    start_time = time.time()

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, filename)
        current_status = f"[{idx}/{total_images}]"
        
        # 进度条显示
        progress = int(idx/total_images*20)
        progress_bar = f"{current_status} [{'='*progress}{' '*(20-progress)}]"

        print(f"\n🚀 正在处理 {progress_bar} {filename}")
        processed_count += 1

        try:
            single_start = time.time()
            
            # 人脸分析
            results = DeepFace.analyze(
                img_path=image_path,
                actions=['gender'],
                detector_backend='retinaface',
                enforce_detection=True,
                silent=True
            )
            
            # 收集结果
            cost_time = time.time() - single_start
            genders = [result["gender"] for result in results]
            
            print(f"✅ 完成！耗时 {cost_time:.2f}s → 检测到 {len(genders)} 张人脸")
            print(f"   👥 性别分布: {genders}")
            
            # 记录成功结果
            results_list.append({
                "filename": filename,
                "status": "success",
                "processing_time": cost_time,
                "face_count": len(genders),
                "genders": genders
            })
            success_count += 1

        except Exception as e:
            # 记录失败结果
            error_type = str(e).split("\n")[0]
            print(f"❌ 处理失败！错误类型: {error_type}")
            
            results_list.append({
                "filename": filename,
                "status": "failed",
                "error": error_type
            })
            failed_files.append(filename)

    # 保存JSON结果
    output_path = os.path.join(image_folder, "1922_gender_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)

    # 最终统计报告
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("📊 处理完成！最终统计：")
    print(f"   ✅ 成功处理: {success_count}/{total_images}")
    print(f"   ❌ 失败文件: {len(failed_files)}")
    if failed_files:
        print(f"       失败列表: {', '.join(failed_files)}")
    print(f"⏱️ 总耗时: {total_time:.2f} 秒 (平均 {total_time/processed_count:.2f}s/张)")
    print(f"💾 分析结果已保存至: {output_path}")

if __name__ == "__main__":
    target_folder = "/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1923"
    analyze_genders_deepface(target_folder)
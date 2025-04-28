import cv2
import os
import glob

input_dir = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1906'
output_dir = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1906-double'

os.makedirs(output_dir, exist_ok=True)
image_files = glob.glob(os.path.join(input_dir, '*.*'))

# 存储两个裁剪区域的坐标点
regions = [[], []]  # 每个区域包含两个点
current_region = 0  # 当前正在选择的区域索引（0或1）

def mouse_callback(event, x, y, flags, param):
    global regions, current_region
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(regions[current_region]) < 2:
            regions[current_region].append((x, y))
            print(f"区域 {current_region+1} 选择点 {len(regions[current_region])}: ({x}, {y})")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    regions = [[], []]  # 重置所有区域
    current_region = 0
    cv2.namedWindow('双区域裁剪')
    cv2.setMouseCallback('双区域裁剪', mouse_callback)
    
    while True:
        display_img = img.copy()
        
        # 绘制所有区域的标记
        for idx, region in enumerate(regions):
            color = (0, 0, 255) if idx == 0 else (0, 255, 0)  # 第一个区域红色，第二个区域绿色
            # 绘制已选择的点
            for point in region:
                cv2.circle(display_img, point, 5, color, -1)
            # 绘制矩形
            if len(region) == 2:
                x1, y1 = region[0]
                x2, y2 = region[1]
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
        
        # 显示提示文字
        status_text = f"选择区域 {current_region+1} (红色: 区域1, 绿色: 区域2)"
        cv2.putText(display_img, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('双区域裁剪', display_img)
        key = cv2.waitKey(1)
        
        # 处理键盘输入
        if key == 13:  # 回车确认当前区域
            if len(regions[current_region]) == 2:
                if current_region < 1:
                    current_region += 1
                else:
                    break  # 两个区域都选择完成
        elif key == ord('r'):  # 重置当前区域
            regions[current_region] = []
        elif key == 27:  # ESC跳过当前图片
            break
    
    if key == 27:
        cv2.destroyAllWindows()
        continue
    
    # 进行双区域裁剪
    valid_regions = []
    for region in regions:
        if len(region) == 2:
            # 规范坐标
            x_coords = sorted([region[0][0], region[1][0]])
            y_coords = sorted([region[0][1], region[1][1]])
            
            # 边界检查
            x1 = max(0, x_coords[0])
            y1 = max(0, y_coords[0])
            x2 = min(img.shape[1], x_coords[1])
            y2 = min(img.shape[0], y_coords[1])
            
            valid_regions.append((x1, y1, x2, y2))
    
    # 保存结果
    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    for i, (x1, y1, x2, y2) in enumerate(valid_regions):
        cropped = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(output_dir, f'{name}_crop{i+1}{ext}'), cropped)
    
    cv2.destroyAllWindows()

print("处理完成！")
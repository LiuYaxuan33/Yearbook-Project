import cv2
import os
import glob
input_dir = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1916'  # 输入图片目录
output_dir = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1916-cropped'  # 输出目录

os.makedirs(output_dir, exist_ok=True)
image_files = glob.glob(os.path.join(input_dir, '*.*'))  # 支持所有图片格式

points = []  # 存储选择的两个点

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"选择点 {len(points)}: ({x}, {y})")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    points = []  # 为每张图片重置点
    cv2.namedWindow('选择裁剪区域')
    cv2.setMouseCallback('选择裁剪区域', mouse_callback)
    
    while True:
        # 显示带标记的图片
        display_img = img.copy()
        
        # 绘制已选择的点
        if len(points) >= 1:
            cv2.circle(display_img, points[0], 5, (0, 0, 255), -1)
        if len(points) >= 2:
            cv2.circle(display_img, points[1], 5, (0, 255, 0), -1)
            # 绘制矩形区域
            x1, y1 = points[0]
            x2, y2 = points[1]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        cv2.imshow('选择裁剪区域', display_img)
        key = cv2.waitKey(1)
        
        # ESC键跳过当前图片
        if key == 27:
            break
        # 按回车确认选择
        elif key == 13 and len(points) == 2:
            break
        # 按r键重置点
        elif key == ord('r'):
            points = []
    
    if key == 27:  # 如果按了ESC，跳过当前图片
        cv2.destroyAllWindows()
        continue
    
    if len(points) == 2:
        # 确定坐标范围
        x_coords = sorted([points[0][0], points[1][0]])
        y_coords = sorted([points[0][1], points[1][1]])
        
        # 确保坐标在图像范围内
        x1 = max(0, x_coords[0])
        y1 = max(0, y_coords[0])
        x2 = min(img.shape[1], x_coords[1])
        y2 = min(img.shape[0], y_coords[1])
        
        # 裁剪图像
        cropped = img[y1:y2, x1:x2]
        
        # 保存结果
        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)
        cv2.imwrite(os.path.join(output_dir, f'{name}_cropped{ext}'), cropped)
    
    cv2.destroyAllWindows()

print("处理完成！")
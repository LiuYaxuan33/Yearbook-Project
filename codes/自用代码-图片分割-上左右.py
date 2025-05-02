import cv2
import os
import glob

input_dir = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1928/1'
output_dir = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1928-splited/1'

os.makedirs(output_dir, exist_ok=True)
image_files = glob.glob(os.path.join(input_dir, '*.*'))

click_x = None
click_y = None

def mouse_callback(event, x, y, flags, param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x = x
        click_y = y
        print(f"分割位置：x={x}, y={y}")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    
    cv2.namedWindow('选择分割位置')
    cv2.setMouseCallback('选择分割位置', mouse_callback)
    
    click_x = None
    click_y = None
    while True:
        cv2.imshow('选择分割位置', img)
        key = cv2.waitKey(1)
        if (click_x is not None and click_y is not None) or key == 27:
            break
    
    if key == 27:
        break
    
    if click_x is not None and click_y is not None:
        # 确保坐标在有效范围内
        x = max(0, min(click_x, w-1))
        y = max(0, min(click_y, h-1))
        
        # 分割为三个区域
        top = img[0:y, :]               # 上半部分
        left_bottom = img[y:h, 0:x]     # 左下部分
        right_bottom = img[y:h, x:w]    # 右下部分
        
        # 保存结果
        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)
        
        if top.size > 0:
            cv2.imwrite(f'{output_dir}/{name}_top{ext}', top)
        if left_bottom.size > 0:
            cv2.imwrite(f'{output_dir}/{name}_left{ext}', left_bottom)
        if right_bottom.size > 0:
            cv2.imwrite(f'{output_dir}/{name}_right{ext}', right_bottom)
    
    cv2.destroyAllWindows()

print("处理完成！")
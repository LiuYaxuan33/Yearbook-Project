import cv2
import os
import glob

input_dir = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1932'  # 输入图片目录
output_dir = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1932-splited'  # 输出目录

os.makedirs(output_dir, exist_ok=True)
image_files = glob.glob(os.path.join(input_dir, '*.*'))  # 支持所有图片格式

click_x = None

def mouse_callback(event, x, y, flags, param):
    global click_x
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x = x
        print(f"分割位置：x={x}")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    
    cv2.namedWindow('选择分割位置')
    cv2.setMouseCallback('选择分割位置', mouse_callback)
    
    click_x = None
    while True:
        cv2.imshow('选择分割位置', img)
        key = cv2.waitKey(1)
        if click_x is not None or key == 27:
            break
    if key == 27:  # 按ESC退出
        break
    
    if click_x is not None and 0 < click_x < w:
        left = img[:, :click_x]
        right = img[:, click_x:]
        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)
        cv2.imwrite(f'{output_dir}/{name}_left{ext}', left)
        cv2.imwrite(f'{output_dir}/{name}_right{ext}', right)
    
    cv2.destroyAllWindows()

print("处理完成！")
import os
import numpy as np
import imutils
import cv2
from PIL import Image
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator
from datetime import datetime

def find_plate(edged):
    keypts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypts)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    locations = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 5, True)
        if len(approx) == 4:
            locations.append(approx)
    return locations

def hide_plate(image_PIL):
    img = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(img_gray, 170, 200)
    locations = find_plate(edged)
    
    if len(locations) == 0:
        return False, None
    
    plate = locations[0]
    rect = cv2.minAreaRect(plate)
    box = np.intp(cv2.boxPoints(rect))
    final_image = img.copy()
    final_image = cv2.drawContours(final_image, [box], -1, (255, 255, 255), -1)
    final_image = cv2.drawContours(final_image, [box], 0, (207, 149, 1), 2)
    
    if not os.path.exists("resources"):
        os.makedirs("resources")
    
    saved = cv2.imwrite("resources/car_wo_plate.png", final_image)
    if not saved:
        print("Error: Failed to save car_wo_plate.png")
        return False, None
    
    return True, "resources/car_wo_plate.png"

def remove_car_bg(image_path):
    seg_net = TracerUniversalB7(device='cpu', batch_size=1)
    fba = FBAMatting(device='cpu', input_tensor_size=2048, batch_size=1)
    trimap = TrimapGenerator()
    preprocessing = PreprocessingStub()
    postprocessing = MattingMethod(matting_module=fba, trimap_generator=trimap, device='cpu')
    interface = Interface(pre_pipe=preprocessing, post_pipe=postprocessing, seg_pipe=seg_net)
    
    image = Image.open(image_path)
    car_wo_bg = interface([image])[0]
    car_wo_bg.save("resources/car_wo_bg.png")

def place_car_on_background(background_image_path, output_path):
    background = Image.open(background_image_path).convert("RGBA")
    bg_width, bg_height = background.size
    car = Image.open("resources/car_wo_bg.png").convert("RGBA")
    car_width, car_height = car.size
    
    scale_factor = min((bg_width * 0.95) / car_width, (bg_height * 0.95) / car_height)
    new_car_width = int(car_width * scale_factor)
    new_car_height = int(car_height * scale_factor)
    car = car.resize((new_car_width, new_car_height), Image.LANCZOS)
    
    x = (bg_width - new_car_width) // 2
    y = (bg_height - new_car_height) // 2
    
    background.paste(car, (x, y), car)
    background.save(output_path, format="PNG")
    print(f"✅ Image saved at {output_path}")

def create_generated_car_name(name, index=None):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    if index is not None:
        return f"{name}_{index}_{date_str}_{time_str}.png"
    return f"{name}_{date_str}_{time_str}.png"

def process_car_image(car_image_path, background_image_path, index=None):
    image_PIL = Image.open(car_image_path)
    hide_success, car_wo_plate_path = hide_plate(image_PIL)

    if hide_success:
        print(f"✅ Car plate hidden successfully for image: {car_image_path}")
        image_for_bg_removal = car_wo_plate_path
    else:
        print(f"⚠️ No plate detected or failed to save processed image. Proceeding with the original image: {car_image_path}")
        image_for_bg_removal = car_image_path

    remove_car_bg(image_for_bg_removal)
    
    output_image_name = create_generated_car_name("generate_car", index)
    place_car_on_background(background_image_path, output_image_name)

def process_multiple_cars(car_images, background_image_path):
    for index, car_image in enumerate(car_images):
        print(f"\n🚀 Processing image {index + 1}/{len(car_images)}: {car_image}")
        process_car_image(car_image, background_image_path, index)

if __name__ == "__main__":
    print("\nChoose an option:\n1. Process a single car image\n2. Process multiple car images")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        car_image = input("Enter the car image path: ").strip()
        background_image = input("Enter the background image path: ").strip()
        process_car_image(car_image, background_image)

    elif choice == "2":
        num_images = int(input("Enter the number of car images: ").strip())
        car_images = [input(f"Enter path for car image {i + 1}: ").strip() for i in range(num_images)]
        background_image = input("Enter the background image path: ").strip()
        process_multiple_cars(car_images, background_image)

    else:
        print("❌ Invalid choice! Please restart the program and enter 1 or 2.")

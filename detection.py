import os
import cv2
from ultralytics import YOLO

# Paths
model_path = "soap_best.pt"
input_folder = "images"
output_folder = "output_images"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Load model
model = YOLO(model_path)
class_names = model.names

# Loop through images
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, file_name)
        img = cv2.imread(image_path)

        # Run detection
        results = model(img)

        # Draw detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{class_names[cls]} {conf:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

        # Save output image
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, img)
        print(f"Processed: {file_name}")

print("✅ All images processed successfully!")

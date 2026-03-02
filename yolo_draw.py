import cv2

# Paths
image_path = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\images\training\pexels-aleks-magnusson-2907762.jpg"
label_path = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\labels\training\pexels-aleks-magnusson-2907762.txt"

# Load image
image = cv2.imread(image_path)
# h, w, _ = image.shape
h, w = 320, 320
image = cv2.resize(image, (h, w))


# Read label file
with open(label_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    bbox_width = float(parts[3])
    bbox_height = float(parts[4])

    # Convert normalized coordinates to pixel values
    x_center *= w
    y_center *= h
    bbox_width *= w
    bbox_height *= h

    # Convert to top-left corner
    x1 = int(x_center - bbox_width / 2)
    y1 = int(y_center - bbox_height / 2)
    x2 = int(x_center + bbox_width / 2)
    y2 = int(y_center + bbox_height / 2)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Optional: draw class id
    cv2.putText(image, str(class_id), (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show image
cv2.imshow("Image with Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



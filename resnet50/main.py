from PIL import Image, ImageDraw
from huggingface_hub import InferenceClient

# 1. 이미지 로드
image_path = "cats.jpg"
image = Image.open(image_path).convert("RGB")

# 2. InferenceClient로 객체 감지 실행
client = InferenceClient(api_key="api키를 입력하세요")
results = client.object_detection(image_path, model="facebook/detr-resnet-50")

print(results)

# 3. 이미지에 박스 그리기
draw = ImageDraw.Draw(image)
for result in results:
    box = result["box"]
    label = result["label"]
    score = result["score"]
    
    draw.rectangle(
        [(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])],
        outline="red",
        width=4
    )
    draw.text(
        (box["xmin"], box["ymin"] - 10),
        f"{label} ({score:.2f})",
        fill="red"
    )

# 4. 이미지 보여주기
image.show()

# 5. 혹은 파일로 저장
image.save("output_with_boxes.jpg")

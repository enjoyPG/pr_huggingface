from PIL import Image, ImageDraw
from huggingface_hub import InferenceClient
import random

# 형광색 생성 함수 (R/G/B 중 하나를 255로 고정)
def neon_color():
    values = [255, random.randint(100, 255), random.randint(100, 255)]
    random.shuffle(values)
    return tuple(values)

# 이미지 로드
image_path = "savanna.jpg"
image = Image.open(image_path).convert("RGB")

# Inference API
client = InferenceClient(api_key="허깅페이스 API넣으세요")  # <- 토큰 입력
results = client.object_detection(image_path, model="facebook/detr-resnet-50")

# 그리기 객체 생성
draw = ImageDraw.Draw(image)

for idx, obj in enumerate(results, start=1):
    box = obj["box"]
    label = obj["label"]
    score = obj["score"]

    color = neon_color()

    # 사각형 그리기
    draw.rectangle(
        [(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])],
        outline=color,
        width=3
    )

    # 텍스트(라벨만) 그리기
    draw.text(
        (box["xmin"] + 5, box["ymin"] - 20),
        f"{label}",
        fill=color
    )

    # cmd에 출력
    print(f"{idx}. {label} detected with {score*100:.1f}% confidence")

# 이미지 결과 확인
image.show()
image.save("detected_neon_output.jpg")

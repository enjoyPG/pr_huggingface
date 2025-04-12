import torch
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection
import torchvision.transforms as T
import random

# ✅ 형광색 생성 함수
def neon_color():
    values = [255, random.randint(100, 255), random.randint(100, 255)]
    random.shuffle(values)
    return tuple(values)

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ 모델 및 전처리기 로드 (처음 실행 시 자동 다운로드)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
model.eval()

# ✅ 이미지 불러오기
image_path = "savanna.jpg"  # <- 원하는 이미지로 바꾸세요
image = Image.open(image_path).convert("RGB")

# ✅ 전처리 및 모델 입력 생성
inputs = processor(images=image, return_tensors="pt").to(device)

# ✅ 추론
with torch.no_grad():
    outputs = model(**inputs)

# ✅ 결과 후처리
target_sizes = torch.tensor([image.size[::-1]], device=device)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# ✅ 이미지에 그리기
draw = ImageDraw.Draw(image)

for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"]), start=1):
    box = [round(i.item()) for i in box]
    label_name = model.config.id2label[label.item()]
    confidence = score.item()

    # 형광색
    color = neon_color()

    # 사각형 그리기
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)

    # 텍스트(라벨만) 그리기
    draw.text((box[0] + 5, box[1] - 20), label_name, fill=color)

    # 콘솔 출력
    print(f"{idx}. {label_name} detected with {confidence*100:.1f}% confidence")

# ✅ 결과 보기 & 저장
image.show()
image.save("detected_local_output.jpg")

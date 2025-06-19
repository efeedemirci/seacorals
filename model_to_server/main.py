from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

# Modeli yükle
model = YOLO("best.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # YOLOv8 tahmin
        results = model(image)

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": xyxy,
                    "label": model.names[cls_id]
                })

        return JSONResponse(content={"detections": detections})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

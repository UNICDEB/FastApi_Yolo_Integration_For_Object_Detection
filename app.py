# import os
# import io
# import time
# from fastapi import FastAPI, UploadFile, Form, Request
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from ultralytics import YOLO
# import cv2
# import numpy as np

# app = FastAPI()

# # Load YOLO model once
# model = YOLO("weights/best.pt")

# # Serve static directory for CSS and detected images
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Template directory
# templates = Jinja2Templates(directory="templates")


# def detection(frame, model, threshold):
#     results = model(frame)
#     boxes = results[0].boxes.xyxy.tolist()
#     confidences = results[0].boxes.conf.tolist()
#     classes = results[0].boxes.cls.tolist()
#     names = results[0].names

#     detected_boxes = []
#     detected_centers = []
#     detected_confidences = []

#     annotated_frame = frame.copy()

#     for box, confidence in zip(boxes, confidences):
#         if confidence >= threshold:
#             start_point = (round(box[0]), round(box[1]))
#             end_point = (round(box[2]), round(box[3]))
#             # Draw bounding box
#             cv2.rectangle(annotated_frame, start_point, end_point, (0, 0, 255), 3)
#             # Compute center
#             center_x = round((box[0] + box[2]) / 2)
#             center_y = round((box[1] + box[3]) / 2)
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
#             detected_boxes.append(box)
#             detected_centers.append([center_x, center_y])
#             detected_confidences.append(confidence)

#     return annotated_frame, detected_boxes, detected_centers, detected_confidences


# @app.get("/")
# async def main(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/detect/")
# async def detect_image(
#     file: UploadFile,
#     threshold: float = Form(0.5)
# ):
#     if not (0.1 <= threshold <= 1.0):
#         return JSONResponse(
#             status_code=400,
#             content={"error": "Threshold must be between 0.1 and 1.0"}
#         )

#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     start_time = time.time()
#     annotated_image, boxes, centers, confidences = detection(img, model, threshold)
#     end_time = time.time()

#     # If no detections, return message
#     if not boxes:
#         return JSONResponse(
#             content = {
#                 "message": "No Object Detected",
#                 "Processing_Time":"round(end_time-start_time, 3)"
#             }
#         )
#     # Save annotated image
#     save_path = "static/detected.jpg"
#     cv2.imwrite(save_path, annotated_image)

#     response = {
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confidences,
#         "processing_time_sec": round(end_time - start_time, 3),
#         "detected_image_url": "/static/detected.jpg"
#     }
#     return JSONResponse(content=response)

import os
import io
import time
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load YOLO model once
model = YOLO("weights/best.pt")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def detection(frame, model, threshold):
    results = model(frame)
    boxes = results[0].boxes.xyxy.tolist()
    confidences = results[0].boxes.conf.tolist()
    detected_boxes = []
    detected_centers = []
    detected_confidences = []
    annotated_frame = frame.copy()

    for box, confidence in zip(boxes, confidences):
        if confidence >= threshold:
            start_point = (round(box[0]), round(box[1]))
            end_point = (round(box[2]), round(box[3]))
            cv2.rectangle(annotated_frame, start_point, end_point, (0, 0, 255), 3)
            center_x = round((box[0] + box[2]) / 2)
            center_y = round((box[1] + box[3]) / 2)
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            detected_boxes.append(box)
            detected_centers.append([center_x, center_y])
            detected_confidences.append(confidence)

    return annotated_frame, detected_boxes, detected_centers, detected_confidences


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": None,
        "num_detections": None,
        "bounding_boxes": [],
        "centers": [],
        "confidences": [],
        "processing_time_sec": None,
        "image_url": None
    })


@app.post("/detect/")
async def detect_html(request: Request, file: UploadFile, threshold: float = Form(0.5)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    start_time = time.time()
    annotated_image, boxes, centers, confidences = detection(img, model, threshold)
    end_time = time.time()

    if boxes:
        cv2.imwrite("static/detected.jpg", annotated_image)
        message = "Objects detected"
        image_url = "/static/detected.jpg"
    else:
        cv2.imwrite("static/detected.jpg", img)
        message = "No object detected"
        image_url = "/static/detected.jpg"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": message,
        "num_detections": len(boxes),
        "bounding_boxes": boxes,
        "centers": centers,
        "confidences": confidences,
        "processing_time_sec": round(end_time - start_time, 3),
        "image_url": image_url
    })


@app.post("/detect-json/")
async def detect_json(file: UploadFile, threshold: float = Form(0.5)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    start_time = time.time()
    annotated_image, boxes, centers, confidences = detection(img, model, threshold)
    end_time = time.time()

    if boxes:
        cv2.imwrite("static/detected.jpg", annotated_image)
        message = "Objects detected"
        image_url = "/static/detected.jpg"
    else:
        cv2.imwrite("static/detected.jpg", img)
        message = "No object detected"
        image_url = "/static/detected.jpg"

    return JSONResponse({
        "message": message,
        "num_detections": len(boxes),
        "bounding_boxes": boxes,
        "centers": centers,
        "confidences": confidences,
        "processing_time_sec": round(end_time - start_time, 3),
        "detected_image_url": image_url
    })

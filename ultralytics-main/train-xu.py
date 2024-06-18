from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/v8n-zuoye-train3/weights/last.pt')  # load a partially trained model
if __name__ == '__main__':
# Resume training
    results = model.train(resume=True)

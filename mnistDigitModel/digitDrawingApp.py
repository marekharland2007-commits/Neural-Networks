import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from digitModel import CNNNet

CANVAS_SIZE = 280
MODEL_INPUT = 28
BRUSH_RADIUS = 10

device = torch.device("cpu")

model = CNNNet().to(device)
state_dict = torch.load("digit_model_pytorch.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Drawing App")

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            cursor="cross"
        )
        self.canvas.pack(padx=12, pady=12)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        controls = tk.Frame(root)
        controls.pack(pady=(0, 12))

        tk.Button(controls, text="Predict", command=self.predict_digit, width=12).grid(row=0, column=0, padx=6)
        tk.Button(controls, text="Clear", command=self.clear_canvas, width=12).grid(row=0, column=1, padx=6)
        tk.Button(controls, text="Save PNG", command=self.save_png, width=12).grid(row=0, column=2, padx=6)

        self.result_var = tk.StringVar(value="Draw a digit and click Predict")
        tk.Label(root, textvariable=self.result_var, font=("Arial", 14)).pack(pady=(0, 10))

        self.top3_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.top3_var, font=("Courier New", 11), justify="left").pack(pady=(0, 10))

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y
        self._draw_point(event.x, event.y)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=BRUSH_RADIUS * 2,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True
            )
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=BRUSH_RADIUS * 2)
            self._draw_point(x, y)
        self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.last_x, self.last_y = None, None

    def _draw_point(self, x, y):
        r = BRUSH_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_var.set("Draw a digit and click Predict")
        self.top3_var.set("")

    def preprocess(self):
        img = self.image.copy()
        bbox = ImageOps.invert(img).getbbox()
        if bbox is None:
            return None, None

        img = img.crop(bbox)
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)

        canvas = Image.new("L", (MODEL_INPUT, MODEL_INPUT), 255)
        x = (MODEL_INPUT - img.width) // 2
        y = (MODEL_INPUT - img.height) // 2
        canvas.paste(img, (x, y))

        arr = np.array(canvas).astype(np.float32)
        arr = 255.0 - arr
        arr /= 255.0

        return canvas, arr

    def predict_digit(self):
        processed_img, arr = self.preprocess()
        if arr is None:
            self.result_var.set("Please draw a digit first")
            self.top3_var.set("")
            return

        x = torch.from_numpy(arr).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        top3_idx = np.argsort(probs)[-3:][::-1]
        lines = [f"{i}: {probs[i]*100:5.2f}%" for i in top3_idx]

        self.result_var.set(f"Prediction: {pred}   Confidence: {conf*100:.2f}%")
        self.top3_var.set("Top 3 guesses\n" + "\n".join(lines))

    def save_png(self):
        self.image.save("digit_canvas.png")
        self.result_var.set("Saved current drawing as digit_canvas.png")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
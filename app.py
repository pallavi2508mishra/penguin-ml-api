from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.models.predict import predict


class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


app = FastAPI()


# ---------------------- 1) Health + Root ----------------------

@app.get("/", tags=["api"])
def root():
    return {"message": "Penguin ML API is running ✅"}


@app.get("/health", tags=["api"])
def healthcheck():
    return {"status": "ok"}


# ---------------------- 2) JSON Predict Endpoint ----------------------

@app.post("/predict", tags=["api"])
def predict_endpoint(features: PenguinFeatures):
    data = features.dict()
    species = predict(data)
    return {"predicted_species": species}


# ---------------------- 3) Beautiful Web UI ----------------------

def render_ui(prediction: str | None = None):
    return f"""
    <html>
    <head>
        <title>🐧 Penguin Classifier</title>
        <style>
            body {{
                margin: 0;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}
            .card {{
                background: white;
                padding: 35px;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                width: 380px;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 25px;
            }}
            input {{
                width: 100%;
                padding: 10px;
                margin: 8px 0 16px 0;
                border-radius: 6px;
                border: 1px solid #ccc;
                font-size: 14px;
            }}
            button {{
                width: 100%;
                padding: 12px;
                background-color: #2a5298;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 15px;
                cursor: pointer;
                transition: 0.2s;
            }}
            button:hover {{
                background-color: #1e3c72;
            }}
            .result {{
                margin-top: 20px;
                padding: 12px;
                background: #e6f4ea;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
                font-size: 16px;
            }}
            .footer {{
                margin-top: 15px;
                text-align: center;
                font-size: 12px;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🐧 Penguin Classifier</h1>
            <form method="post">
                <label>Bill length (mm)</label>
                <input name="bill_length_mm" type="number" step="0.1" required>

                <label>Bill depth (mm)</label>
                <input name="bill_depth_mm" type="number" step="0.1" required>

                <label>Flipper length (mm)</label>
                <input name="flipper_length_mm" type="number" step="1" required>

                <label>Body mass (g)</label>
                <input name="body_mass_g" type="number" step="1" required>

                <button type="submit">Predict Species</button>
            </form>

            {f'<div class="result">Prediction: {prediction}</div>' if prediction else ''}

            <div class="footer">
                Built with FastAPI • Docker • Render
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/ui", response_class=HTMLResponse, tags=["ui"])
def ui_form():
    return render_ui()


@app.post("/ui", response_class=HTMLResponse, tags=["ui"])
def ui_predict(
    bill_length_mm: float = Form(...),
    bill_depth_mm: float = Form(...),
    flipper_length_mm: float = Form(...),
    body_mass_g: float = Form(...),
):
    features = {
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
    }

    species = predict(features)

    return render_ui(prediction=species)

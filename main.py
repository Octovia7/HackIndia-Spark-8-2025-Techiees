from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image
import io
import itertools
import os
API_KEYS = [
    os.getenv("API_KEY_1"),
    os.getenv("API_KEY_2"),
    os.getenv("API_KEY_3"),
    os.getenv("API_KEY_4"),
    os.getenv("API_KEY_5"),
    os.getenv("API_KEY_6"),
    os.getenv("API_KEY_7"),
    os.getenv("API_KEY_8"),
    os.getenv("API_KEY_9"),
    os.getenv("API_KEY_10"),
]


api_key_cycle = itertools.cycle(API_KEYS)

def get_gemini_model():
    current_key = next(api_key_cycle)
    genai.configure(api_key=current_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model, current_key

app = FastAPI(
    title="AI Rasoi Recipe API 🍛",
    description="An AI API for recipes, fridge ingredient detection, diet plans & TTS with Gemini API key rotation.",
    version="5.0.0"
)

class RecipeRequest(BaseModel):
    ingredients: str
    preferences: str = "Less spicy, Punjabi style"

class DietRequest(BaseModel):
    height_cm: int
    weight_kg: int
    gender: str
    goal: str

class RecipeResponse(BaseModel):
    recipe: str

class DietResponse(BaseModel):
    diet_plan: str

def generate_recipe(ingredients, preferences):
    prompt = f"""
Suggest a simple Indian recipe with these ingredients: {ingredients}.
Preferences: {preferences}.
Include: 
- Dish name 
- Quick steps 
- Estimated calories 
- Spice level (1–5) 
- Region (e.g. Punjabi)
End with one casual desi kitchen tip.
"""
    for _ in range(len(API_KEYS)):
        try:
            model, _ = get_gemini_model()
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            continue
    raise HTTPException(status_code=429, detail="❌ All API keys exhausted for recipe generation.")

def detect_ingredients_from_image(image_bytes):
    prompt = """
List visible food ingredients from this fridge image. Ignore bottles, containers, and utensils.
Reply 'No visible ingredients detected.' if none.
"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((640, 480))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    resized_image_bytes = buf.getvalue()
    for _ in range(len(API_KEYS)):
        try:
            model, _ = get_gemini_model()
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": resized_image_bytes}]
            )
            return response.text.strip()
        except Exception:
            continue
    raise HTTPException(status_code=429, detail="❌ All API keys exhausted for image detection.")

def generate_diet_plan(height_cm, weight_kg, gender, goal):
    prompt = f"""
Create an Indian diet plan for:
Height: {height_cm} cm
Weight: {weight_kg} kg
Gender: {gender}
Goal: {goal}
Include:
- Daily calorie estimate
- 5 meal plan
- Macronutrient split
- 1 desi health tip.
Keep it concise and simple.
"""
    for _ in range(len(API_KEYS)):
        try:
            model, _ = get_gemini_model()
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            continue
    raise HTTPException(status_code=429, detail="All API keys exhausted for diet plan generation.")



@app.post("/generate_recipe", response_model=RecipeResponse)
async def get_recipe(request: RecipeRequest):
    recipe_text = generate_recipe(request.ingredients, request.preferences)
    return {"recipe": recipe_text}

@app.post("/fridge_recipe", response_model=RecipeResponse)
async def fridge_recipe(file: UploadFile = File(...), preferences: str = ""):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")
    image_bytes = await file.read()
    detected_ingredients = detect_ingredients_from_image(image_bytes)
    if "No visible ingredients detected" in detected_ingredients:
        return {"recipe": detected_ingredients}
    recipe_text = generate_recipe(detected_ingredients, preferences)
    return {"recipe": recipe_text}

@app.post("/diet_plan", response_model=DietResponse)
async def diet_plan(request: DietRequest):
    diet_text = generate_diet_plan(request.height_cm, request.weight_kg, request.gender, request.goal)
    return {"diet_plan": diet_text}


@app.get("/")
def root():
    return {"message": "Welcome to AI Rasoi Recipe API 🍛. Visit /docs for API documentation."}
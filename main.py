from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import pickle
import os

url = "https://xiachjnxrcqqvlkquazq.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhpYWNoam54cmNxcXZsa3F1YXpxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE4Mzk3OTgsImV4cCI6MjA1NzQxNTc5OH0.2e8TrRYic61lJh3hs1oPqQegQSmzjC_NBySAQzKSq6E"

supabase: Client = create_client(url, key)

app = FastAPI()

with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

class NewsRequest(BaseModel):
    news: str

class NewsUpdate(BaseModel):
    news: str = None

@app.post('/predict')
def classify_news(news: NewsRequest):
    input_data = [news.news]
    vectorized_data = vectorizer.transform(input_data)
    prediction = model.predict(vectorized_data).tolist()

    item_data = news.dict()
    item_data["label"] = prediction[0]

    try:
        result = supabase.table("news").insert(item_data).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting data: {e}")

    return {"label": prediction[0], "inserted": result.data}

@app.get("/items/{label}")
def get_news_by_label(label: str):
    response = supabase.table("news").select("*").eq("label", label).execute()
    if response.data:
        return response.data
    raise HTTPException(status_code=404, detail="Items not found")

@app.get("/items")
def get_all_news():
    response = supabase.table("news").select("*").execute()
    return response.data


@app.put("/items/{item_id}")
def update_news_item(item_id: int, update: NewsUpdate):
    update_data = update.dict(exclude_unset=True)

    existing = supabase.table("news").select("*").eq("id", item_id).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Item not found")

    if "news" in update_data:
        input_text = [update_data["news"]]
        vectorized = vectorizer.transform(input_text)
        prediction = model.predict(vectorized).tolist()[0]
        update_data["label"] = prediction 

    try:
        result = supabase.table("news").update(update_data).eq("id", item_id).execute()
        return {
            "message": "Item updated successfully",
            "updated": result.data,
            "predicted_label": update_data.get("label")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating item: {e}")

@app.delete("/items/{item_id}")
def delete_news_item(item_id: int):
    existing = supabase.table("news").select("*").eq("id", item_id).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Item not found")

    supabase.table("news").delete().eq("id", item_id).execute()
    return {"message": f"Item with ID {item_id} deleted successfully"}

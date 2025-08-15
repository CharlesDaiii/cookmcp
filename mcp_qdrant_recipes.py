from __future__ import annotations
from typing import Optional, Literal, List
from typing_extensions import TypedDict
import os, uuid, json, logging
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

from dotenv import load_dotenv

load_dotenv()

# ---- Logging (STDIO 模式请勿往 stdout 打印) ----
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mcp-qdrant")

# ---- Env ----
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# ---- Initialize clients ----
mcp = FastMCP("recipes-mcp-qdrant")

@dataclass
class Embedder:
    model_name: str
    model: SentenceTransformer

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

EMB = Embedder(EMBED_MODEL_NAME, SentenceTransformer(EMBED_MODEL_NAME))
log.info(f"Embedder: {EMB.model_name}, dim={EMB.dim}")

def qdrant() -> QdrantClient:
    assert QDRANT_URL and QDRANT_API_KEY, "Please set QDRANT_URL and QDRANT_API_KEY"
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def ensure_collection():
    client = qdrant()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMB.dim, distance=Distance.COSINE),
        )
        log.info(f"Created collection '{COLLECTION}' with dim={EMB.dim} (cosine)")
    return client

# ---- Types (English-first) ----
class Ingredient(TypedDict, total=False):
    item: str
    quantity: float
    unit: str
    notes: Optional[str]

class Step(TypedDict, total=False):
    order: int
    instruction: str
    timer_seconds: Optional[int]
    temperature_c: Optional[int]

class Nutrition(TypedDict, total=False):
    calories: int
    protein_g: float
    fat_g: float
    carbs_g: float
    sodium_mg: int

class Recipe(TypedDict, total=False):
    title: str
    cuisine: Optional[str]
    servings: int
    total_time_min: int
    ingredients: List[Ingredient]
    steps: List[Step]
    substitutions: List[str]
    tags: List[str]
    nutrition: Nutrition

# ---- Heuristic cooking patterns (English) ----
BASIC_TECHNIQUES = {
    "stir-fry": ["Heat pan until lightly smoking, add oil.", "Add main ingredients and stir-fry until just cooked; season.", "Reduce sauce slightly and serve."],
    "boil": ["Bring water/stock to a rolling boil.", "Add main ingredients; simmer until done; season.", "Reduce slightly if needed."],
    "roast": ["Preheat oven to 200°C.", "Roast 15–25 minutes depending on size.", "Rest for 3 minutes before slicing."],
    "braise": ["Sauté aromatics with a little oil.", "Add liquid to cover; simmer on low.", "Season and cook until tender."],
}

def choose_technique(ingredients: List[str], max_time: int) -> str:
    proteins = ["chicken","beef","pork","lamb","fish","shrimp","tofu"]
    has_protein = any(any(p in i.lower() for p in proteins) for i in ingredients)
    if max_time <= 20: return "stir-fry"
    if not has_protein and max_time >= 30: return "braise"
    return "boil"

def build_query_text(ingredients: List[str], cuisine: Optional[str], dietary: List[str], technique_hint: Optional[str]) -> str:
    parts = []
    if ingredients: parts.append("ingredients: " + ", ".join(ingredients))
    if cuisine: parts.append("cuisine: " + cuisine)
    if dietary: parts.append("dietary: " + ", ".join(dietary))
    if technique_hint: parts.append("technique: " + technique_hint)
    return " | ".join(parts) if parts else "quick home recipe"

def text_for_embedding(doc: dict) -> str:
    # Flexible: handles both simple docs and structured Recipe
    title = doc.get("title", "")
    ingredients = doc.get("ingredients", [])
    steps = doc.get("steps", [])
    if ingredients and isinstance(ingredients[0], dict):
        ingredients = [f"{i.get('item','')}" for i in ingredients]
    if steps and isinstance(steps[0], dict):
        steps = [s.get("instruction","") for s in steps]
    return "\n".join([title] + ingredients + steps)

def rough_nutrition(ings: List[Ingredient], servings: int) -> Nutrition:
    total_g = sum(i.get("quantity", 0) for i in ings if i.get("unit") in ("g","ml"))
    factor = (total_g / 100.0) if total_g else 1
    return {
        "calories": int(150 * factor / max(1, servings)),
        "protein_g": round(10 * factor / max(1, servings), 1),
        "fat_g": round(8 * factor / max(1, servings), 1),
        "carbs_g": round(12 * factor / max(1, servings), 1),
        "sodium_mg": int(600 / max(1, servings)),
    }

# ---------------- MCP Tools ----------------

@mcp.tool()
def ping() -> str:
    """Check Qdrant connectivity."""
    client = ensure_collection()
    info = client.get_collection(COLLECTION)

    # prefer points_count, fall back to exact count
    pts = info.points_count or client.count(COLLECTION, exact=True).count
    dim = info.config.params.vectors.size
    metric = info.config.params.vectors.distance
    indexed = info.indexed_vectors_count
    segs = info.segments_count
    return f"OK - '{COLLECTION}': points={pts}, indexed={indexed}, dim={dim}, metric={metric}, segments={segs}"

@mcp.tool()
def search_recipes(
    query: str,
    top_k: int = 5,
    cuisine: Optional[str] = None,
    tags: List[str] = [],
) -> List[dict]:
    """
    Semantic search over Qdrant Cloud (cosine). Optional filters by cuisine/tags.
    Returns: [{id, score, title, source_url, cuisine, tags}]
    """
    client = ensure_collection()
    qvec = EMB.encode([query])[0].tolist()

    qfilter: Optional[Filter] = None
    must = []
    if cuisine:
        must.append(FieldCondition(key="cuisine", match=MatchValue(value=cuisine)))
    for t in tags or []:
        must.append(FieldCondition(key="tags", match=MatchValue(value=t)))
    if must:
        qfilter = Filter(must=must)

    hits = client.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
        query_filter=qfilter
    )
    out = []
    for h in hits:
        p = h.payload or {}
        out.append({
            "id": str(h.id),
            "score": float(h.score),
            "title": p.get("title"),
            "source_url": p.get("source_url"),
            "cuisine": p.get("cuisine"),
            "tags": p.get("tags", []),
        })
    return out

@mcp.tool()
def generate_recipe(
    ingredients: List[str],
    servings: int = 2,
    cuisine: Optional[str] = None,
    dietary: List[Literal["vegan","vegetarian","gluten-free","halal","kosher","low-carb","high-protein"]] = [],
    avoid: List[str] = [],
    max_time_min: int = 30,
    equipment: List[str] = [],
    skill: Literal["beginner","intermediate","advanced"] = "beginner",
    use_rag: bool = True,
    language: Literal["en","zh"] = "en",
) -> Recipe:
    """
    Generate a structured recipe. If use_rag=True, first retrieve similar recipes
    from Qdrant and adapt their structure. Language defaults to English ("en").
    """
    usable = [i for i in ingredients if all(a.lower() not in i.lower() for a in avoid)] or ingredients[:]
    tech = choose_technique(usable, max_time_min)
    title = f"{(cuisine or 'Home-style').title()} {tech} " + (usable[0] if usable else "Vegetables")

    # Base ingredients (simple proportional sizing)
    base_qty = 120 if servings >= 3 else 90
    structured_ings: List[Ingredient] = []
    for idx, name in enumerate(usable):
        qty = max(30, base_qty - idx * 10) * max(1, servings) / 2
        structured_ings.append({"item": name, "quantity": round(qty, 1), "unit": "g"})

    steps: List[Step] = []
    for i, text in enumerate(BASIC_TECHNIQUES[tech], start=1):
        steps.append({"order": i, "instruction": text, "timer_seconds": 240 if i==2 else None, "temperature_c": 200 if tech=="roast" and i==2 else None})

    tags = [tech, skill] + ([cuisine] if cuisine else []) + dietary
    subs = []
    if any(d == "gluten-free" for d in dietary): subs.append("Use gluten-free soy sauce or salt.")
    if any(d in ("vegan","vegetarian") for d in dietary): subs.append("Swap animal protein for tofu or mushrooms.")

    # RAG: borrow structure from nearest example
    client = ensure_collection()
    if use_rag and client and client.count(COLLECTION).count > 0:
        q = build_query_text(usable, cuisine, dietary, tech)
        hits = search_recipes(q, top_k=5, cuisine=cuisine)  # reuse tool logic
        if hits:
            # Optionally fetch top-1 full payload for richer steps
            top = hits[0]
            # (We could call /points to fetch payload, but we already returned it above)
            title = f"Inspired by: {top.get('title')} → {title}"

    recipe: Recipe = {
        "title": title,
        "cuisine": cuisine,
        "servings": servings,
        "total_time_min": max_time_min,
        "ingredients": structured_ings,
        "steps": steps,
        "substitutions": subs,
        "tags": [t for t in tags if t],
        "nutrition": rough_nutrition(structured_ings, servings),
    }

    return recipe

@mcp.tool()
def scale_recipe(recipe: Recipe, new_servings: int) -> Recipe:
    """Scale ingredient quantities and per-serving nutrition to new servings."""
    if new_servings <= 0: new_servings = 1
    old = max(1, recipe.get("servings", 1))
    ratio = new_servings / old
    new_ings: List[Ingredient] = []
    for ing in recipe["ingredients"]:
        q = round(float(ing.get("quantity", 0)) * ratio, 2)
        new_ings.append({**ing, "quantity": q})
    nut = recipe.get("nutrition", {"calories":0,"protein_g":0,"fat_g":0,"carbs_g":0,"sodium_mg":0})
    per_ratio = old / new_servings
    new_nut = {
        "calories": int(nut.get("calories",0) * per_ratio),
        "protein_g": round(float(nut.get("protein_g",0)) * per_ratio, 1),
        "fat_g": round(float(nut.get("fat_g",0)) * per_ratio, 1),
        "carbs_g": round(float(nut.get("carbs_g",0)) * per_ratio, 1),
        "sodium_mg": int(nut.get("sodium_mg",0) * per_ratio),
    }
    return {**recipe, "servings": new_servings, "ingredients": new_ings, "nutrition": new_nut}

@mcp.tool()
def suggest_substitutions(recipe: Recipe, pantry: List[str] = [], avoid: List[str] = []) -> List[str]:
    """Suggest ingredient swaps based on pantry items and avoid list."""
    tips: List[str] = []
    for ing in recipe.get("ingredients", []):
        name = str(ing.get("item","")).lower()
        if any(a.lower() in name for a in avoid):
            tips.append(f"Consider replacing '{ing.get('item')}' with tofu or mushrooms.")
        if "soy sauce" in name:
            tips.append("Need gluten-free? Use coconut aminos or salt.")
        if "milk" in name:
            tips.append("Lactose-free? Use oat or soy milk.")
    if not tips:
        tips.append("No mandatory substitutions. Adjust salt/sugar to taste.")
    return tips

if __name__ == "__main__":
    ensure_collection()
    mcp.run(transport="stdio")
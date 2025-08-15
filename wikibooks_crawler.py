# file: wikibooks_crawler.py
import json, time, re, sys
from pathlib import Path
import requests
import mwparserfromhell
from tqdm import tqdm

API = "https://en.wikibooks.org/w/api.php"
HEADERS = {"User-Agent": "recipe-mcp-bot/0.1 (contact: dairuiyang3@gmail.com)"}

# 读取分类里的所有菜谱条目
def list_category_members(category="Category:Recipes", limit_per_req=200, max_pages=None):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": limit_per_req,
        "cmtype": "page",
        "format": "json",
    }
    cont = None
    pages = []
    while True:
        if cont:
            params.update(cont)
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        pages.extend(data.get("query", {}).get("categorymembers", []))
        cont = data.get("continue")
        if not cont or (max_pages and len(pages) >= max_pages):
            break
        time.sleep(1.0)  # 礼貌限速
    return pages[:max_pages] if max_pages else pages

# 获取页面 wikitext
def get_wikitext(title):
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "*",
        "titles": title,
        "formatversion": "2",
        "format": "json",
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    pages = js.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        return None
    content = pages[0]["revisions"][0]["slots"]["main"]["content"]
    return content

# 粗解析 wikitext -> ingredients/steps
INGR_HEADS = re.compile(r'^\s*==+\s*(ingredients?)\s*==+', re.I)
PROC_HEADS = re.compile(r'^\s*==+\s*(procedure|method|directions|steps)\s*==+', re.I)

def parse_recipe(wikitext):
    code = mwparserfromhell.parse(wikitext)
    ingredients, steps = [], []
    current = None
    for section in code.get_sections(include_lead=False, include_headings=True, flat=True):
        heading = section.filter_headings()[0].title.strip_code().strip().lower() if section.filter_headings() else ""
        text = section.strip_code().strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if INGR_HEADS.search("== " + heading + " =="):
            # 抓取以*或-开头的行，或半角逗号拆分的行
            for l in lines:
                if l.startswith(("*","-")):
                    ingredients.append(l.lstrip("*- ").strip())
            # 回退方案：没有星号则直接取若干行
            if not ingredients:
                ingredients = lines[:20]
        if PROC_HEADS.search("== " + heading + " =="):
            # 抓取编号行（# 或数字.）
            for l in lines:
                if l.startswith(("#","1.","2.","3.","4.","5.")):
                    steps.append(l.lstrip("# ").strip())
            if not steps:
                steps = lines[:20]
    # 兜底：全局也扫一下
    if not ingredients:
        for l in wikitext.splitlines():
            if l.strip().startswith(("*","-")) and len(l) < 200:
                ingredients.append(l.lstrip("*- ").strip())
    if not steps:
        for l in wikitext.splitlines():
            if l.strip().startswith(("#","1.","2.")):
                steps.append(l.lstrip("# ").strip())
    return ingredients[:50], steps[:100]

def main(out_path="recipes_wikibooks.jsonl", max_pages=300):
    pages = list_category_members(max_pages=max_pages)
    out = Path(out_path).open("w", encoding="utf-8")
    for p in tqdm(pages, desc="fetching"):
        title = p["title"]  # 形如 "Cookbook:Cabbage Kimchi"
        wt = get_wikitext(title)
        if not wt: 
            continue
        ingredients, steps = parse_recipe(wt)
        rec = {
            "id": "wikibooks:" + title.replace(" ", "_"),
            "title": title.split("Cookbook:")[-1],
            "source_url": f"https://en.wikibooks.org/wiki/{title.replace(' ', '_')}",
            "cuisine": None,
            "tags": ["Wikibooks","Recipe"],
            "ingredients": ingredients,
            "steps": steps,
            "servings": None,
            "total_time_min": None,
            "license": "CC BY-SA 3.0",
            "raw_wikitext": wt[:20000]
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        time.sleep(0.5)
    out.close()
    print("Saved ->", out_path)

if __name__ == "__main__":
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    main(max_pages=max_pages)
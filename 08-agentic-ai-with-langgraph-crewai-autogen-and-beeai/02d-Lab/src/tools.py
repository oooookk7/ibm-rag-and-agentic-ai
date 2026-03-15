from dotenv import load_dotenv
load_dotenv()

import os
import base64
import requests
from crewai.tools import tool
from openai import OpenAI
from io import BytesIO
from typing import List, Optional
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Extracting ingredients from image...")

HF_BASE_URL = os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1")
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
HF_TEXT_MODEL = os.getenv(
    "HUGGINGFACE_TEXT_MODEL", "Qwen/Qwen3.5-397B-A17B:novita"
)
HF_VISION_MODEL = os.getenv(
    "HUGGINGFACE_VISION_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct:novita"
)
HF_TEMPERATURE = float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.0"))
HF_HTTP_TIMEOUT = float(os.getenv("HUGGINGFACE_HTTP_TIMEOUT", "240"))
HF_TOOL_MAX_TOKENS = int(os.getenv("HUGGINGFACE_TOOL_MAX_TOKENS", "1200"))
HF_MAX_RETRIES = int(os.getenv("HUGGINGFACE_MAX_RETRIES", "2"))


def _hf_client() -> OpenAI:
    if not HF_API_KEY:
        raise RuntimeError(
            "Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN)."
        )
    return OpenAI(
        api_key=HF_API_KEY,
        base_url=HF_BASE_URL,
        timeout=HF_HTTP_TIMEOUT,
        max_retries=HF_MAX_RETRIES,
    )


def _call_hf(messages, model: str, max_tokens: int | None = None) -> str:
    max_tokens = max_tokens or HF_TOOL_MAX_TOKENS
    response = _hf_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=HF_TEMPERATURE,
    )
    return (response.choices[0].message.content or "").strip()


def _load_image_data_url(image_input: str) -> str:
    if image_input.startswith("http"):
        response = requests.get(image_input, timeout=30)
        response.raise_for_status()
        raw = response.content
        mime = response.headers.get("Content-Type", "image/jpeg").split(";")[0]
    else:
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"No file found at path: {image_input}")
        with open(image_input, "rb") as file:
            raw = file.read()
        ext = os.path.splitext(image_input)[1].lower()
        mime = {
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/jpeg")

    encoded_image = base64.b64encode(BytesIO(raw).read()).decode("utf-8")
    return f"data:{mime};base64,{encoded_image}"


class ExtractIngredientsTool():
    @tool("Extract ingredients")
    def extract_ingredient(image_input: str):
        """
        Extract ingredients from a food item image.
        
        :param image_input: The image file path (local) or URL (remote).
        :return: A list of ingredients extracted from the image.
        """
        data_url = _load_image_data_url(image_input)
        prompt = (
            "Extract all visible food ingredients from this image. "
            "Return only a comma-separated ingredient list."
        )
        return _call_hf(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            model=HF_VISION_MODEL,
            max_tokens=500,
        )


class FilterIngredientsTool:
    @tool("Filter ingredients")
    def filter_ingredients(raw_ingredients: str) -> List[str]:
        """
        Processes the raw ingredient data and filters out non-food items or noise.
        
        :param raw_ingredients: Raw ingredients as a string.
        :return: A list of cleaned and relevant ingredients.
        """
        # Example implementation: parse the raw ingredients string into a list
        # This can be enhanced with more sophisticated parsing as needed
        ingredients = [ingredient.strip().lower() for ingredient in raw_ingredients.split(',') if ingredient.strip()]
        return ingredients

class DietaryFilterTool:
    @tool("Filter based on dietary restrictions")
    def filter_based_on_restrictions(ingredients: List[str], dietary_restrictions: Optional[str] = None) -> List[str]:
        """
        Uses an LLM model to filter ingredients based on dietary restrictions.

        :param ingredients: List of ingredients.
        :param dietary_restrictions: Dietary restrictions (e.g., vegan, gluten-free). Defaults to None.
        :return: Filtered list of ingredients that comply with the dietary restrictions.
        """
        # If no dietary restrictions are provided, return the original ingredients
        if not dietary_restrictions:
            return ingredients

        prompt = f"""
        You are an AI nutritionist specialized in dietary restrictions. 
        Given the following list of ingredients: {', '.join(ingredients)}, 
        and the dietary restriction: {dietary_restrictions}, 
        remove any ingredient that does not comply with this restriction. 
        Return only the compliant ingredients as a comma-separated list with no additional commentary.
        """
        filtered = _call_hf(
            messages=[{"role": "user", "content": prompt}],
            model=HF_TEXT_MODEL,
            max_tokens=250,
        ).lower()
        filtered = filtered.replace("\n", ",")
        filtered_list = [item.strip(" -*.\t") for item in filtered.split(",") if item.strip()]
        return filtered_list

    
class NutrientAnalysisTool():
    @tool("Analyze nutritional values and calories of the dish from uploaded image")
    def analyze_image(image_input: str):
        """
        Provide a detailed nutrient breakdown and estimate the total calories of all ingredients from the uploaded image.
        
        :param image_input: The image file path (local) or URL (remote).
        :return: A string with nutrient breakdown (protein, carbs, fat, etc.) and estimated calorie information.
        """
        data_url = _load_image_data_url(image_input)
        # Assistant prompt (can be customized)
        assistant_prompt = """
            You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:
        1. **Identification**: List each identified food item clearly, one per line.
        2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
        - **[Food Item]**: [Portion Size], [Number of Calories] calories
        Example:
        *   **Salmon**: 6 ounces, 210 calories
        *   **Asparagus**: 3 spears, 25 calories
        3. **Total Calories**: Provide the total number of calories for all food items.
        Example:
        Total Calories: [Number of Calories]
        4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.
        Example:
        *   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]
        5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.
        6. **Disclaimer**: Include the following exact text as a disclaimer:
        The nutritional information and calorie estimates provided are approximate and are based on general food data. 
        Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations. 
        For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.
        Format your response exactly like the template above to ensure consistency.
        """
        return _call_hf(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": assistant_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            model=HF_VISION_MODEL,
            max_tokens=1600,
        )

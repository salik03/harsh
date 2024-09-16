
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import requests
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Set up device (ensure CUDA is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "CUDA is required for this model."

# Define model and processor settings
model_id = "microsoft/Phi-3-vision-128k-instruct"
model_cache_dir = "./my_models/phi_3_vision"

# Create directories for storing models
os.makedirs(model_cache_dir, exist_ok=True)

# Load the processor and model
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=model_cache_dir,
    device_map="auto",  # Automatically map the model to available GPUs
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="eager"
).to(device)


# Utility function to load an image from URL or file
def load_image(image_source: Optional[str] = None, image_file: Optional[UploadFile] = None) -> Image.Image:
    if image_source:
        response = requests.get(image_source)
        image = Image.open(BytesIO(response.content))
    elif image_file:
        image = Image.open(image_file.file)
    else:
        raise HTTPException(status_code=400, detail="Image source or image file required.")
    return image


# Pydantic model for request body
class ImageInput(BaseModel):
    image_url: Optional[str] = None  # Optionally, allow image URL


# Function to generate the prompt and extract data
async def extract_data(image, prompt):
    # Apply the prompt
    messages = [{"role": "user", "content": prompt}]
    processed_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process the image and prompt
    inputs = processor(processed_prompt, [image], return_tensors="pt").to(device)

    # Define generation arguments
    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }

    # Generate response
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    # Remove input tokens and decode the result
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response


@app.post("/extract-medicine-data/")
async def extract_medicine_data(image_url: Optional[ImageInput] = None, image_file: Optional[UploadFile] = File(None)):
    try:
        # Load the image either from URL or file upload
        image = load_image(image_source=image_url.image_url if image_url else None, image_file=image_file)

        # Define the input prompt for medicine data
        prompt = "<|image_1|> Extract only the medicine data from the table, including columns such as medicine name, quantity, expiry date, and price in markdown format. Exclude any additional information like patient details, doctor details, terms and condition, invoice, bill, and GST-related content."

        # Extract data using the prompt
        response = await extract_data(image, prompt)

        # Return the extracted medicine data
        return {"medicine_data": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-demographic-details/")
async def extract_demographic_details(image_url: Optional[ImageInput] = None, image_file: Optional[UploadFile] = File(None)):
    try:
        # Load the image either from URL or file upload
        image = load_image(image_source=image_url.image_url if image_url else None, image_file=image_file)

        # Define the input prompt for demographic details
        prompt = "<|image_1|> Please extract the customer name and customer address."

        # Extract data using the prompt
        response = await extract_data(image, prompt)

        # Return the extracted demographic details
        return {"demographic_details": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the table and demographic data extraction API!"}

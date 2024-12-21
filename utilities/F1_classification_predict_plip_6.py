# conda env:clip_for_test_chb

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("./plip", local_files_only= True)
processor = CLIPProcessor.from_pretrained("./plip", local_files_only= True)

image = Image.open("./test.tif")

inputs = processor(text=["a photo of label 1", "a photo of label 2"],
                   images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  
print(probs)
image.resize((224, 224))

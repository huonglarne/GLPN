import torch
import requests
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import matplotlib.pyplot as plt
from PIL import Image

#Define model and feature extractor
feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

#Prepare image
url = 'https://images.all-free-download.com/images/graphicwebp/simple_room_picture_167607.webp'
image = Image.open(requests.get(url, stream=True).raw)
image

pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)

#Forward pass
with torch.no_grad():
  outputs = model(pixel_values)
  predicted_depth = outputs.predicted_depth

predicted_depth.shape

#Visualize
prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=pixel_values.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
             )
prediction = prediction.squeeze().cpu().numpy()

plt.imshow(prediction, cmap="jet")
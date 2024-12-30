# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

from functools import lru_cache

import folium
import gradio as gr
import pandas as pd
import torch
from einops import rearrange, repeat
from matplotlib import cm
from matplotlib import pyplot as plt
from torch import Tensor, fill, fill_
from tqdm import tqdm

from engine.attempts.cb_finetune import GeoModule
from engine.attempts.lib.dataset import GeoVitDataModule
from engine.attempts.lib.utils import setup_environment
from engine.lib.geo import xyz_to_latlon_torch
from engine.lib.utils import DATA

Batch = tuple[Tensor, Tensor, str, str]  # image, target, country, key

model = GeoModule.load_from_checkpoint(DATA / "models" / "siraka" / "last.ckpt").eval().to("cuda")
config = model.config

setup_environment(config)
datamodule = GeoVitDataModule(config, num_workers=1, cache_size=0, with_key=True)
datamodule.setup()

batch = next(iter(datamodule.val_dataloader()))
batch = datamodule.on_after_batch_transfer(batch, 0)

pano_map = {key: (image[None, ...], target[None, ...], country, key) for image, target, country, key in zip(*batch)}

@lru_cache(maxsize=128)
def reverse_diffusion(pano_id):
  image, target, country, key = pano_map[pano_id]
  image, target = image.to("cuda"), target.to("cuda")
  with torch.no_grad():
    vit_features = model.forward_vit(image)
    hat, trace = model.diffusion.reverse_random(
        lambda x, var: model.forward_diffusion(repeat(vit_features, "b c -> (n b) c", n=1), var, x),
        lambda: torch.randn((1, *target.shape), device=target.device, dtype=target.dtype) * model.config.init_noise_scale,
        return_x0_trace=True,
    )
    return hat.cpu(), trace.cpu()

def calc_score(pano_id):
  image, target, country, key = pano_map[pano_id]
  hat, _ = reverse_diffusion(pano_id)
  return model.geoguess_score(hat, target).item()

def preview_image(image):
  projections = image.size(0) // 3
  fig, axes = plt.subplots(1, projections, figsize=(5 * projections, 5))
  for i in range(projections):
    image_i = image[i * 3:i * 3 + 3]
    axes[i].imshow(rearrange(image_i / 2 + 0.5, "c h w -> h w c"))  # type: ignore
    axes[i].axis('off')  # type: ignore
  return fig

def on_select(_, evt: gr.SelectData):
  pano_id = evt.row_value[0]

  image, target, country, key = pano_map[pano_id]
  hat, trace = reverse_diffusion(pano_id)
  score = model.geoguess_score(hat, target).item()
  lat, lon = xyz_to_latlon_torch(*target[0])
  hat_lat, hat_lon = xyz_to_latlon_torch(*hat[0])
  hat_radius = torch.norm(hat[0]).item()
  result = f"Pano ID: {key} ({country})\nScore: {score:.2f}\nRadius: {hat_radius:.4f}"
  fig = preview_image(image[0])

  map_center = [(lat.item() + hat_lat.item()) / 2, (lon.item() + hat_lon.item()) / 2]
  folium_map = folium.Map(location=map_center, zoom_start=2)
  actual_popup = f'<a href="https://www.google.com/maps?q={lat.item()},{lon.item()}" target="_blank">Actual</a>'
  predicted_popup = f'<a href="https://www.google.com/maps?q={hat_lat.item()},{hat_lon.item()}" target="_blank">Predicted</a>'
  folium.Marker([lat.item(), lon.item()], popup=actual_popup, icon=folium.Icon(color="green")).add_to(folium_map)
  folium.Marker([hat_lat.item(), hat_lon.item()], popup=predicted_popup, icon=folium.Icon(color="red")).add_to(folium_map)

  colormap = cm.get_cmap('viridis')
  for i in range(trace.size(0)):
    step = i / (trace.size(0) - 1)
    color = colormap(1 - step)
    lat, lon = xyz_to_latlon_torch(*trace[i, 0])
    radius = torch.norm(trace[i, 0]).item()
    popup = f'<a href="https://www.google.com/maps?q={lat.item()},{lon.item()}" target="_blank">Step {i} ({radius:.4f})</a>'
    folium.CircleMarker(
        [lat.item(), lon.item()],
        radius=5,
        popup=popup,
        color=f"rgb({color[0] * 150:.0f}, {color[1] * 150:.0f}, {color[2] * 150:.0f})",
        fill_color=f"rgb({color[0] * 255:.0f}, {color[1] * 255:.0f}, {color[2] * 255:.0f})",
        fill_opacity=0.3,
        fill=True,
        opacity=0.5,
        weight=2,
    ).add_to(folium_map)

  return fig, folium_map._repr_html_(), result

df = pd.DataFrame({
    "Pano ID": [key for key in pano_map.keys()],
    "Country": [country for _, _, country, _ in pano_map.values()],
    "Score": [round(calc_score(key)) for key in tqdm(list(pano_map.keys()))]
})

with gr.Blocks(title="GeoGuessr Interface") as interface:
  gr.Markdown("# GeoGuessr Interface")
  with gr.Row():
    dataframe = gr.Dataframe(df, label="Select Pano ID", interactive=False)
    with gr.Column():
      panorama_image = gr.Plot(label="Panorama Image", show_label=False)
      map_output = gr.HTML(label="Map", show_label=False)
      inference_result = gr.Textbox(label="Inference Result", show_label=False)

  dataframe.select(on_select, inputs=[dataframe], outputs=[panorama_image, map_output, inference_result])

interface.launch()

import math
import numpy as np
from PIL import Image
import torch
import os
import time
import cv2
from modules import images, devices, processing, shared
from modules.processing import process_images
from modules.scripts import Script, AlwaysVisible
from modules.shared import opts, state
from dataclasses import dataclass

@dataclass
class GridMetadata:
    tiles: list
    tile_w: int
    tile_h: int
    overlap_x: int
    overlap_y: int
    image_w: int
    image_h: int

class Script(Script):
    def title(self):
        return "Tile Generation"

    def describe(self):
        return "Generate images by splitting into tiles, processing each tile with img2img, and recombining"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        import gradio as gr
        
        with gr.Row():
            grid_size = gr.Slider(minimum=2, maximum=8, step=1, label='Grid Size', value=2)
        with gr.Row():
            overlap_x = gr.Slider(minimum=0, maximum=256, step=16, label='Horizontal Overlap', value=64)
            overlap_y = gr.Slider(minimum=0, maximum=256, step=16, label='Vertical Overlap', value=128)
        with gr.Row():
            mask_blur = gr.Slider(minimum=0, maximum=256, step=16, label='Mask Blur', value=0)
        
        return [grid_size, overlap_x, overlap_y, mask_blur]

    def process_tile(self, tile, target_w, target_h):
        return tile.resize((target_w, target_h), Image.LANCZOS)

    def split_image_into_tiles(self, image, grid_size, overlap_x, overlap_y):
        img_w, img_h = image.size
        tile_w = math.ceil(img_w / grid_size)
        tile_h = math.ceil(img_h / grid_size)

        tiles = []
        for y in range(grid_size):
            row = []
            for x in range(grid_size):
                x1 = max(0, x * tile_w - overlap_x)
                y1 = max(0, y * tile_h - overlap_y)
                x2 = min(img_w, (x + 1) * tile_w + overlap_x)
                y2 = min(img_h, (y + 1) * tile_h + overlap_y)
                
                tile = image.crop((x1, y1, x2, y2))
                row.append([x1, y1, x2-x1, y2-y1, tile])
            tiles.append(row)

        return GridMetadata(
            tiles=tiles,
            tile_w=tile_w,
            tile_h=tile_h,
            overlap_x=overlap_x,
            overlap_y=overlap_y,
            image_w=img_w,
            image_h=img_h
        )

    def create_tile_mask(self, size, overlap_x, overlap_y, blur_amount):
        mask = np.ones(size, dtype=np.float32)
        h, w = size
        
        # Create gradients for overlap regions
        if overlap_x > 0:
            gradient_x = np.linspace(0, 1, overlap_x * 2)
            for i in range(w):
                if i < overlap_x:
                    mask[:, i] *= gradient_x[i]
                if i >= w - overlap_x:
                    mask[:, i] *= gradient_x[2 * overlap_x - (i - (w - overlap_x)) - 1]
        
        if overlap_y > 0:
            gradient_y = np.linspace(0, 1, overlap_y * 2)
            for i in range(h):
                if i < overlap_y:
                    mask[i, :] *= gradient_y[i]
                if i >= h - overlap_y:
                    mask[i, :] *= gradient_y[2 * overlap_y - (i - (h - overlap_y)) - 1]
        
        # Apply Gaussian blur to the mask if specified
        if blur_amount > 0:
            # Ensure blur_amount is odd
            blur_kernel = 2 * int(blur_amount) + 1
            mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        
        return mask

    def combine_tiles_with_mask(self, tiles, original_size, overlap_x, overlap_y, mask_blur):
        final_image = np.zeros((original_size[1], original_size[0], 3), dtype=np.float32)
        weight_sum = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
        
        for row in tiles:
            for tile_data in row:
                x, y, w, h, tile = tile_data
                
                # Convert tile to numpy array
                tile_np = np.array(tile.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
                
                # Create mask for this tile position
                effective_overlap_x = min(overlap_x, x, original_size[0]-x-w)
                effective_overlap_y = min(overlap_y, y, original_size[1]-y-h)
                
                mask = self.create_tile_mask(
                    (h, w),
                    effective_overlap_x,
                    effective_overlap_y,
                    mask_blur
                )
                
                # Add weighted tile to final image
                final_image[y:y+h, x:x+w] += tile_np * mask[:, :, np.newaxis]
                weight_sum[y:y+h, x:x+w] += mask
        
        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-5)[:, :, np.newaxis]
        final_image = (final_image / weight_sum * 255).clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(final_image)

    def run(self, p, grid_size, overlap_x, overlap_y, mask_blur):
        initial_info = None
        
        # Store original parameters
        original_width = p.width
        original_height = p.height
        original_batch_size = p.batch_size
        
        # Get initial image
        init_img = p.init_images[0]
        init_img = images.flatten(init_img, opts.img2img_background_color)
        
        # Split into grid
        grid = self.split_image_into_tiles(init_img, grid_size, overlap_x, overlap_y)
        
        # Set processing parameters
        p.width = 1024
        p.height = 1024
        p.batch_size = 1
        p.n_iter = 1
        p.do_not_save_grid = False
        p.do_not_save_samples = False
        
        # Change save path to img2img-grids
        original_outpath_grids = p.outpath_grids
        original_outpath_samples = p.outpath_samples
        p.outpath_grids = os.path.join(os.path.dirname(p.outpath_grids), 'img2img-grids')
        p.outpath_samples = os.path.join(os.path.dirname(p.outpath_samples), 'img2img-grids')
        
        # Create directory if it doesn't exist
        os.makedirs(p.outpath_grids, exist_ok=True)
        os.makedirs(p.outpath_samples, exist_ok=True)

        processed_tiles = []
        total_tiles = grid_size * grid_size
        state.job_count = total_tiles
        
        try:
            # Process each row
            for y, row in enumerate(grid.tiles):
                processed_row = []
                for x, tile_data in enumerate(row):
                    state.job = f"Processing tile {y*grid_size + x + 1} of {total_tiles}"
                    
                    # Process tile
                    _, _, _, _, tile = tile_data
                    processed_tile = self.process_tile(tile, 1024, 1024)
                    
                    # Process with img2img
                    p.init_images = [processed_tile]
                    processed = process_images(p)
                    
                    if initial_info is None:
                        initial_info = processed.info
                    
                    processed_row.append([
                        tile_data[0],
                        tile_data[1],
                        tile_data[2],
                        tile_data[3],
                        processed.images[0]
                    ])
                
                processed_tiles.append(processed_row)

            # Combine tiles with masked blending
            final_image = self.combine_tiles_with_mask(
                processed_tiles, 
                (grid.image_w, grid.image_h),
                overlap_x,
                overlap_y,
                mask_blur
            )
            
            # Save the final combined image
            final_image.save(os.path.join(p.outpath_grids, f'combined_{int(time.time())}.png'))
            
            # Create final processed object
            processed.images = [final_image]
            processed.width = original_width
            processed.height = original_height
            processed.info = initial_info
            
            # Restore original parameters and paths
            p.width = original_width
            p.height = original_height
            p.batch_size = original_batch_size
            p.outpath_grids = original_outpath_grids
            p.outpath_samples = original_outpath_samples
            
            return processed

        except Exception as e:
            # Restore original paths even if there's an error
            p.outpath_grids = original_outpath_grids
            p.outpath_samples = original_outpath_samples
            print(f"Error during processing: {str(e)}")
            raise e
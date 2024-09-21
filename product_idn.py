from sentence_transformers import SentenceTransformer
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


demov="Demo1"

# Encode image embeddings, for embedding images, we need the non-multilingual CLIP model
model_name='clip-ViT-L-14' #clip-ViT-L-14, clip-ViT-B-16, clip-ViT-B-32
img_model = SentenceTransformer(model_name)


inventory_path=demov+"/inventory/"
inventory_imgs=os.listdir(inventory_path)

inventory_img_list = [Image.open(inventory_path+filepath).convert("RGB") for filepath in inventory_imgs]

inventory_emb=img_model.encode(inventory_img_list, batch_size=32, convert_to_tensor=True, show_progress_bar=True)


zoom_path=demov+"/zoom/"
zoom_imgs=os.listdir(zoom_path)

zoom_img_list = [Image.open(zoom_path+filepath).convert("RGB") for filepath in zoom_imgs]

zoom_emb=img_model.encode(zoom_img_list, batch_size=32, convert_to_tensor=True, show_progress_bar=True)


cm=cosine_similarity(zoom_emb.detach().cpu().numpy(),inventory_emb.detach().cpu().numpy())

# Adjusted code for better readability
plt.figure(figsize=(15, 10))  # Increase the figure size

# Plot the confusion matrix with better label formatting
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=inventory_imgs, yticklabels=zoom_imgs, 
            annot_kws={"size": 11}, cbar_kws={'label': 'Cosine Similarity'})

plt.xticks(rotation=45, ha='right', fontsize=11)  # Rotate x labels for better visibility
plt.yticks(fontsize=11)  # Increase font size for y ticks
plt.xlabel('Products from inventory', fontsize=11)
plt.ylabel('From CCTV', fontsize=11)
plt.title('Similarity Matrix', fontsize=14)

# Save the confusion matrix
plt.tight_layout()  # Ensure labels fit into the saved image
plt.savefig(f'{demov}/{model_name}_confusion_matrix.png')  # Save as an image file
#plt.show()


def create_composite_image(zoom_img_list, inventory_img_list, similarity_matrix, save_path, padding=10):

    # Step 1: Calculate the size of the composite image
    total_width = 0
    total_height = 0
    row_heights = []
    
    resized_rows = []
    for i, zoom_img in enumerate(zoom_img_list):
        # Get top 2 most similar inventory images
        top_2_indices = np.argsort(similarity_matrix[i])[-2:][::-1]
        top_2_imgs = [inventory_img_list[idx] for idx in top_2_indices]

        # Combine zoom image and inventory images into one row
        row_images = [zoom_img] + top_2_imgs
        max_height = max(img.height for img in row_images)
        row_heights.append(max_height)

        # Resize all images in the row to have the same height, maintaining aspect ratio
        resized_row = []
        for img in row_images:
            aspect_ratio = img.width / img.height
            new_width = int(max_height * aspect_ratio)
            resized_img = img.resize((new_width, max_height))
            resized_row.append(resized_img)
        
        # Add resized row
        resized_rows.append(resized_row)

        # Calculate total width and height for the composite image
        row_width = sum(img.width for img in resized_row) + padding * 2
        total_width = max(total_width, row_width)  # Take the maximum row width
        total_height += max_height + padding  # Add height for each row + padding

    # Add space for the black slit at the top
    label_height = 50
    total_height += label_height

    # Step 2: Create the composite image
    final_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))  # White background

    # Add black slit for labels at the top
    draw = ImageDraw.Draw(final_image)
    draw.rectangle([0, 0, total_width, label_height], fill=(0, 0, 0))  # Create black rectangle

    # Load a font for text (or use a custom font)
    font = ImageFont.load_default()

    # Write the labels on the black slit
    label_texts = ["CCTV", "Best match from inventory", "Second best match"]
    x_offset = padding
    for text in label_texts:
        draw.text((x_offset, label_height // 4), text, font=font, fill=(255, 255, 255))  # White text
        x_offset += (total_width - 2 * padding) // len(label_texts)  # Distribute the labels evenly

    # Step 3: Paste images below the black slit
    y_offset = label_height + padding
    for resized_row, max_height in zip(resized_rows, row_heights):
        x_offset = padding
        for img in resized_row:
            final_image.paste(img, (x_offset, y_offset))
            x_offset += img.width + padding
        y_offset += max_height + padding

    # Step 4: Save the final image
    final_image.save(save_path)
    print(f"Composite image saved at {save_path}")

    return final_image

# Example usage:
save_path = f'{demov}/{model_name}_composite_image.png'
create_composite_image(zoom_img_list, inventory_img_list, cm, save_path, padding=20)

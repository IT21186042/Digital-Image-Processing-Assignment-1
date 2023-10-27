import tkinter as tk
from tkinter import filedialog, Scale
import cv2
from PIL import Image, ImageTk, ImageFilter
import os
import shutil
import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageEnhance


root = tk.Tk()
root.title("Image Manipulator")
root.configure(bg="#222222")

selected_image_path = None
original_image_reference = None
selected_image_reference = None
color_change_canvas = None
grayscale_image_reference = None
segmented_image_reference = None
color_change_canvas = None


history = []
start_x = None
start_y = None
end_x = None
end_y = None
rect_id = None
rotation_angle = tk.DoubleVar()

def create_button(parent, text, command, bg, fg, font):
    button = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, font=font)
    return button


#
def adjust_color():
    global selected_image_reference

    if selected_image_reference:
        try:
            img = ImageTk.getimage(selected_image_reference)

            # Get the values from the scales
            saturation = saturation_scale.get()
            hue = hue_scale.get()
            lightness = lightness_scale.get()

            # Convert the image to the HSV color space
            img_hsv = img.convert('HSV')

            # Adjust the color based on the values
            img_hsv = ImageEnhance.Color(img_hsv).enhance(saturation)
            img_hsv = ImageEnhance.Brightness(img_hsv).enhance(lightness)
            img_hsv = ImageEnhance.Contrast(img_hsv).enhance(lightness)

            # Convert the image back to RGB
            img_rgb = img_hsv.convert('RGB')

            # Update the canvas with the adjusted image
            adjusted_image_reference = ImageTk.PhotoImage(image=img_rgb)
            update_image_canvas(adjusted_image_reference)
        except Exception as e:
            print(f"Error adjusting color: {str(e)}")





# Uncomment the following code for image segmentation
def perform_image_segmentation():
    global selected_image_path, selected_image_reference

    if selected_image_path:
        try:
            img = cv2.imread(selected_image_path)

            # Convert the image to HSV color space for better color-based segmentation
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for the color you want to segment
            lower_bound = np.array([0, 0, 0])  # Adjust these values based on your color
            upper_bound = np.array([255, 255, 255])  # Adjust these values based on your color

            # Create a mask using inRange function
            mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

            # Bitwise AND operation to segment the image
            segmented_image = cv2.bitwise_and(img, img, mask=mask)

            photo_segmented = ImageTk.PhotoImage(image=Image.fromarray(segmented_image))

            # Update the canvas with the segmented image
            update_image_canvas(photo_segmented)
        except Exception as e:
            print(f"Error performing image segmentation: {str(e)}")
    else:
        print("Please select an image before performing image segmentation.")





        
def display_segmented_image():
    global segmented_image_reference
    if segmented_image_reference:
        update_image_canvas(segmented_image_reference)



def update_image_canvas(new_image):
    global selected_image_reference
    selected_image_reference = new_image
    history.append(selected_image_reference)
    canvas.delete("all")
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    img_width = new_image.width()
    img_height = new_image.height()
    x = (canvas_width - img_width) // 2
    y = (canvas_height - img_height) // 2
    canvas.create_image(x, y, anchor=tk.NW, image=selected_image_reference)
    
def update_deep_canvas(new_image):
    global selected_image_reference
    selected_image_reference = ImageTk.PhotoImage(Image.fromarray((new_image * 255).astype('uint8')))
    history.append(selected_image_reference)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=selected_image_reference)


def on_zoom(event):
    global selected_image_reference

    if not selected_image_reference:
        return

    x, y = event.x, event.y
    if event.delta > 0:
        factor = 1.1
    else:
        factor = 0.9

    img = ImageTk.getimage(selected_image_reference)
    img = img.resize((int(img.width * factor), int(img.height * factor)), Image.LANCZOS)

    selected_image_reference = ImageTk.PhotoImage(image=img)
    update_image_canvas(selected_image_reference)

def open_image():
    global selected_image_path, original_image_reference, selected_image_reference, color_change_canvas, grayscale_image_reference
    file_path = filedialog.askopenfilename(parent=root)
    if file_path:
        try:
            img = Image.open(file_path)
            selected_image_path = file_path
            img = img.resize((500, 500))
            selected_image_reference = ImageTk.PhotoImage(image=img)
            original_image_reference = selected_image_reference
            grayscale_image_reference = ImageTk.PhotoImage(image=img.convert("L"))
            update_image_canvas(selected_image_reference)
            if color_change_canvas:
                color_change_canvas.delete("all")
        except Exception as e:
            print(f"Error opening the image: {str(e)}")

def back_to_original():
    global original_image_reference
    if original_image_reference:
        update_image_canvas(original_image_reference)

def remove_image():
    global selected_image_path, selected_image_reference, color_change_canvas, grayscale_image_reference
    selected_image_path = None
    selected_image_reference = None
    canvas.delete("all")
    if color_change_canvas:
        color_change_canvas.delete("all")
    grayscale_image_reference = None

def save_image():
    global selected_image_path, selected_image_reference
    if selected_image_reference:
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                shutil.copy(selected_image_path, file_path)
        except Exception as e:
            print(f"Error saving the image: {str(e)}")

def convert_to_grayscale():
    global selected_image_reference, grayscale_image_reference, color_change_canvas
    if selected_image_reference:
        img = ImageTk.getimage(selected_image_reference)
        grayscale_img = img.convert("L")
        grayscale_image_reference = ImageTk.PhotoImage(image=grayscale_img)
        update_image_canvas(grayscale_image_reference)

def convert_to_bw():
    global selected_image_reference, color_change_canvas
    if selected_image_reference:
        img = ImageTk.getimage(selected_image_reference)
        img_array = np.array(img)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, img_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
        photo_bw = ImageTk.PhotoImage(image=Image.fromarray(img_bw))
        update_image_canvas(photo_bw)
        
        
        
def undo_last_action():
    global selected_image_reference
    if len(history) > 1:
        history.pop()
        selected_image_reference = history[-1]
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=selected_image_reference)

def start_crop(event):
    global start_x, start_y
    start_x = event.x
    start_y = event.y

def display_crop_rectangle(event):
    global start_x, start_y, rect_id
    if not start_x or not start_y:
        return
    if rect_id:
        canvas.delete(rect_id)
    rect_id = canvas.create_rectangle(start_x, start_y, event.x, event.y, outline='red')

def end_crop(event):
    global start_x, start_y, end_x, end_y, selected_image_reference
    end_x, end_y = event.x, event.y
    if selected_image_reference and start_x and start_y and end_x and end_y:
        try:
            img = ImageTk.getimage(selected_image_reference)
            cropped_img = img.crop((start_x, start_y, end_x, end_y))
            cropped_img_reference = ImageTk.PhotoImage(image=cropped_img)
            update_image_canvas(cropped_img_reference)
        except Exception as e:
            print(f"Error cropping the image: {str(e)}")
    start_x = start_y = end_x = end_y = None
    if rect_id:
        canvas.delete(rect_id)

def flip_image():
    global selected_image_reference
    if selected_image_reference:
        try:
            img = ImageTk.getimage(selected_image_reference)
            flipped_img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            flipped_img_reference = ImageTk.PhotoImage(image=flipped_img)
            update_image_canvas(flipped_img_reference)
        except Exception as e:
            print(f"Error flipping the image: {str(e)}")

def hide_rotation_slider(event=None):
    rotation_slider.pack_forget()

def setup_rotation_slider():
    global rotation_slider
    rotation_slider = Scale(root, from_=0, to=360, orient=tk.HORIZONTAL, variable=rotation_angle, label="Rotation Angle")
    rotation_slider.pack(pady=0)
    rotation_slider.bind("<ButtonRelease>", rotate_image_using_slider)

def rotate_image_using_slider(event):
    global selected_image_reference, rotation_angle
    if selected_image_reference:
        try:
            angle = rotation_angle.get()
            img = ImageTk.getimage(selected_image_reference)
            rotated_img = img.rotate(-angle, expand=True, resample=Image.BICUBIC)
            rotated_img_reference = ImageTk.PhotoImage(image=rotated_img)
            update_image_canvas(rotated_img_reference)
            hide_rotation_slider()
        except Exception as e:
            print(f"Error rotating the image: {str(e)}")
            
            
#style transfer part
def preprocess_image(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [512, 512])
    image = image[tf.newaxis, :]
    return image

# Function to create a VGG-19 model
def vgg_model(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    return model

# Function to calculate the Gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Function to perform neural style transfer
def style_transfer(content_image, style_image, epochs=10, style_weight=1e-2, content_weight=1e-4):
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    vgg = vgg_model(style_layers + content_layers)

    style_features = [gram_matrix(style) for style in vgg(preprocess_image(style_image))[:num_style_layers]]
    content_features = vgg(preprocess_image(content_image))[num_style_layers:]

    image = tf.Variable(preprocess_image(content_image))
    optimizer = tf.optimizers.Adam(learning_rate=0.02)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            tape.watch(image)
            outputs = vgg(image)
            loss = tf.zeros(shape=())

            style_weight /= num_style_layers
            for gram_target, output in zip(style_features, outputs[:num_style_layers]):
                loss += style_weight * tf.reduce_mean((gram_matrix(output) - gram_target) ** 2)

            content_weight /= num_content_layers
            for content_target, output in zip(content_features, outputs[num_style_layers:]):
                loss += content_weight * tf.reduce_mean((output - content_target) ** 2)

        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

    return image


style_image_path='images/starnight.jpg'

def perform_style_transfer():
    global selected_image_path

    # Check if a valid image is selected
    if selected_image_path:
        
        content_image = selected_image_path
        

        # Path to your style image
        style_image_path = 'images/starnight.jpg'

        style_image=style_image_path 

        # Perform style transfer
        stylized_image = style_transfer(content_image, style_image, epochs=10)
        display_stylized_image(stylized_image)
        
        
    else:
        print("Please select an image before performing style transfer.")


# Function to display the stylized image
def display_stylized_image(stylized_image):
    img = stylized_image[0].numpy()
    img = np.clip(img, 0, 1)
    update_deep_canvas(img)
    

header_frame = tk.Frame(root, bg="#03254C")
header_frame.pack(fill="x")

open_button = create_button(header_frame, "Open Image", open_image, "#333333", "white", ("Helvetica", 12))
open_button.pack(side=tk.RIGHT, padx=20, pady=10)

back_to_original_button = create_button(header_frame, "Back To Original", back_to_original, "#333333", "white", ("Helvetica", 12))
back_to_original_button.pack(side=tk.RIGHT, padx=10, pady=10)

undo_button = create_button(header_frame, "Undo", undo_last_action, "#333333", "white", ("Helvetica", 12))
undo_button.pack(side=tk.RIGHT, padx=10, pady=10)

download_button = create_button(header_frame, "Download Image", save_image, "#333333", "white", ("Helvetica", 12))
download_button.pack(side=tk.RIGHT, padx=10, pady=10)

sidebar_frame = tk.Frame(root, width=250, bg="#001133")
sidebar_frame.pack(fill="y", side=tk.LEFT)

right_sidebar_frame = tk.Frame(root, width=550, bg="#001133")
right_sidebar_frame.pack(fill="y", side=tk.RIGHT)

# Add value bars for saturation, hue, and lightness
saturation_scale = Scale(right_sidebar_frame, from_=0, to=2, orient="horizontal", length=200, resolution=0.1, label="Saturation", background="#333333", foreground="white")
saturation_scale.pack(pady=15)

hue_scale = Scale(right_sidebar_frame, from_=0, to=2, orient="horizontal", length=200, resolution=0.1, label="Hue", background="#333333", foreground="white")
hue_scale.pack(pady=10)

lightness_scale = Scale(right_sidebar_frame, from_=-1, to=1, orient="horizontal",length=200, resolution=0.1, label="Lightness", background="#333333", foreground="white")
lightness_scale.pack(pady=10)


# Create a button to trigger color adjustment
adjust_color_button = create_button(right_sidebar_frame, "Adjust Color", adjust_color, "#333333", "white", ("Helvetica", 12))
adjust_color_button.pack(pady=10)

# Create a button for image segmentation
segmentation_button = create_button(right_sidebar_frame, "Image Segmentation", perform_image_segmentation, "#333333", "white", ("Helvetica", 12))
segmentation_button.pack(pady=10)


right_sidebar_label = tk.Label(right_sidebar_frame, text="Advanced Conversions",font=("Helvetica", 14, "bold"), background="#001133", foreground="white")
right_sidebar_label.pack(pady=20)

style_transfer_button = create_button(right_sidebar_frame, "Style Transfer", perform_style_transfer, "#333333", "white", ("Helvetica", 12))
style_transfer_button.pack(pady=10)


basic_conversions_label = tk.Label(sidebar_frame, text="Basic Conversions", font=("Helvetica", 14, "bold"), background="#001133", foreground="white")
basic_conversions_label.pack(pady=10)

color_conversions_label = tk.Label(sidebar_frame, text="Color Conversions", font=("Helvetica", 12), background="#001133", foreground="white")
color_conversions_label.pack(pady=10)

convert_to_grayscale_button = create_button(sidebar_frame, "Convert to Grayscale", convert_to_grayscale, "#333333", "white", ("Helvetica", 12))
convert_to_grayscale_button.pack(pady=10)

convert_to_bw_button = create_button(sidebar_frame, "Convert to B/W", convert_to_bw, "#333333", "white", ("Helvetica", 12))
convert_to_bw_button.pack(pady=10)







transformations_menu = tk.Menubutton(sidebar_frame, text="Transformations", relief=tk.RAISED, bg="#333333", fg="white", font=("Helvetica", 12))
transformations_menu.pack(pady=10)
transformations_menu.menu = tk.Menu(transformations_menu, tearoff=0)
transformations_menu["menu"] = transformations_menu.menu

def setup_crop_bindings():
    canvas.bind("<Button-1>", start_crop)
    canvas.bind("<B1-Motion>", display_crop_rectangle)
    canvas.bind("<ButtonRelease-1>", end_crop)

crop_image_option = transformations_menu.menu.add_command(label="Crop Image", command=setup_crop_bindings)
rotate_image_option = transformations_menu.menu.add_command(label="Rotate Image", command=setup_rotation_slider)
flip_image_option = transformations_menu.menu.add_command(label="Flip Image", command=flip_image)

canvas = tk.Canvas(root, width=500, height=500, bg="#4F738E")
canvas.pack(pady=50, padx=50)
canvas.bind("<MouseWheel>", on_zoom)

remove_image_button = create_button(root, "Remove Image", remove_image, "#333333", "white", ("Helvetica", 12))
remove_image_button.pack(side=tk.BOTTOM, pady=10)

Filters_label = tk.Label(sidebar_frame, text="Filters", font=("Helvetica", 12), background="#333333", foreground="white")
Filters_label.pack(pady=10)

strength_label = tk.Label(sidebar_frame, text="Filter Strength", background="#333333", foreground="white")
strength_label.pack()
strength_scale = tk.Scale(sidebar_frame, from_=0, to=10, orient="horizontal", length=200, resolution=0.1, background="#333333", foreground="white")
strength_scale.pack(pady=10)

filter_var = tk.StringVar()
filter_var.set("none")
filter_menu = tk.OptionMenu(sidebar_frame, filter_var, "none", "sharpen", "smooth", "edge", "emboss", "blur", "contour", "detail")
filter_menu.configure(background="#333333", foreground="white")
filter_menu.pack(pady=10)





def apply_filter():
    global selected_image_reference
    strength = strength_scale.get()

    if strength == 0:
        update_image_canvas(original_image_reference)
        return

    if selected_image_reference:
        try:
            filter_type = filter_var.get()
            filters = {
                "sharpen": ImageFilter.SHARPEN,
                "smooth": ImageFilter.SMOOTH,
                "edge": ImageFilter.FIND_EDGES,
                "emboss": ImageFilter.EMBOSS,
                "blur": ImageFilter.BLUR,
                "contour": ImageFilter.CONTOUR,
                "detail": ImageFilter.DETAIL
            }
            filter_function = filters.get(filter_type, None)

            if filter_function:
                img = ImageTk.getimage(selected_image_reference)
                filtered_image = img.filter(filter_function)
                filtered_image = Image.blend(img, filtered_image, strength)
                filtered_image = filtered_image.resize((500, 500))
                selected_image_reference = ImageTk.PhotoImage(image=filtered_image)
                update_image_canvas(selected_image_reference)
            else:
                # If no filter is selected, update the image with the original image
                update_image_canvas(original_image_reference)
        except Exception as e:
            print(f"Error applying filter: {str(e)}")

apply_filter_button = create_button(sidebar_frame, "Apply Filter", apply_filter, "#333333", "white", ("Helvetica", 12))
apply_filter_button.pack(pady=10)






root.mainloop()
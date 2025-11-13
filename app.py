import os
import random
from PIL import Image
import streamlit as st
from image_patterns import IMAGE_PATTERNS
from main import create_circle_on_image, create_ellipse_on_image, create_circle_ellipse_merged_on_image

# Page config
st.set_page_config(page_title="Defect Generator", layout="wide")

# Custom CSS for minimal styling

# Header
st.title("Image Shape Defect Generator")

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'defect_image' not in st.session_state:
    st.session_state.defect_image = None
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = "images/"
    os.makedirs(st.session_state.output_dir, exist_ok=True)


def generate_random_image(size=(512, 512), seed=None):
    """Generate a random pattern image"""
    if seed is None:
        seed = random.randint(1, 10000)

    random_img_idx = random.choice(range(len(IMAGE_PATTERNS)))
    random_img = IMAGE_PATTERNS[random_img_idx](size=size, seed=seed)
    return Image.fromarray(random_img, mode="RGB")


def save_temp_image(img):
    """Save PIL Image to generated_images folder and return path"""
    os.makedirs(st.session_state.output_dir, exist_ok=True)
    temp_path = os.path.join(st.session_state.output_dir,
                             f"input_{random.randint(1000, 9999)}.png")
    img.save(temp_path)
    return temp_path


def generate_defect(image, defect_type, seed=None):
    """Generate defect on image based on type"""
    if seed is None:
        seed = random.randint(1, 10000)

    # Save original image to generated_images folder
    input_path = save_temp_image(image)
    output_dir = st.session_state.output_dir

    try:
        if defect_type == "Circle":
            output_path = create_circle_on_image(
                input_path,
                output_dir,
                radius=random.randint(20, 80),
                seed=seed
            )
        elif defect_type == "Ellipse":
            output_path = create_ellipse_on_image(
                input_path,
                output_dir,
                radius_x=random.randint(20, 60),
                radius_y=random.randint(30, 90),
                seed=seed
            )
        else:  # Circle + Ellipse
            output_path = create_circle_ellipse_merged_on_image(
                input_path,
                output_dir,
                ellipse_rx=random.randint(25, 50),
                ellipse_ry=random.randint(40, 80),
                circle_radius=random.randint(30, 70),
                seed=seed
            )

        return Image.open(output_path)
    except Exception as e:
        st.error(f"Error generating defect: {str(e)}")
        return None


# Main layout - 2 columns
col1, col2 = st.columns([1, 1])

# Column 1 - Image Input and Preview
with col1:
    st.markdown("### Image Input")

    # Upload section
    uploaded_file = st.file_uploader("Upload PNG Image", type=[
                                     'png'], label_visibility="collapsed",)

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state.original_image = img
        st.session_state.defect_image = None

    # Generate random image button
    if st.button("Generate Random Image", type="primary"):
        img = generate_random_image()
        st.session_state.original_image = img
        st.session_state.defect_image = None

    # Show original image preview
    if st.session_state.original_image is not None:
        st.markdown("### Original Image")
        st.image(st.session_state.original_image, width='stretch')
    else:
        st.info("Upload an image or generate a random one to get started")

# Column 2 - Defect Selection and Defect Preview
with col2:
    if st.session_state.original_image is not None:
        st.markdown("### Defect Shape Generation")

        # Defect type selection
        defect_type = st.radio(
            "Select Defect Type",
            ["Circle", "Ellipse", "Circle + Ellipse"],
            label_visibility="collapsed",
            horizontal=True,
            width="stretch"
        )
        st.space()
        # Generate defect button
        if st.button("Generate Defect", type="primary", disabled=False):
            with st.spinner("Generating defect..."):
                defect_img = generate_defect(
                    st.session_state.original_image, defect_type)
                if defect_img:
                    st.session_state.defect_image = defect_img

        # Show defect image preview
        if st.session_state.defect_image is not None:
            st.markdown("### Defect Image")
            st.image(st.session_state.defect_image, width='stretch')


st.markdown("Made by [Parampreet Singh](https://github.com/Param302)")

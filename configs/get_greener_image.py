from PIL import Image
def increase_green(image_path, output_path, green_factor=1.2):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert the image to RGB mode if it is not already
        img = img.convert('RGB')
        
        # Split the image into R, G, B channels
        r, g, b = img.split()

        # Enhance the green channel by multiplying with the green_factor
        g = g.point(lambda i: min(255, int(i * green_factor)))

        # Merge the channels back
        enhanced_img = Image.merge('RGB', (r, g, b))

        # Save the image
        enhanced_img.save(output_path)

        # Display the image (optional)

# Usage
increase_green('../testImages/4.png', 'path_to_save_enhanced_image.png')

# +
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import math
import sionna

sionna.config.xla_compat = True


# -

def jpeg_binary_stream(image, quality=50):
    pil_image = Image.fromarray(image.astype(np.uint8)) #Creates an image memory from an object exporting the array interface (using the buffer protocol):
    
    buffer = io.BytesIO()  # byte buffer to store jpeg image
    pil_image.save(buffer, format='JPEG', quality=quality) # save image to buffer in jpeg format
    
    
    jpeg_data = extract_jpeg_payload(buffer.getvalue()) # extract payload (useful part) and conver binary
    return np.unpackbits(jpeg_data)


def extract_jpeg_payload(jpeg_bytes):
    """
    Extract the actual image data from JPEG binary stream, 
    skipping file format overhead.
    
    Args:
        jpeg_bytes: bytes object containing JPEG data
        
    Returns:
        numpy array containing only the image data bbutytes
    """
    # Find Start Of Scan marker (0xFFDA) 0xFF, 0xDA
    sos_marker = b'\xFF\xDA'
    sos_pos = jpeg_bytes.find(sos_marker)
    
    if sos_pos == -1:
        raise ValueError("Invalid JPEG data: Start Of Scan marker not found")
    
    # Skip SOS marker and its length field
    sos_length = 2 + int.from_bytes(jpeg_bytes[sos_pos+2:sos_pos+4], 'big')
    payload_start = sos_pos + sos_length
    
    # Find End Of Image marker (0xFFD9) 0xFF, 0xD9
    eoi_marker = b'\xFF\xD9'
    eoi_pos = jpeg_bytes.find(eoi_marker, payload_start)
    
    if eoi_pos == -1:
        raise ValueError("Invalid JPEG data: End Of Image marker not found")
    
    # Extract payload between SOS and EOI
    return np.frombuffer(jpeg_bytes[payload_start:eoi_pos], dtype=np.uint8) # actuall compressed data

def jpeg_binary_stream_full(image, quality=50):
    """
    Convert numpy array image to JPEG binary stream.
    
    Args:
        image: numpy array of image data
        quality: JPEG quality factor (1-100)
    
    Returns:
        numpy array of binary values (0s and 1s)
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image.astype(np.uint8))
    
    # Save to buffer in JPEG format
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    
    # Get binary data
    buffer.seek(0)
    binary_data = np.frombuffer(buffer.read(), dtype=np.uint8)
    
    # Convert to bit stream
    return np.unpackbits(binary_data)


def jpeg_binary_stream_decode(binary_stream, original_shape):
    # Ensure input is a 1D array of 0s and 1s
    if binary_stream.ndim != 1:
        raise ValueError("Input binary stream must be a 1D array")
    
    # Verify that the binary stream length is divisible by 8
    if len(binary_stream) % 8 != 0:
        raise ValueError("Binary stream length must be divisible by 8")
    
    # Pack bits back into bytes
    packed_bytes = np.packbits(binary_stream)
    
    # Create a BytesIO buffer with the packed bytes
    buffer = io.BytesIO(packed_bytes)
    
    try:
        # Open the image from the buffer
        pil_image = Image.open(buffer)
        
        # Convert PIL Image to numpy array
        decoded_image = np.array(pil_image)
        
        # Verify shape matches expected shape
        if decoded_image.shape != original_shape:
            print(f"Warning: Decoded image shape {decoded_image.shape} does not match expected shape {original_shape}")
        
        return decoded_image
    
    except Exception as e:
        raise ValueError(f"Error decoding JPEG binary stream: {str(e)}")


def create_5g_ldpc_encoder_decoder(K=672, N=1344):
    """
    Create 5G NR LDPC Encoder and Decoder
    Args:
        K (int): Number of information bits
        N (int): Codeword length
    """
    # Base graph 2 (more common in 5G)
    encoder = sionna.fec.ldpc.LDPC5GEncoder(k=K,n=N)
    
    decoder = sionna.fec.ldpc.LDPC5GDecoder(encoder = encoder,num_iter=20)
    
    return encoder, decoder


def compare_images(original_image, reconstructed_image, quality=None):
    """
    Compare original and reconstructed images side by side
    
    Args:
        original_image: numpy array of the original image
        reconstructed_image: numpy array of the reconstructed image
        quality: optional quality parameter to display
    """
    # Create a figure with two subplots side by side
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    if original_image.ndim == 2:  # Grayscale
        plt.imshow(original_image, cmap='gray')
    else:  # Color image
        plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Add shape information for original image
    plt.text(0.5, -0.1, f"Shape: {original_image.shape}", 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform=plt.gca().transAxes)
    
    # Reconstructed Image
    plt.subplot(1, 2, 2)
    if reconstructed_image.ndim == 2:  # Grayscale
        plt.imshow(reconstructed_image, cmap='gray')
    else:  # Color image
        plt.imshow(reconstructed_image)
    
    # Title with quality if provided
    title = 'Reconstructed Image'
    if quality is not None:
        title += f' (Quality: {quality})'
    plt.title(title)
    plt.axis('off')
    
    # Add shape information for reconstructed image
    plt.text(0.5, -0.1, f"Shape: {reconstructed_image.shape}", 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform=plt.gca().transAxes)
    
    # Calculate and display image similarity metrics
    mse = np.mean((original_image - reconstructed_image) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    plt.suptitle(f'Image Comparison\nMSE: {mse:.4f}, PSNR: {psnr:.2f} dB', 
                 fontsize=12)
    
    plt.tight_layout()
    plt.show()

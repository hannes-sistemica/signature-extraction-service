# This is just an example of how to use the OpenCV library to extract signatures from a PDF file
import cv2
import numpy as np
import pdf2image
import os

def convert_pdf_to_images(pdf_path):
    """
    Convert PDF pages to images
    """
    return pdf2image.convert_from_path(pdf_path)

def preprocess_image(image):
    """
    Preprocess the image for signature detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def detect_signature_regions(preprocessed_image):
    """
    Detect potential signature regions using contour analysis
    """
    # Find contours
    contours, _ = cv2.findContours(
        preprocessed_image, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    signature_regions = []
    
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Filter small regions
        if area < 500:
            continue
            
        # Calculate shape complexity (signature tends to be complex)
        complexity = perimeter * perimeter / (4 * np.pi * area)
        
        # Signatures typically have high complexity
        if complexity > 20:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Typical signature aspect ratios
            if 0.5 < aspect_ratio < 5:
                signature_regions.append((x, y, w, h))
    
    return signature_regions

def analyze_pdf_for_signatures(pdf_path):
    """
    Main function to analyze PDF for signatures
    """
    # Convert PDF to images
    images = convert_pdf_to_images(pdf_path)
    
    results = []
    
    for page_num, image in enumerate(images, 1):
        # Preprocess the image
        preprocessed = preprocess_image(image)
        
        # Detect signature regions
        regions = detect_signature_regions(preprocessed)
        
        if regions:
            results.append({
                'page': page_num,
                'regions': regions
            })
    
    return results

def visualize_results(pdf_path, results):
    """
    Visualize detected signatures on the PDF pages
    """
    images = convert_pdf_to_images(pdf_path)
    
    for result in results:
        page_num = result['page']
        image = np.array(images[page_num - 1])
        
        # Draw rectangles around detected signatures
        for x, y, w, h in result['regions']:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save or display the result
        output_path = f'detected_signatures_page_{page_num}.png'
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def main():
    pdf_path = 'file.pdf'
    
    # Detect signatures
    results = analyze_pdf_for_signatures(pdf_path)
    
    if results:
        print(f"Found signatures on {len(results)} pages:")
        for result in results:
            print(f"Page {result['page']}: {len(result['regions'])} signature(s) detected")
        
        # Visualize results
        visualize_results(pdf_path, results)
    else:
        print("No signatures detected")

if __name__ == "__main__":
    main()

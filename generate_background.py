import numpy as np
import os
import json
import time

def create_smooth_green_aquamarine_gradient(output_svg_path, output_bounds_path):
    """
    Generate a beautiful smooth gradient background with green and aquamarine colors.
    
    Args:
        output_svg_path: Path to save the SVG file
        output_bounds_path: Path to save the data bounds JSON
    """
    print("Generating beautiful smooth green and aquamarine gradient background...")
    start_time = time.time()
    
    # Define data bounds (same as before for consistency)
    bounds = {
        'x_min': -115.94,
        'x_max': 112.19,
        'y_min': -103.64,
        'y_max': 103.30
    }
    
    # Save bounds
    with open(output_bounds_path, 'w') as f:
        json.dump(bounds, f, indent=2)
    
    print(f"Data bounds: X[{bounds['x_min']:.2f}, {bounds['x_max']:.2f}], Y[{bounds['y_min']:.2f}, {bounds['y_max']:.2f}]")
    
    # Start building SVG
    svg_width = 1200
    svg_height = 800
    
    # Start SVG content
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <!-- Multiple radial gradients radiating from different points -->
        <radialGradient id="gradient1" cx="25%" cy="30%" r="50%">
            <stop offset="0%" style="stop-color:#228B22;stop-opacity:1" />
            <stop offset="30%" style="stop-color:#32CD32;stop-opacity:0.8" />
            <stop offset="60%" style="stop-color:#00CED1;stop-opacity:0.6" />
            <stop offset="100%" style="stop-color:#40E0D0;stop-opacity:0.3" />
        </radialGradient>
        
        <radialGradient id="gradient2" cx="75%" cy="25%" r="45%">
            <stop offset="0%" style="stop-color:#006400;stop-opacity:1" />
            <stop offset="25%" style="stop-color:#228B22;stop-opacity:0.8" />
            <stop offset="50%" style="stop-color:#00CED1;stop-opacity:0.6" />
            <stop offset="100%" style="stop-color:#7FFFD4;stop-opacity:0.3" />
        </radialGradient>
        
        <radialGradient id="gradient3" cx="20%" cy="70%" r="55%">
            <stop offset="0%" style="stop-color:#32CD32;stop-opacity:1" />
            <stop offset="35%" style="stop-color:#00CED1;stop-opacity:0.8" />
            <stop offset="70%" style="stop-color:#40E0D0;stop-opacity:0.6" />
            <stop offset="100%" style="stop-color:#98FB98;stop-opacity:0.3" />
        </radialGradient>
        
        <radialGradient id="gradient4" cx="80%" cy="75%" r="40%">
            <stop offset="0%" style="stop-color:#00CED1;stop-opacity:1" />
            <stop offset="40%" style="stop-color:#40E0D0;stop-opacity:0.8" />
            <stop offset="75%" style="stop-color:#7FFFD4;stop-opacity:0.6" />
            <stop offset="100%" style="stop-color:#F0FFF0;stop-opacity:0.3" />
        </radialGradient>
        
        <radialGradient id="gradient5" cx="50%" cy="50%" r="35%">
            <stop offset="0%" style="stop-color:#228B22;stop-opacity:0.8" />
            <stop offset="50%" style="stop-color:#00CED1;stop-opacity:0.6" />
            <stop offset="100%" style="stop-color:#7FFFD4;stop-opacity:0.4" />
        </radialGradient>
        
        <!-- Additional diagonal gradient for more complexity -->
        <linearGradient id="diagonalGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#006400;stop-opacity:0.2" />
            <stop offset="25%" style="stop-color:#228B22;stop-opacity:0.3" />
            <stop offset="50%" style="stop-color:#00CED1;stop-opacity:0.4" />
            <stop offset="75%" style="stop-color:#40E0D0;stop-opacity:0.3" />
            <stop offset="100%" style="stop-color:#7FFFD4;stop-opacity:0.2" />
        </linearGradient>
        
        <!-- Fade to white gradient for areas without data -->
        <radialGradient id="fadeGradient" cx="50%" cy="50%" r="80%">
            <stop offset="0%" style="stop-color:#FFFFFF;stop-opacity:0.0" />
            <stop offset="40%" style="stop-color:#F8FFFE;stop-opacity:0.2" />
            <stop offset="70%" style="stop-color:#F0FFFA;stop-opacity:0.5" />
            <stop offset="100%" style="stop-color:#FFFFFF;stop-opacity:0.9" />
        </radialGradient>
    </defs>
    
    <!-- Multiple gradient layers radiating from different points -->
    <rect width="{svg_width}" height="{svg_height}" fill="url(#gradient1)"/>
    <rect width="{svg_width}" height="{svg_height}" fill="url(#gradient2)"/>
    <rect width="{svg_width}" height="{svg_height}" fill="url(#gradient3)"/>
    <rect width="{svg_width}" height="{svg_height}" fill="url(#gradient4)"/>
    <rect width="{svg_width}" height="{svg_height}" fill="url(#gradient5)"/>
    
    <!-- Additional diagonal gradient overlay for more depth -->
    <rect width="{svg_width}" height="{svg_height}" fill="url(#diagonalGradient)"/>
    
    <!-- Fade to white overlay for areas without data -->
    <rect width="{svg_width}" height="{svg_height}" fill="url(#fadeGradient)"/>
</svg>'''
    
    # Save SVG file
    with open(output_svg_path, 'w') as f:
        f.write(svg_content)
    
    total_time = time.time() - start_time
    print(f"Generated smooth green and aquamarine gradient background in {total_time:.2f} seconds")
    print(f"SVG saved to: {output_svg_path}")
    print(f"Bounds saved to: {output_bounds_path}")
    
    return bounds

def main():
    """Main function to generate smooth green and aquamarine gradient background"""
    
    # Output files
    svg_output = "cluster_background.svg"
    bounds_output = "data_bounds.json"
    
    print("Creating beautiful smooth green and aquamarine gradient background...")
    
    try:
        # Generate smooth green and aquamarine gradient background
        bounds = create_smooth_green_aquamarine_gradient(svg_output, bounds_output)
        
        print(f"\nSuccessfully generated smooth green and aquamarine gradient background!")
        print(f"SVG file: {svg_output}")
        print(f"Bounds file: {bounds_output}")
        
    except Exception as e:
        print(f"Error generating smooth green and aquamarine gradient background: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

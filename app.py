"""
Background Removal API - Professional AI Background Removal Service
API similar to remove.bg with support for multiple AI models
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from io import BytesIO
import os
import cv2
import time
import base64
import zipfile
import traceback
from datetime import datetime
import psutil
import logging
from rembg import remove, new_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Environment variables
API_KEY = os.environ.get('API_KEY', None)
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE_MB', 25)) * 1024 * 1024  # Default 25MB
MODEL_CACHE_DIR = os.environ.get('U2NET_HOME', '/app/.u2net')
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create model cache directory if it doesn't exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# In-memory cache for model sessions
_model_sessions = {}

def get_session(model_name='u2net'):
    """Get or create a model session with caching"""
    if model_name not in _model_sessions:
        logger.info(f"Loading model: {model_name}")
        try:
            session = new_session(model_name)
            _model_sessions[model_name] = session
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    return _model_sessions[model_name]

def verify_api_key():
    """Verify API key if set in environment"""
    if API_KEY:
        provided_key = request.headers.get('X-API-Key') or request.form.get('api_key')
        if provided_key != API_KEY:
            return jsonify({
                "error": "Unauthorized",
                "message": "Invalid or missing API key"
            }), 401
    return None

def validate_image_file(file):
    """Validate uploaded image file"""
    if not file or file.filename == '':
        return False, "No image file provided"
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return False, "Invalid file type. Allowed: PNG, JPG, JPEG, WEBP, BMP"
    
    return True, "Valid"

@app.route('/', methods=['GET'])
def home():
    """API Home - Documentation"""
    return jsonify({
        "service": "Background Removal API",
        "version": "2.0.0",
        "description": "Professional AI-powered background removal service similar to remove.bg",
        "status": "operational",
        "authentication": "required" if API_KEY else "not required",
        "limits": {
            "max_file_size": f"{MAX_FILE_SIZE // 1024 // 1024}MB",
            "bulk_max_files": 5,
            "rate_limit": "None by default"
        },
        "endpoints": {
            "/": "GET - API documentation",
            "/health": "GET - Service health check",
            "/remove": "POST - Remove background from single image",
            "/remove/bulk": "POST - Remove background from multiple images",
            "/models": "GET - List available AI models",
            "/stats": "GET - Service statistics",
            "/analyze": "POST - Analyze image for best removal settings"
        },
        "usage_examples": {
            "single_image": "curl -X POST -H 'X-API-Key: YOUR_KEY' -F 'image=@photo.jpg' https://your-api.com/remove",
            "bulk_images": "curl -X POST -H 'X-API-Key: YOUR_KEY' -F 'images[]=@img1.jpg' -F 'images[]=@img2.jpg' https://your-api.com/remove/bulk"
        },
        "github": "https://github.com/yourusername/background-removal-api",
        "documentation": "https://docs.example.com"
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Test model loading
        session = get_session('u2netp')  # Lightweight model for health check
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "background-removal-api",
            "version": "2.0.0",
            "models_loaded": list(_model_sessions.keys())
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }), 500

@app.route('/remove', methods=['POST'])
def remove_background():
    """
    Remove background from a single image
    
    Parameters (form-data):
    - image: file (required)
    - model: 'u2net' | 'u2netp' | 'u2net_human_seg' | 'silueta' | 'isnet-general-use' | 'isnet-anime' 
              (default: 'u2net')
    - format: 'auto' | 'png' | 'jpg' | 'webp' (default: 'auto')
    - quality: 1-100 (default: 95 for lossy formats)
    - size: 'original' | 'preview' | 'medium' | 'small' (default: 'original')
    - bg_color: hex color (e.g., '#FF0000') - replace background with solid color
    - bg_image: file - replace background with another image
    - alpha_matting: 'true' | 'false' (default: 'false')
    - alpha_matting_foreground_threshold: 0-255 (default: 240)
    - alpha_matting_background_threshold: 0-255 (default: 10)
    - post_process_mask: 'true' | 'false' (default: 'true')
    
    Returns: Image with transparent background
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    start_time = time.time()
    
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Get parameters
        model_name = request.form.get('model', 'u2net').lower()
        output_format = request.form.get('format', 'auto').lower()
        quality = int(request.form.get('quality', 95))
        size_option = request.form.get('size', 'original').lower()
        bg_color = request.form.get('bg_color', None)
        bg_image_file = request.files.get('bg_image', None)
        alpha_matting = request.form.get('alpha_matting', 'false').lower() == 'true'
        alpha_matting_foreground = int(request.form.get('alpha_matting_foreground_threshold', 240))
        alpha_matting_background = int(request.form.get('alpha_matting_background_threshold', 10))
        post_process_mask = request.form.get('post_process_mask', 'true').lower() == 'true'
        
        # Validate parameters
        valid_models = ['u2net', 'u2netp', 'u2net_human_seg', 'silueta', 'isnet-general-use', 'isnet-anime']
        if model_name not in valid_models:
            return jsonify({
                "error": f"Invalid model. Choose from: {', '.join(valid_models)}"
            }), 400
        
        if quality < 1 or quality > 100:
            return jsonify({"error": "Quality must be between 1 and 100"}), 400
        
        # Read image
        input_bytes = file.read()
        original_size = len(input_bytes)
        
        if original_size > MAX_FILE_SIZE:
            return jsonify({
                "error": f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB",
                "actual_size": f"{original_size / 1024 / 1024:.2f}MB"
            }), 400
        
        # Get model session
        session = get_session(model_name)
        
        # Process image with background removal
        logger.info(f"Processing image with model: {model_name}")
        
        if alpha_matting:
            output_bytes = remove(
                input_bytes,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=alpha_matting_foreground,
                alpha_matting_background_threshold=alpha_matting_background,
                alpha_matting_erode_size=10,
                post_process_mask=post_process_mask
            )
        else:
            output_bytes = remove(
                input_bytes,
                session=session,
                post_process_mask=post_process_mask
            )
        
        # Handle background replacement
        if bg_color or bg_image_file:
            from PIL import Image, ImageOps
            import io
            
            # Open result image
            result_img = Image.open(io.BytesIO(output_bytes)).convert('RGBA')
            
            # Prepare background
            if bg_color:
                # Parse hex color
                bg_color = bg_color.lstrip('#')
                if len(bg_color) == 6:
                    bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
                    bg_img = Image.new('RGBA', result_img.size, (*bg_rgb, 255))
                else:
                    return jsonify({"error": "Invalid color format. Use hex like #FF0000"}), 400
            else:
                # Use uploaded background image
                bg_img = Image.open(bg_image_file.stream).convert('RGBA')
                bg_img = ImageOps.fit(bg_img, result_img.size, method=Image.Resampling.LANCZOS)
            
            # Composite images
            final_img = Image.alpha_composite(bg_img, result_img)
            output_buffer = io.BytesIO()
            final_img.save(output_buffer, format='PNG')
            output_bytes = output_buffer.getvalue()
        
        # Determine output format
        if output_format == 'auto':
            # Auto-detect based on transparency needs
            if bg_color or bg_image_file:
                output_format = 'jpg'  # No transparency needed
            else:
                output_format = 'png'  # Keep transparency
        
        # Convert format if needed
        if output_format in ['jpg', 'jpeg', 'webp']:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(output_bytes))
            
            if output_format in ['jpg', 'jpeg']:
                img = img.convert('RGB')  # Remove alpha for JPEG
                output_buffer = io.BytesIO()
                img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
                mime_type = 'image/jpeg'
                file_ext = 'jpg'
            elif output_format == 'webp':
                output_buffer = io.BytesIO()
                img.save(output_buffer, format='WEBP', quality=quality, method=6)
                mime_type = 'image/webp'
                file_ext = 'webp'
            
            output_bytes = output_buffer.getvalue()
        else:
            # Default to PNG
            mime_type = 'image/png'
            file_ext = 'png'
        
        # Handle resizing
        if size_option != 'original':
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(output_bytes))
            
            # Define size presets
            size_presets = {
                'small': 512,
                'medium': 1024,
                'preview': 2048
            }
            
            if size_option in size_presets:
                max_dimension = size_presets[size_option]
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    output_buffer = io.BytesIO()
                    if output_format in ['jpg', 'jpeg']:
                        img.save(output_buffer, format='JPEG', quality=quality)
                    elif output_format == 'webp':
                        img.save(output_buffer, format='WEBP', quality=quality)
                    else:
                        img.save(output_buffer, format='PNG', optimize=True)
                    
                    output_bytes = output_buffer.getvalue()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        output = BytesIO(output_bytes)
        output.seek(0)
        
        # Generate filename
        original_name = file.filename.rsplit('.', 1)[0]
        safe_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name[:50]  # Limit length
        download_name = f"{safe_name}_nobg.{file_ext}"
        
        response = send_file(
            output,
            mimetype=mime_type,
            as_attachment=True,
            download_name=download_name
        )
        
        # Add informative headers
        response.headers['X-Processing-Time'] = f"{processing_time:.3f}s"
        response.headers['X-Model-Used'] = model_name
        response.headers['X-Original-Size'] = str(original_size)
        response.headers['X-Output-Size'] = str(len(output_bytes))
        response.headers['X-Compression-Ratio'] = f"{len(output_bytes) / max(1, original_size):.2f}"
        response.headers['X-Format'] = output_format
        
        logger.info(f"Successfully processed image in {processing_time:.3f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Background removal failed: {str(e)}\n{traceback.format_exc()}")
        
        return jsonify({
            "error": "Background removal failed",
            "message": str(e),
            "tip": "Try using a different model or check if the image format is supported"
        }), 500

@app.route('/remove/bulk', methods=['POST'])
def remove_background_bulk():
    """
    Remove background from multiple images (max 5)
    
    Parameters:
    - images[]: multiple files (required, max 5)
    - model: model name (default: 'u2net')
    - format: 'png' | 'jpg' | 'webp' (default: 'png')
    - as_zip: 'true' | 'false' (default: 'true')
    
    Returns: ZIP file containing processed images
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    start_time = time.time()
    
    try:
        if 'images[]' not in request.files:
            return jsonify({"error": "No images provided. Use 'images[]' parameter."}), 400
        
        files = request.files.getlist('images[]')
        if len(files) > 5:
            return jsonify({"error": "Maximum 5 images allowed for bulk processing"}), 400
        
        if len(files) == 0:
            return jsonify({"error": "No valid images provided"}), 400
        
        # Get parameters
        model_name = request.form.get('model', 'u2net')
        output_format = request.form.get('format', 'png').lower()
        as_zip = request.form.get('as_zip', 'true').lower() == 'true'
        quality = int(request.form.get('quality', 95))
        
        # Validate format
        if output_format not in ['png', 'jpg', 'jpeg', 'webp']:
            return jsonify({"error": "Invalid format. Use 'png', 'jpg', or 'webp'"}), 400
        
        # Get model session
        session = get_session(model_name)
        
        # Process images
        processed_images = []
        failed_images = []
        
        for idx, file in enumerate(files):
            try:
                # Validate file
                is_valid, message = validate_image_file(file)
                if not is_valid:
                    failed_images.append({
                        'filename': file.filename,
                        'error': message
                    })
                    continue
                
                # Read file
                input_bytes = file.read()
                
                # Check size
                if len(input_bytes) > MAX_FILE_SIZE:
                    failed_images.append({
                        'filename': file.filename,
                        'error': f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)"
                    })
                    continue
                
                # Process image
                output_bytes = remove(input_bytes, session=session, post_process_mask=True)
                
                # Convert format if needed
                if output_format in ['jpg', 'jpeg', 'webp']:
                    from PIL import Image
                    import io
                    
                    img = Image.open(io.BytesIO(output_bytes))
                    
                    if output_format in ['jpg', 'jpeg']:
                        img = img.convert('RGB')
                        output_buffer = io.BytesIO()
                        img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
                        file_ext = 'jpg'
                    elif output_format == 'webp':
                        output_buffer = io.BytesIO()
                        img.save(output_buffer, format='WEBP', quality=quality, method=6)
                        file_ext = 'webp'
                    
                    output_bytes = output_buffer.getvalue()
                else:
                    file_ext = 'png'
                
                # Create safe filename
                original_name = file.filename.rsplit('.', 1)[0]
                safe_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_name = safe_name[:50]
                
                processed_images.append({
                    'filename': f"{safe_name}_nobg.{file_ext}",
                    'data': output_bytes,
                    'size': len(output_bytes),
                    'original_name': file.filename,
                    'success': True
                })
                
                logger.info(f"Processed {file.filename} successfully")
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {str(e)}")
                failed_images.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        # Prepare response
        if as_zip and processed_images:
            # Create ZIP file
            import io
            from datetime import datetime
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for img in processed_images:
                    zip_file.writestr(img['filename'], img['data'])
            
            zip_buffer.seek(0)
            
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            response = send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'background_removed_{timestamp}.zip'
            )
            
            # Add headers
            response.headers['X-Total-Images'] = str(len(files))
            response.headers['X-Successful'] = str(len(processed_images))
            response.headers['X-Failed'] = str(len(failed_images))
            response.headers['X-Processing-Time'] = f"{processing_time:.3f}s"
            response.headers['X-Model-Used'] = model_name
            
            return response
        else:
            # Return JSON response
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "processing_time": f"{processing_time:.3f}s",
                "total_images": len(files),
                "successful": len(processed_images),
                "failed": len(failed_images),
                "model_used": model_name,
                "results": []
            }
            
            # Add base64 encoded images
            for img in processed_images:
                result['results'].append({
                    'filename': img['filename'],
                    'original_name': img['original_name'],
                    'size': img['size'],
                    'data_url': f"data:image/{output_format};base64,{base64.b64encode(img['data']).decode('utf-8')}"
                })
            
            # Add failed images info
            if failed_images:
                result['failed_details'] = failed_images
            
            return jsonify(result)
        
    except Exception as e:
        logger.error(f"Bulk processing failed: {str(e)}\n{traceback.format_exc()}")
        
        return jsonify({
            "error": "Bulk processing failed",
            "message": str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """
    List all available AI models with details
    """
    models = [
        {
            "name": "u2net",
            "description": "General purpose model (highest quality)",
            "size": "176MB",
            "recommended_for": ["General images", "Products", "Animals", "Objects", "Landscapes"],
            "not_recommended_for": ["Speed-critical applications"],
            "speed": "Medium",
            "accuracy": "Excellent",
            "default": True
        },
        {
            "name": "u2netp",
            "description": "Lightweight version of u2net",
            "size": "5.7MB",
            "recommended_for": ["Fast processing", "Mobile applications", "Low memory environments"],
            "not_recommended_for": ["Very complex images with fine details"],
            "speed": "Fast",
            "accuracy": "Good"
        },
        {
            "name": "u2net_human_seg",
            "description": "Specialized for human segmentation",
            "size": "176MB",
            "recommended_for": ["Portraits", "People", "Fashion", "Group photos", "Selfies"],
            "not_recommended_for": ["Non-human subjects"],
            "speed": "Medium",
            "accuracy": "Excellent for humans"
        },
        {
            "name": "silueta",
            "description": "Very lightweight silhouette model",
            "size": "4.7MB",
            "recommended_for": ["Simple shapes", "Silhouettes", "Extreme speed requirements"],
            "not_recommended_for": ["Complex backgrounds", "Fine details"],
            "speed": "Very Fast",
            "accuracy": "Basic"
        },
        {
            "name": "isnet-general-use",
            "description": "General purpose model (alternative to u2net)",
            "size": "103MB",
            "recommended_for": ["General images", "Alternative when u2net fails"],
            "speed": "Medium",
            "accuracy": "Very Good"
        },
        {
            "name": "isnet-anime",
            "description": "Optimized for anime and cartoon images",
            "size": "103MB",
            "recommended_for": ["Anime", "Cartoons", "Illustrations", "Digital art"],
            "not_recommended_for": ["Real photographs"],
            "speed": "Medium",
            "accuracy": "Excellent for anime"
        }
    ]
    
    return jsonify({
        "models": models,
        "recommendations": {
            "general_use": "u2net",
            "people_portraits": "u2net_human_seg",
            "fast_processing": "u2netp",
            "anime_cartoons": "isnet-anime",
            "mobile_apps": "u2netp or silueta"
        },
        "notes": "Models are downloaded on first use and cached for faster subsequent requests"
    })

@app.route('/stats', methods=['GET'])
def service_stats():
    """
    Get service statistics and system information
    """
    try:
        # System information
        import platform
        
        # Memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Disk usage for model cache
        model_dir = MODEL_CACHE_DIR
        model_files = []
        total_model_size = 0
        
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.onnx'):
                    file_path = os.path.join(model_dir, f)
                    size = os.path.getsize(file_path)
                    model_files.append({
                        'name': f,
                        'size_mb': size / 1024 / 1024
                    })
                    total_model_size += size
        
        # Uptime (simplified)
        try:
            uptime_seconds = time.time() - psutil.boot_time()
        except:
            uptime_seconds = 0
        
        return jsonify({
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": {
                "name": "Background Removal API",
                "version": "2.0.0",
                "uptime_seconds": int(uptime_seconds),
                "models_loaded": list(_model_sessions.keys()),
                "active_sessions": len(_model_sessions)
            },
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "memory_used_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads()
            },
            "models": {
                "cache_directory": model_dir,
                "files": model_files,
                "total_size_mb": total_model_size / 1024 / 1024,
                "count": len(model_files)
            },
            "limits": {
                "max_file_size_mb": MAX_FILE_SIZE // 1024 // 1024,
                "max_bulk_files": 5,
                "supported_formats": ["PNG", "JPG", "JPEG", "WEBP", "BMP"]
            }
        })
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve statistics",
            "message": str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze an image and recommend best settings for background removal
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        from PIL import Image
        import io
        
        # Open and analyze image
        img = Image.open(file.stream)
        
        # Basic image info
        width, height = img.size
        aspect_ratio = width / height
        total_pixels = width * height
        mode = img.mode
        
        # Check if image has transparency
        has_transparency = mode == 'RGBA' and any(
            pixel[3] < 255 for pixel in img.getdata()
        )
        
        # Analyze colors (simple histogram)
        if mode != 'RGB':
            rgb_img = img.convert('RGB')
        else:
            rgb_img = img
        
        # Convert to numpy for analysis
        img_array = np.array(rgb_img)
        
        # Calculate average brightness
        brightness = np.mean(img_array) / 255.0
        
        # Calculate color variance
        color_variance = np.var(img_array) / 255.0
        
        # Analyze for human faces (simple heuristic)
        # This is a very basic check - in production you'd use face detection
        is_likely_portrait = False
        if aspect_ratio > 0.6 and aspect_ratio < 1.4:  # Square-ish
            if width > 300 and height > 300:  # Not too small
                # Check if center region has skin-like colors
                center_h = height // 4
                center_w = width // 4
                center_region = img_array[center_h:3*center_h, center_w:3*center_w]
                
                # Simple skin tone detection (RGB ranges)
                r, g, b = center_region[:,:,0], center_region[:,:,1], center_region[:,:,2]
                skin_mask = (r > 100) & (g > 70) & (b > 50) & (r > g) & (r > b)
                skin_ratio = np.sum(skin_mask) / skin_mask.size if skin_mask.size > 0 else 0
                
                if skin_ratio > 0.1:
                    is_likely_portrait = True
        
        # Check if image is simple/complex
        is_complex_image = color_variance > 0.05
        
        # Recommend model
        if is_likely_portrait:
            recommended_model = 'u2net_human_seg'
            model_reason = "Image appears to contain people/portraits"
        elif total_pixels > 2000000:  # > 2MP
            recommended_model = 'u2netp'
            model_reason = "Large image - using lightweight model for speed"
        elif not is_complex_image:
            recommended_model = 'u2netp'
            model_reason = "Simple image - lightweight model sufficient"
        else:
            recommended_model = 'u2net'
            model_reason = "Complex image - using high-quality model"
        
        # Recommend alpha matting
        recommend_alpha_matting = (has_transparency or 
                                   is_likely_portrait or 
                                   (brightness > 0.7 and is_complex_image))
        
        # Recommend format
        if has_transparency or recommend_alpha_matting:
            recommended_format = 'png'
            format_reason = "Preserve transparency"
        else:
            recommended_format = 'jpg'
            format_reason = "Smaller file size, transparency not needed"
        
        return jsonify({
            "analysis": {
                "dimensions": f"{width}x{height}",
                "aspect_ratio": round(aspect_ratio, 2),
                "total_pixels": total_pixels,
                "megapixels": round(total_pixels / 1000000, 2),
                "color_mode": mode,
                "has_transparency": has_transparency,
                "average_brightness": round(brightness, 3),
                "color_complexity": round(color_variance, 3),
                "likely_contains_people": is_likely_portrait,
                "image_complexity": "complex" if is_complex_image else "simple"
            },
            "recommendations": {
                "model": {
                    "name": recommended_model,
                    "reason": model_reason
                },
                "alpha_matting": {
                    "recommended": recommend_alpha_matting,
                    "reason": "Better for hair, fur, and complex edges" if recommend_alpha_matting else "Not needed for this image"
                },
                "format": {
                    "name": recommended_format,
                    "reason": format_reason
                },
                "size_preset": "original"  # Always recommend original size for analysis
            },
            "estimated_processing": {
                "time_seconds": "2-5" if is_complex_image else "1-3",
                "output_size_ratio": "0.7-1.2"  # Compared to original
            }
        })
        
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        return jsonify({
            "error": "Image analysis failed",
            "message": str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        "error": "File too large",
        "message": f"Maximum file size is {MAX_FILE_SIZE // 1024 // 1024}MB"
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "Check the API documentation at /"
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on our end. Please try again."
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Background Removal API on {host}:{port}")
    logger.info(f"Model cache directory: {MODEL_CACHE_DIR}")
    logger.info(f"Max file size: {MAX_FILE_SIZE // 1024 // 1024}MB")
    logger.info(f"API Key required: {API_KEY is not None}")
    
    # Pre-load default model for faster first request
    try:
        logger.info("Pre-loading default model (u2net)...")
        get_session('u2net')
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-load default model: {str(e)}")
    
    app.run(host=host, port=port, debug=DEBUG)
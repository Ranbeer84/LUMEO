"""
Object Service - Object Detection, Color Extraction, and Scene Classification
Phase 2.4 & 2.5
"""

from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectService:
    """Handle object detection and scene classification"""
    
    # Scene classification rules based on detected objects
    OUTDOOR_INDICATORS = ['car', 'tree', 'bench', 'bicycle', 'motorcycle', 'airplane', 
                          'bird', 'horse', 'dog', 'cat', 'truck', 'boat', 'traffic light']
    
    INDOOR_INDICATORS = ['chair', 'couch', 'tv', 'laptop', 'keyboard', 'mouse', 
                        'book', 'clock', 'vase', 'bed', 'dining table', 'toilet', 
                        'sink', 'refrigerator', 'microwave', 'oven']
    YOLO_LABEL_MAP = {
                    'bicycle':      'bicycle',    
                    'motorbike':    'motorcycle',
                    'aeroplane':    'airplane',
                    'sofa':         'couch',
                    'pottedplant':  'plant',
                    'tvmonitor':    'tv',
                    'diningtable':  'dining table',
                    'cell phone':   'cell phone',
                    'hot dog':      'hot dog',
                }
    
    BEACH_INDICATORS = ['umbrella', 'surfboard', 'boat']
    SPORTS_INDICATORS = ['sports ball', 'baseball bat', 'tennis racket', 'skateboard', 
                        'skis', 'snowboard', 'frisbee']
    FOOD_INDICATORS = ['bowl', 'cup', 'fork', 'knife', 'spoon', 'wine glass', 'cake', 
                      'pizza', 'donut', 'hot dog', 'sandwich']
    PARTY_INDICATORS = ['cake', 'wine glass', 'cup', 'donut']
    WORK_INDICATORS = ['laptop', 'keyboard', 'mouse', 'book']
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLO model
        
        Args:
            model_path: Path to YOLO model (yolov8n.pt for nano/fast)
        """
        try:
            logger.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model = None
    
    def detect_objects(self, image_path, conf_threshold=0.5):
        """
        Detect objects in an image
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold (0-1)
        
        Returns:
            list: List of detected objects with bounding boxes and colors
        """
        if self.model is None:
            logger.error("YOLO model not loaded")
            return []
        
        try:
            logger.info(f"Detecting objects in: {image_path}")
            
            # Run YOLO inference
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            
            # Load original image for color extraction
            image = cv2.imread(str(image_path))
            
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Extract dominant color from detected region
                    dominant_color = self._extract_dominant_color(image, (x1, y1, x2, y2))
                    color_name = self._color_to_name(dominant_color)

                    normalized_label = self.normalize_label(class_name)
                    
                    obj_data = {
                        'label': normalized_label,
                        'confidence': round(confidence, 3),
                        'bbox': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        },
                        'dominant_color_rgb': dominant_color,
                        'color_name': color_name
                    }
                    
                    detected_objects.append(obj_data)
            
            logger.info(f"Detected {len(detected_objects)} objects")
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return []
    
    def _extract_dominant_color(self, image, bbox):
        """
        Extract dominant color from a bounding box region
        
        Args:
            image: OpenCV image array
            bbox: Tuple (x1, y1, x2, y2)
        
        Returns:
            tuple: (r, g, b) dominant color
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Extract region
            region = image[y1:y2, x1:x2]
            
            # Reshape to 2D array of pixels
            pixels = region.reshape(-1, 3)
            
            # Calculate mean color
            mean_color = np.mean(pixels, axis=0).astype(int)
            
            # Convert BGR to RGB
            r, g, b = mean_color[2], mean_color[1], mean_color[0]
            
            return (int(r), int(g), int(b))
            
        except Exception as e:
            logger.error(f"Error extracting color: {str(e)}")
            return (128, 128, 128)  # Default gray
    
    def _color_to_name(self, rgb):
        """
        Convert RGB to basic color name
        
        Args:
            rgb: Tuple (r, g, b)
        
        Returns:
            str: Color name
        """
        r, g, b = rgb
        
        # Simple color classification
        if r > 200 and g > 200 and b > 200:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > g and r > b:
            if r > 180:
                return 'red'
            else:
                return 'brown'
        elif g > r and g > b:
            return 'green'
        elif b > r and b > g:
            return 'blue'
        elif r > 150 and g > 150 and b < 100:
            return 'yellow'
        elif r > 150 and g < 100 and b > 150:
            return 'purple'
        elif r > 150 and g > 100 and b < 100:
            return 'orange'
        else:
            return 'gray'
        
    def normalize_label(self, label: str) -> str:
        """
        Normalize YOLO/COCO label to consistent stored label
        """
        return self.YOLO_LABEL_MAP.get(label.lower(), label.lower())

    def classify_scene(self, detected_objects):
        """
        Classify scene type based on detected objects
        
        Args:
            detected_objects: List of object detections
        
        Returns:
            dict: {
                'scene_type': str (indoor/outdoor),
                'location': str,
                'activity': str,
                'confidence': float
            }
        """
        if not detected_objects:
            return {
                'scene_type': 'unknown',
                'location': 'unknown',
                'activity': 'unknown',
                'confidence': 0.0
            }
        
        # Extract object labels
        labels = [obj['label'] for obj in detected_objects]
        label_counts = Counter(labels)
        
        # Determine indoor vs outdoor
        outdoor_score = sum(1 for label in labels if label in self.OUTDOOR_INDICATORS)
        indoor_score = sum(1 for label in labels if label in self.INDOOR_INDICATORS)
        
        if outdoor_score > indoor_score:
            scene_type = 'outdoor'
        elif indoor_score > outdoor_score:
            scene_type = 'indoor'
        else:
            scene_type = 'unknown'
        
        # Determine specific location
        location = 'general'
        if any(label in self.BEACH_INDICATORS for label in labels):
            location = 'beach'
        elif 'car' in labels or 'truck' in labels:
            location = 'road/parking'
        elif 'dining table' in labels or any(label in self.FOOD_INDICATORS for label in labels):
            location = 'dining'
        elif 'bed' in labels or 'couch' in labels:
            location = 'home'
        elif 'laptop' in labels and 'chair' in labels:
            location = 'office'
        
        # Determine activity
        activity = 'general'
        if any(label in self.SPORTS_INDICATORS for label in labels):
            activity = 'sports'
        elif sum(1 for label in labels if label in self.FOOD_INDICATORS) >= 2:
            activity = 'dining'
        elif any(label in self.PARTY_INDICATORS for label in labels):
            activity = 'celebration'
        elif any(label in self.WORK_INDICATORS for label in labels):
            activity = 'working'
        
        # Calculate confidence based on number of relevant objects
        confidence = min(len(detected_objects) / 10.0, 1.0)
        
        scene_data = {
            'scene_type': scene_type,
            'location': location,
            'activity': activity,
            'confidence': round(confidence, 3)
        }
        
        logger.info(f"Scene classified: {scene_type} - {location} - {activity}")
        
        return scene_data
    
    def detect_scene_and_weather(self, image_path: str, detected_objects: list) -> dict:
        """
        Infer scene type and weather from detected objects + image analysis.
        Called after detect_objects() in the pipeline.
        """
        labels = [obj['label'].lower() for obj in detected_objects]
        label_set = set(labels)

        # --- Scene detection from object co-occurrence ---
        scene_rules = {
            'beach':      {'surfboard', 'umbrella', 'frisbee', 'kite', 'boat'},
            'kitchen':    {'oven', 'microwave', 'refrigerator', 'sink', 'toaster', 'bowl'},
            'dining':     {'dining table', 'fork', 'knife', 'spoon', 'cup', 'wine glass', 'bottle'},
            'office':     {'laptop', 'keyboard', 'mouse', 'monitor', 'chair', 'book'},
            'sports':     {'ball', 'sports ball', 'tennis racket', 'baseball bat', 'skateboard', 'skis'},
            'party':      {'cake', 'wine glass', 'bottle', 'cup', 'balloon'},
            'road/street':{'car', 'truck', 'bus', 'traffic light', 'stop sign', 'bicycle', 'motorcycle'},
            'nature':     {'bird', 'cow', 'horse', 'sheep', 'elephant', 'bear', 'tree'},
            'living room':{'couch', 'tv', 'remote', 'potted plant', 'clock'},
            'bedroom':    {'bed', 'pillow'},
            'bathroom':   {'toilet', 'sink'},
        }

        detected_scene = 'general'
        best_score = 0
        for scene, keywords in scene_rules.items():
            score = len(keywords & label_set)
            if score > best_score:
                best_score = score
                detected_scene = scene

        # --- Weather detection from image color/brightness ---
        weather = self._detect_weather(image_path)

        return {
            'scene_label': detected_scene,
            'weather': weather,
        }
    
    def _detect_weather(self, image_path: str) -> str:
        """
        Simple weather inference from image color statistics.
        Returns: 'rainy', 'snowy', 'sunny', 'cloudy', or 'unknown'
        """
        try:
            import cv2
            import numpy as np

            img = cv2.imread(str(image_path))
            if img is None:
                return 'unknown'
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()
            value = hsv[:, :, 2].mean()          # brightness

            # Convert to grayscale for variance
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            std_dev = gray.std()

            # Check sky region (top 30% of image)
            h = img.shape[0]
            sky_region = img[:int(h * 0.3), :]
            sky_hsv = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
            sky_sat = sky_hsv[:, :, 1].mean()
            sky_val = sky_hsv[:, :, 2].mean()

            # Snowy: very bright, low saturation overall
            if value > 200 and saturation < 30:
                return 'snowy'
            
            # Rainy/overcast: low brightness, low saturation, low contrast
            if value < 110 and saturation < 50 and std_dev < 45:
                return 'rainy'
            
            # Sunny: high brightness + high saturation sky
            if sky_val > 160 and sky_sat > 80:
                return 'sunny'
            
            # Cloudy: bright but desaturated sky
            if sky_val > 140 and sky_sat < 60:
                return 'cloudy'
            
            return 'unknown'

        except Exception as e:
            logger.error(f"Weather detection failed: {e}")
            return 'unknown'
    
    def get_clothing_colors(self, detected_objects):
        """
        Extract clothing colors from person detections
        
        Args:
            detected_objects: List of object detections
        
        Returns:
            list: List of color names found in clothing
        """
        colors = []
        
        for obj in detected_objects:
            if obj['label'] == 'person':
                color_name = obj.get('color_name')
                if color_name:
                    colors.append(color_name)
        
        return list(set(colors))  # Unique colors


# Singleton instance
_object_service = None

def get_object_service():
    """Get or create object service singleton"""
    global _object_service
    if _object_service is None:
        _object_service = ObjectService()
    return _object_service
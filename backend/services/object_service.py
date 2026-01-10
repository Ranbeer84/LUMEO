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
                    
                    obj_data = {
                        'label': class_name,
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
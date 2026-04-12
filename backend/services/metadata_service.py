"""
Metadata Service - EXIF Extraction and Temporal Context
Phase 2.7
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataService:
    """Handle metadata extraction and enrichment"""
    
    def __init__(self):
        pass
    
    def extract_exif(self, image_path):
        """
        Extract EXIF metadata from image
        
        Args:
            image_path: Path to image file
        
        Returns:
            dict: EXIF metadata including date, camera info, GPS, etc.
        """
        try:
            logger.info(f"Extracting EXIF from: {image_path}")
            
            image = Image.open(image_path)
            exif_data = {}
            
            # Get EXIF data
            exif = image._getexif()
            
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
            
            # Parse important fields
            metadata = {
                'date_taken': self._parse_datetime(exif_data.get('DateTimeOriginal') or exif_data.get('DateTime')),
                'camera_make': exif_data.get('Make'),
                'camera_model': exif_data.get('Model'),
                'width': image.width,
                'height': image.height,
                'orientation': exif_data.get('Orientation', 1),
                'flash': exif_data.get('Flash'),
                'focal_length': exif_data.get('FocalLength'),
                'iso': exif_data.get('ISOSpeedRatings'),
                'exposure_time': exif_data.get('ExposureTime'),
                'f_number': exif_data.get('FNumber'),
                'gps': self._parse_gps(exif_data.get('GPSInfo'))
            }
            
            # Add temporal context
            if metadata['date_taken']:
                metadata.update(self._get_temporal_context(metadata['date_taken']))
            
            # Add image quality metrics
            metadata.update(self._calculate_image_quality(image_path))
            
            logger.info(f"EXIF extracted successfully")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting EXIF: {str(e)}")
            # Return minimal metadata
            return self._get_minimal_metadata(image_path)
    
    def _parse_datetime(self, datetime_str):
        """
        Parse EXIF datetime string to datetime object
        
        Args:
            datetime_str: EXIF datetime string (e.g., '2023:12:25 14:30:45')
        
        Returns:
            datetime: Parsed datetime object, or None
        """
        if not datetime_str:
            return None
        
        try:
            # EXIF datetime format: YYYY:MM:DD HH:MM:SS
            return datetime.strptime(str(datetime_str), '%Y:%m:%d %H:%M:%S')
        except:
            try:
                # Try alternative format
                return datetime.strptime(str(datetime_str), '%Y-%m-%d %H:%M:%S')
            except:
                logger.warning(f"Could not parse datetime: {datetime_str}")
                return None
    
    def _parse_gps(self, gps_info):
        """
        Parse GPS information from EXIF
        
        Args:
            gps_info: GPS info dict from EXIF
        
        Returns:
            dict: {'latitude': float, 'longitude': float} or None
        """
        if not gps_info:
            return None
        
        try:
            gps_data = {}
            for tag_id, value in gps_info.items():
                tag = GPSTAGS.get(tag_id, tag_id)
                gps_data[tag] = value
            
            # Extract coordinates
            lat = self._convert_to_degrees(gps_data.get('GPSLatitude'))
            lon = self._convert_to_degrees(gps_data.get('GPSLongitude'))
            
            # Apply reference (N/S, E/W)
            if gps_data.get('GPSLatitudeRef') == 'S':
                lat = -lat
            if gps_data.get('GPSLongitudeRef') == 'W':
                lon = -lon
            
            if lat and lon:
                return {'latitude': lat, 'longitude': lon}
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing GPS: {str(e)}")
            return None
    
    def _convert_to_degrees(self, value):
        """Convert GPS coordinates to degrees"""
        if not value:
            return None
        
        d, m, s = value
        return float(d) + float(m) / 60.0 + float(s) / 3600.0
    
    def _get_temporal_context(self, date_taken):
        """
        Infer temporal context from date
        
        Args:
            date_taken: datetime object
        
        Returns:
            dict: {'season': str, 'time_of_day': str, 'year': int, 'month': int, 'day': int}
        """
        if not date_taken:
            return {
                'season': 'unknown',
                'time_of_day': 'unknown',
                'year': None,
                'month': None,
                'day': None
            }
        
        # Determine season (Northern Hemisphere)
        month = date_taken.month
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        else:
            season = 'autumn'
        
        # Determine time of day
        hour = date_taken.hour
        if 5 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= hour < 21:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
        
        return {
            'season': season,
            'time_of_day': time_of_day,
            'year': date_taken.year,
            'month': date_taken.month,
            'day': date_taken.day,
            'weekday': date_taken.strftime('%A')
        }
    
    def _calculate_image_quality(self, image_path):
        """
        Calculate image quality metrics
        
        Args:
            image_path: Path to image
        
        Returns:
            dict: Quality metrics (blur, brightness, contrast, etc.)
        """
        try:
            image = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 500, 1.0)
            
            # 2. Brightness
            brightness = np.mean(gray) / 255.0
            
            # 3. Contrast (standard deviation)
            contrast = np.std(gray) / 128.0
            
            # 4. Overall quality score
            quality_score = (sharpness * 0.4 + 
                           (1 - abs(brightness - 0.5) * 2) * 0.3 + 
                           min(contrast, 1.0) * 0.3)
            
            return {
                'sharpness': round(sharpness, 3),
                'brightness': round(brightness, 3),
                'contrast': round(contrast, 3),
                'quality_score': round(quality_score, 3)
            }
            
        except Exception as e:
            logger.error(f"Error calculating image quality: {str(e)}")
            return {
                'sharpness': 0.5,
                'brightness': 0.5,
                'contrast': 0.5,
                'quality_score': 0.5
            }
    
    def _get_minimal_metadata(self, image_path):
        """
        Get minimal metadata when EXIF extraction fails
        
        Args:
            image_path: Path to image
        
        Returns:
            dict: Basic metadata
        """
        try:
            image = Image.open(image_path)
            return {
                'date_taken': None,
                'width': image.width,
                'height': image.height,
                'season': 'unknown',
                'time_of_day': 'unknown',
                'quality_score': 0.5
            }
        except:
            return {
                'date_taken': None,
                'season': 'unknown',
                'time_of_day': 'unknown',
                'quality_score': 0.5
            }
    
    def generate_caption(self, metadata, detected_objects=None, detected_faces=None, emotion=None):
        """
        Generate a natural language caption from metadata
        
        Args:
            metadata: Metadata dict
            detected_objects: List of detected objects
            detected_faces: Number of faces detected
            emotion: Dominant emotion
        
        Returns:
            str: Natural language caption
        """
        caption_parts = []
        
        # Date and time
        if metadata.get('date_taken'):
            date_str = metadata['date_taken'].strftime('%B %d, %Y')
            caption_parts.append(f"Photo taken on {date_str}")
            
            if metadata.get('time_of_day'):
                caption_parts.append(f"during {metadata['time_of_day']}")
        
        # Season
        # if metadata.get('season') != 'unknown': # <--- TRAP HERE
        #     caption_parts.append(f"in {metadata['season']}") # <--- CRASH HERE

        # Season
        # Check if metadata exists, AND if 'season' is a key, AND if it has a value
        if metadata and 'season' in metadata and metadata['season'] and metadata['season'] != 'unknown':
            caption_parts.append(f"in {metadata['season']}")
        
        # Scene
        if detected_objects:
            objects_str = ', '.join([obj['label'] for obj in detected_objects[:3]])
            caption_parts.append(f"showing {objects_str}")
        
        # People
        if detected_faces and detected_faces > 0:
            caption_parts.append(f"with {detected_faces} {'person' if detected_faces == 1 else 'people'}")
        
        # Emotion
        if emotion and emotion != 'neutral':
            caption_parts.append(f"expressing {emotion}")
        
        caption = '. '.join(caption_parts) + '.'
        
        return caption


# Singleton instance
_metadata_service = None

def get_metadata_service():
    """Get or create metadata service singleton"""
    global _metadata_service
    if _metadata_service is None:
        _metadata_service = MetadataService()
    return _metadata_service
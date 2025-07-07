import cv2
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ImprovedFaceRecognitionSystem:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '')
        
        # Data storage
        self.face_encodings = []
        self.known_names = []
        self.face_id_counter = 0
        self.training_data_path = "face_training_data"
        self.encodings_path = "face_encodings.pkl"
        self.names_path = "face_names.json"
        self.scaler_path = "feature_scaler.pkl"
        
        # Recognition parameters
        self.recognition_threshold = 0.75  # Higher threshold for better accuracy
        self.feature_scaler = StandardScaler()
        
        # Face preprocessing parameters
        self.face_size = (128, 128)  # Larger face size for better features
        
        # Create directories
        os.makedirs(self.training_data_path, exist_ok=True)
        
        # Load existing data
        self.load_training_data()
        
    def load_training_data(self):
        """Load existing training data"""
        try:
            if os.path.exists(self.encodings_path):
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_encodings = data.get('encodings', [])
                    if 'scaler' in data:
                        self.feature_scaler = data['scaler']
                print("Loaded existing face encodings")
            
            if os.path.exists(self.names_path):
                with open(self.names_path, 'r') as f:
                    name_data = json.load(f)
                    self.known_names = name_data.get('names', [])
                    self.face_id_counter = name_data.get('counter', 0)
                print(f"Loaded {len(self.known_names)} known faces")
                
        except Exception as e:
            print(f"Error loading training data: {e}")
    
    def save_training_data(self):
        """Save the current encodings and names"""
        try:
            data = {
                'encodings': self.face_encodings,
                'scaler': self.feature_scaler
            }
            with open(self.encodings_path, 'wb') as f:
                pickle.dump(data, f)
            
            name_data = {
                'names': self.known_names,
                'counter': self.face_id_counter
            }
            with open(self.names_path, 'w') as f:
                json.dump(name_data, f)
            
            print("Training data saved successfully")
        except Exception as e:
            print(f"Error saving training data: {e}")
    
    def preprocess_face(self, face_roi):
        """Advanced face preprocessing"""
        # Resize to standard size
        face = cv2.resize(face_roi, self.face_size)
        
        # Apply histogram equalization for better contrast
        face = cv2.equalizeHist(face)
        
        # Apply Gaussian blur to reduce noise
        face = cv2.GaussianBlur(face, (3, 3), 0)
        
        # Normalize pixel values
        face = face.astype(np.float32) / 255.0
        
        return face
    
    def extract_advanced_features(self, face_roi):
        """Extract comprehensive features from face"""
        # Preprocess face
        face = self.preprocess_face(face_roi)
        
        features = []
        
        # 1. Local Binary Pattern (LBP) features
        lbp_features = self.extract_lbp_features(face)
        features.extend(lbp_features)
        
        # 2. Histogram of Oriented Gradients (HOG) features
        hog_features = self.extract_hog_features(face)
        features.extend(hog_features)
        
        # 3. Gabor filter responses
        gabor_features = self.extract_gabor_features(face)
        features.extend(gabor_features)
        
        # 4. Statistical features
        stat_features = self.extract_statistical_features(face)
        features.extend(stat_features)
        
        # 5. Geometric features
        geometric_features = self.extract_geometric_features(face_roi)
        features.extend(geometric_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_lbp_features(self, face):
        """Extract Local Binary Pattern features"""
        # Convert to uint8 for LBP
        face_uint8 = (face * 255).astype(np.uint8)
        
        # Calculate LBP
        lbp = np.zeros_like(face_uint8)
        for i in range(1, face_uint8.shape[0] - 1):
            for j in range(1, face_uint8.shape[1] - 1):
                center = face_uint8[i, j]
                code = 0
                code |= (face_uint8[i-1, j-1] >= center) << 7
                code |= (face_uint8[i-1, j] >= center) << 6
                code |= (face_uint8[i-1, j+1] >= center) << 5
                code |= (face_uint8[i, j+1] >= center) << 4
                code |= (face_uint8[i+1, j+1] >= center) << 3
                code |= (face_uint8[i+1, j] >= center) << 2
                code |= (face_uint8[i+1, j-1] >= center) << 1
                code |= (face_uint8[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        # Calculate histogram of LBP
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        return hist.flatten()
    
    def extract_hog_features(self, face):
        """Extract Histogram of Oriented Gradients features"""
        # Calculate gradients
        grad_x = cv2.Sobel(face, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and angle
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # Convert angle to degrees and normalize
        angle_deg = np.degrees(angle) % 180
        
        # Divide image into cells and calculate histograms
        cell_size = 16
        n_bins = 9
        features = []
        
        for i in range(0, face.shape[0] - cell_size + 1, cell_size):
            for j in range(0, face.shape[1] - cell_size + 1, cell_size):
                cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
                cell_angle = angle_deg[i:i+cell_size, j:j+cell_size]
                
                # Calculate histogram
                hist, _ = np.histogram(cell_angle, bins=n_bins, range=(0, 180), 
                                    weights=cell_mag)
                features.extend(hist)
        
        return features
    
    def extract_gabor_features(self, face):
        """Extract Gabor filter responses"""
        features = []
        
        # Different orientations and frequencies
        orientations = [0, 45, 90, 135]
        frequencies = [0.1, 0.3, 0.5]
        
        for angle in orientations:
            for freq in frequencies:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 
                                          2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(face, cv2.CV_8UC3, kernel)
                
                # Calculate mean and std of response
                features.extend([np.mean(filtered), np.std(filtered)])
        
        return features
    
    def extract_statistical_features(self, face):
        """Extract statistical features"""
        features = []
        
        # Global statistics
        features.extend([
            np.mean(face),
            np.std(face),
            np.max(face),
            np.min(face),
            np.median(face),
            np.percentile(face, 25),
            np.percentile(face, 75)
        ])
        
        # Regional statistics (divide face into 9 regions)
        h, w = face.shape
        for i in range(3):
            for j in range(3):
                y1, y2 = i * h // 3, (i + 1) * h // 3
                x1, x2 = j * w // 3, (j + 1) * w // 3
                region = face[y1:y2, x1:x2]
                
                features.extend([
                    np.mean(region),
                    np.std(region)
                ])
        
        return features
    
    def extract_geometric_features(self, face_roi):
        """Extract geometric features using facial landmarks"""
        features = []
        
        # Detect eyes for geometric features
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda x: x[0])
            
            # Calculate eye distance
            eye1_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            eye2_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            
            eye_distance = np.sqrt((eye1_center[0] - eye2_center[0])**2 + 
                                 (eye1_center[1] - eye2_center[1])**2)
            
            # Normalize by face width
            eye_distance_norm = eye_distance / face_roi.shape[1]
            
            # Eye positions relative to face
            eye1_x_norm = eye1_center[0] / face_roi.shape[1]
            eye1_y_norm = eye1_center[1] / face_roi.shape[0]
            eye2_x_norm = eye2_center[0] / face_roi.shape[1]
            eye2_y_norm = eye2_center[1] / face_roi.shape[0]
            
            features.extend([
                eye_distance_norm,
                eye1_x_norm,
                eye1_y_norm,
                eye2_x_norm,
                eye2_y_norm
            ])
        else:
            # Default values if eyes not detected
            features.extend([0.3, 0.3, 0.3, 0.7, 0.3])
        
        return features
    
    def calculate_similarity_advanced(self, features1, features2):
        """Calculate advanced similarity between feature vectors"""
        # Ensure features are 2D arrays
        if features1.ndim == 1:
            features1 = features1.reshape(1, -1)
        if features2.ndim == 1:
            features2 = features2.reshape(1, -1)
        
        # Calculate multiple similarity metrics
        cosine_sim = cosine_similarity(features1, features2)[0, 0]
        
        # Euclidean distance (converted to similarity)
        euclidean_dist = np.linalg.norm(features1 - features2)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Manhattan distance (converted to similarity)
        manhattan_dist = np.sum(np.abs(features1 - features2))
        manhattan_sim = 1 / (1 + manhattan_dist)
        
        # Weighted combination
        combined_similarity = (0.5 * cosine_sim + 0.3 * euclidean_sim + 0.2 * manhattan_sim)
        
        return combined_similarity
    
    def collect_face_samples(self, name, num_samples=50):
        """Collect face samples with quality control"""
        if not name.strip():
            print("Please enter a valid name")
            return False
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening camera")
            return False
        
        current_face_id = self.face_id_counter
        self.face_id_counter += 1
        
        # Create directory for this person
        person_dir = os.path.join(self.training_data_path, f"person_{current_face_id}")
        os.makedirs(person_dir, exist_ok=True)
        
        count = 0
        collected_features = []
        quality_scores = []
        
        print(f"Collecting {num_samples} HIGH-QUALITY samples for {name}...")
        print("Instructions:")
        print("1. Look straight at camera")
        print("2. Slowly turn your head left and right")
        print("3. Slightly tilt your head up and down")
        print("4. Maintain good lighting")
        print("Press 'q' to quit early")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, 
                                                     minSize=(80, 80), maxSize=(300, 300))
            
            for (x, y, w, h) in faces:
                # Quality control: check face size and position
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = face_area / frame_area
                
                # Only accept faces that are appropriately sized
                if 0.05 < face_ratio < 0.4:
                    # Extract face region with margin
                    margin = 20
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    
                    face_roi = gray[y1:y2, x1:x2]
                    
                    # Quality check: ensure face has good contrast
                    face_std = np.std(face_roi)
                    if face_std > 20:  # Good contrast
                        # Extract features
                        features = self.extract_advanced_features(face_roi)
                        collected_features.append(features)
                        quality_scores.append(face_std)
                        
                        # Save face sample
                        face_filename = os.path.join(person_dir, f"face_{count}.jpg")
                        cv2.imwrite(face_filename, face_roi)
                        
                        count += 1
                        
                        # Visual feedback
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, "GOOD QUALITY", (x, y-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, "LOW QUALITY", (x, y-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "WRONG SIZE", (x, y-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Display progress
                cv2.putText(frame, f"Collected: {count}/{num_samples}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Name: {name}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Only process first face
                break
            
            cv2.imshow('Collecting Face Samples', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if count >= 10:  # Minimum samples needed
            # Normalize features using scaler
            if len(self.face_encodings) == 0:
                # First person - fit scaler
                self.feature_scaler.fit(collected_features)
            
            normalized_features = self.feature_scaler.transform(collected_features)
            
            # Store the collected features
            self.face_encodings.append({
                'name': name,
                'features': normalized_features.tolist(),
                'face_id': current_face_id,
                'quality_scores': quality_scores
            })
            self.known_names.append(name)
            print(f"Collected {count} high-quality samples for {name}")
            print(f"Average quality score: {np.mean(quality_scores):.2f}")
            self.save_training_data()
            return True
        else:
            print(f"Not enough samples collected ({count}). Need at least 10 samples.")
            return False
    
    def recognize_face(self, face_roi):
        """Recognize face with improved accuracy"""
        if len(self.face_encodings) == 0:
            return "Unknown", 0.0
        
        # Extract features
        face_features = self.extract_advanced_features(face_roi)
        
        # Normalize features
        face_features_normalized = self.feature_scaler.transform([face_features])
        
        best_match_name = "Unknown"
        best_similarity = 0.0
        all_similarities = []
        
        # Compare with all known faces
        for face_data in self.face_encodings:
            name = face_data['name']
            stored_features = np.array(face_data['features'])
            
            # Calculate similarity with all stored samples
            similarities = []
            for stored_feature in stored_features:
                similarity = self.calculate_similarity_advanced(
                    face_features_normalized, stored_feature.reshape(1, -1)
                )
                similarities.append(similarity)
            
            # Use weighted average (give more weight to high-quality samples)
            if 'quality_scores' in face_data:
                weights = np.array(face_data['quality_scores'])
                weights = weights / np.sum(weights)
                avg_similarity = np.average(similarities, weights=weights)
            else:
                avg_similarity = np.mean(similarities)
            
            all_similarities.append((name, avg_similarity))
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match_name = name
        
        # Additional confidence check: second-best match shouldn't be too close
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        if len(all_similarities) > 1:
            second_best = all_similarities[1][1]
            confidence_gap = best_similarity - second_best
            
            # If the gap is too small, be more conservative
            if confidence_gap < 0.1:
                best_similarity *= 0.8
        
        # Apply threshold
        if best_similarity < self.recognition_threshold:
            return "Unknown", best_similarity
        else:
            return best_match_name, best_similarity
    
    def recognize_faces(self):
        """Real-time face recognition with improved accuracy"""
        if len(self.face_encodings) == 0:
            print("No trained faces available. Please add faces first.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening camera")
            return
        
        print("Face recognition started. Press 'q' to quit")
        print(f"Recognition threshold: {self.recognition_threshold}")
        
        # Variables for stability
        recent_predictions = {}
        stability_threshold = 3
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, 
                                                     minSize=(80, 80), maxSize=(300, 300))
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region with margin
                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                
                face_roi = gray[y1:y2, x1:x2]
                
                # Quality check
                face_std = np.std(face_roi)
                if face_std > 20:
                    # Recognize face
                    name, confidence = self.recognize_face(face_roi)
                    
                    # Stability check
                    face_key = f"face_{i}"
                    if face_key not in recent_predictions:
                        recent_predictions[face_key] = []
                    
                    recent_predictions[face_key].append(name)
                    if len(recent_predictions[face_key]) > stability_threshold:
                        recent_predictions[face_key].pop(0)
                    
                    # Use most common recent prediction
                    if len(recent_predictions[face_key]) >= stability_threshold:
                        from collections import Counter
                        most_common = Counter(recent_predictions[face_key]).most_common(1)[0]
                        stable_name = most_common[0]
                        if most_common[1] >= 2:  # At least 2 out of last 3 predictions
                            name = stable_name
                    
                    # Determine color and display
                    if name != "Unknown":
                        color = (0, 255, 0)  # Green
                        text = f"{name} ({confidence:.2f})"
                    else:
                        color = (0, 0, 255)  # Red
                        text = f"Unknown ({confidence:.2f})"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, "Low Quality", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display info
            cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Known faces: {len(self.known_names)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Threshold: {self.recognition_threshold:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Advanced Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def adjust_threshold(self, new_threshold):
        """Adjust recognition threshold"""
        self.recognition_threshold = max(0.1, min(0.95, new_threshold))
        print(f"Recognition threshold set to: {self.recognition_threshold:.2f}")
    
    def list_known_faces(self):
        """List all known faces with details"""
        if len(self.known_names) == 0:
            print("No known faces in the system")
        else:
            print("Known faces:")
            for i, name in enumerate(self.known_names):
                face_data = self.face_encodings[i]
                num_samples = len(face_data['features'])
                avg_quality = np.mean(face_data.get('quality_scores', [0]))
                print(f"{i+1}. {name} - {num_samples} samples, avg quality: {avg_quality:.2f}")
    
    def delete_face(self, name):
        """Delete a known face from the system"""
        if name in self.known_names:
            face_index = self.known_names.index(name)
            
            # Remove from lists
            self.known_names.remove(name)
            self.face_encodings.pop(face_index)
            
            # Remove directory
            person_dir = os.path.join(self.training_data_path, f"person_{face_index}")
            if os.path.exists(person_dir):
                import shutil
                shutil.rmtree(person_dir)
            
            # Retrain scaler if faces remain
            if len(self.face_encodings) > 0:
                all_features = []
                for face_data in self.face_encodings:
                    all_features.extend(face_data['features'])
                self.feature_scaler.fit(all_features)
            
            self.save_training_data()
            print(f"Deleted {name} from the system")
        else:
            print(f"Face {name} not found in the system")

def main():
    print("=== Advanced Face Recognition System ===")
    print("Features:")
    print("- Advanced feature extraction (LBP, HOG, Gabor, Statistical, Geometric)")
    print("- Quality control during sample collection")
    print("- Feature normalization and scaling")
    print("- Multiple similarity metrics")
    print("- Prediction stability checking")
    print("- Adjustable recognition threshold")
    
    try:
        # Check dependencies
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import StandardScaler
        print("✓ All dependencies available")
    except ImportError:
        print("Installing required packages...")
        os.system("pip install scikit-learn")
        print("Please restart the program")
        return
    
    system = ImprovedFaceRecognitionSystem()
    
    while True:
        print("\n" + "="*50)
        print("ADVANCED FACE RECOGNITION SYSTEM")
        print("="*50)
        print("1. Add new face (collect samples)")
        print("2. Start face recognition")
        print("3. List known faces")
        print("4. Delete face")
        print("5. Adjust recognition threshold")
        print("6. System info")
        print("7. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            name = input("Enter person's name: ").strip()
            if name:
                print(f"\nPreparing to collect samples for: {name}")
                print("Make sure you have good lighting and look at the camera")
                input("Press Enter when ready...")
                
                if system.collect_face_samples(name):
                    print(f"✓ Successfully added {name} to the system")
                else:
                    print("✗ Failed to add face")
            else:
                print("Please enter a valid name")
        
        elif choice == '2':
            system.recognize_faces()
        
        elif choice == '3':
            system.list_known_faces()
        
        elif choice == '4':
            system.list_known_faces()
            name = input("Enter name to delete: ").strip()
            if name:
                system.delete_face(name)
        
        elif choice == '5':
            current = system.recognition_threshold
            print(f"Current threshold: {current:.2f}")
            print("Lower = more strict, Higher = more lenient")
            try:
                new_threshold = float(input("Enter new threshold (0.1-0.95): "))
                system.adjust_threshold(new_threshold)
            except ValueError:
                print("Invalid threshold value")
        
        elif choice == '6':
            print(f"Known faces: {len(system.known_names)}")
            print(f"Recognition threshold: {system.recognition_threshold:.2f}")
            print(f"Face size: {system.face_size}")
            print(f"OpenCV version: {cv2.__version__}")
        
        elif choice == '7':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
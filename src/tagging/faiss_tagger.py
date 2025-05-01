import os
import json
import numpy as np
import faiss
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTagger:
    def __init__(self, index_dir: str):
        """
        Initialize the FAISS-based model tagger.
        
        Args:
            index_dir (str): Directory to store FAISS indices and metadata
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.dimension = 512  # Dimension of the feature vectors
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Load existing index if available
        self.index_path = self.index_dir / "model_index.faiss"
        self.metadata_path = self.index_dir / "model_metadata.json"
        
        # Initialize metadata
        self.metadata = []
        
        # Try to load existing index and metadata
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            
            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    logger.info(f"Loaded existing metadata from {self.metadata_path}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to load metadata file: {e}. Starting with empty metadata.")
                    # Create a backup of corrupted file
                    backup_path = self.metadata_path.with_suffix('.json.bak')
                    if self.metadata_path.exists():
                        self.metadata_path.rename(backup_path)
                    # Create new empty metadata file
                    self._save_index()
        except Exception as e:
            logger.warning(f"Error loading existing index/metadata: {e}. Starting fresh.")
            self._save_index()
    
    def _extract_model_features(self, model_path: str, model_info: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from model metadata and performance metrics.
        
        Args:
            model_path (str): Path to the model file
            model_info (Dict[str, Any]): Model metadata and metrics
            
        Returns:
            np.ndarray: Feature vector
        """
        # Create a feature vector from model metadata
        features = []
        
        # Add model architecture features
        features.extend([
            model_info.get('num_layers', 0),
            model_info.get('hidden_neurons', 0),
            model_info.get('total_parameters', 0),
            model_info.get('pruned_parameters', 0) if 'pruned_parameters' in model_info else 0,
            model_info.get('quantization_bits', 0) if 'quantization_bits' in model_info else 0,
            1.0 if model_info.get('activation', '') == 'relu' else 0.0,
            1.0 if model_info.get('activation', '') == 'sigmoid' else 0.0,
            1.0 if model_info.get('activation', '') == 'tanh' else 0.0
        ])
        
        # Add performance metrics
        features.extend([
            model_info.get('accuracy', 0),
            model_info.get('inference_time', 0),
            model_info.get('model_size', 0)
        ])
        
        # Add profiling metrics if available
        if 'profiling' in model_info:
            profiling = model_info['profiling']
            
            # Add metrics for each device
            for device in ["raspberry-pi-4", "raspberry-pi-rp2040"]:
                if device in profiling:
                    device_data = profiling[device]
                    features.extend([
                        device_data.get("inference_time_ms", 0),
                        device_data.get("rom_size_bytes", 0),
                        device_data.get("ram_usage_bytes", 0),
                        1.0 if device_data.get("mcu_supported", False) else 0.0
                    ])
                else:
                    # Add zeros if device data not available
                    features.extend([0, 0, 0, 0])
        else:
            # Add zeros if profiling data not available
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        # Pad or truncate to fixed dimension
        features = np.array(features, dtype=np.float32)
        if len(features) < self.dimension:
            features = np.pad(features, (0, self.dimension - len(features)))
        elif len(features) > self.dimension:
            features = features[:self.dimension]
            
        return features
    
    def _serialize_model_info(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert model info to a serializable format.
        
        Args:
            model_info (Dict[str, Any]): Original model info
            
        Returns:
            Dict[str, Any]: Serializable model info
        """
        serializable_info = {}
        
        # Extract basic metrics
        serializable_info['accuracy'] = float(model_info.get('accuracy', 0))
        serializable_info['inference_time'] = float(model_info.get('inference_time', 0))
        serializable_info['model_size'] = float(model_info.get('model_size', 0))
        
        # Extract architecture info
        serializable_info['num_layers'] = int(model_info.get('num_layers', 0))
        serializable_info['hidden_neurons'] = int(model_info.get('hidden_neurons', 0))
        serializable_info['total_parameters'] = int(model_info.get('total_parameters', 0))
        serializable_info['activation'] = str(model_info.get('activation', 'relu'))
        
        # Extract optimization info
        if 'pruned_parameters' in model_info:
            serializable_info['pruned_parameters'] = int(model_info['pruned_parameters'])
        if 'quantization_bits' in model_info:
            serializable_info['quantization_bits'] = int(model_info['quantization_bits'])
        
        # Extract training info
        if 'epochs' in model_info:
            serializable_info['epochs'] = int(model_info['epochs'])
        if 'learning_rate' in model_info:
            serializable_info['learning_rate'] = float(model_info['learning_rate'])
        
        # Extract dataset info
        if 'dataset_name' in model_info:
            serializable_info['dataset_name'] = str(model_info['dataset_name'])
        
        return serializable_info
    
    def tag_model(self, model_path: str, model_info: Dict[str, Any]) -> int:
        """
        Tag a model and add it to the FAISS index.
        
        Args:
            model_path (str): Path to the model file
            model_info (Dict[str, Any]): Model metadata and metrics
            
        Returns:
            int: Index ID of the added model
        """
        try:
            # Extract features
            features = self._extract_model_features(model_path, model_info)
            
            # Add to FAISS index
            self.index.add(np.array([features]))
            
            # Get the ID of the added vector
            model_id = len(self.metadata)
            
            # Store metadata with serialized info
            self.metadata.append({
                'id': model_id,
                'path': model_path,
                'info': self._serialize_model_info(model_info)
            })
            
            # Save index and metadata
            self._save_index()
            
            logger.info(f"Successfully tagged model {model_id} at {model_path}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error tagging model: {e}")
            raise
    
    def find_similar_models(self, query_model_path: str, query_info: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar models using FAISS search.
        
        Args:
            query_model_path (str): Path to the query model
            query_info (Dict[str, Any]): Query model metadata
            k (int): Number of similar models to return
            
        Returns:
            List[Dict[str, Any]]: List of similar models with their metadata
        """
        try:
            # Extract query features
            query_features = self._extract_model_features(query_model_path, query_info)
            
            # Search in FAISS index
            distances, indices = self.index.search(np.array([query_features]), k)
            
            # Get metadata for similar models
            similar_models = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):  # Ensure valid index
                    model_data = self.metadata[idx].copy()
                    model_data['distance'] = float(distance)
                    similar_models.append(model_data)
            
            return similar_models
            
        except Exception as e:
            logger.error(f"Error finding similar models: {e}")
            return []
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.info(f"Successfully saved index and metadata to {self.index_dir}")
            
        except Exception as e:
            logger.error(f"Error saving index and metadata: {e}")
            raise
    
    def get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve model metadata by ID.
        
        Args:
            model_id (int): ID of the model to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Model metadata if found, None otherwise
        """
        if 0 <= model_id < len(self.metadata):
            return self.metadata[model_id]
        return None 
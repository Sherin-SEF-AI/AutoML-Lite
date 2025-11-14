"""
Architecture Repository for Neural Architecture Search.

This module provides functionality for storing, retrieving, and managing
neural network architectures with transfer learning support.
"""

import json
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from .architecture import Architecture, LayerConfig

logger = logging.getLogger(__name__)


class ArchitectureRepository:
    """
    Repository for storing and retrieving neural network architectures.
    
    Uses SQLite as the backend storage for architecture configurations,
    metadata, and performance metrics. Supports similarity-based retrieval
    for transfer learning.
    
    Attributes:
        db_path: Path to the SQLite database file
        conn: SQLite database connection
    """
    
    def __init__(self, db_path: str = '~/.automl_lite/nas_architectures.db'):
        """
        Initialize the architecture repository.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Expand user path and create directory if needed
        self.db_path = Path(os.path.expanduser(db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to architecture repository at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            cursor = self.conn.cursor()
            
            # Main architectures table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS architectures (
                    id TEXT PRIMARY KEY,
                    architecture_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Dataset metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    architecture_id TEXT PRIMARY KEY,
                    problem_type TEXT,
                    n_samples INTEGER,
                    n_features INTEGER,
                    n_classes INTEGER,
                    dataset_name TEXT,
                    FOREIGN KEY (architecture_id) REFERENCES architectures(id)
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    architecture_id TEXT PRIMARY KEY,
                    accuracy REAL,
                    loss REAL,
                    val_accuracy REAL,
                    val_loss REAL,
                    training_time REAL,
                    FOREIGN KEY (architecture_id) REFERENCES architectures(id)
                )
            """)
            
            # Hardware metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hardware_metrics (
                    architecture_id TEXT PRIMARY KEY,
                    latency_ms REAL,
                    memory_mb REAL,
                    model_size_mb REAL,
                    flops REAL,
                    num_parameters INTEGER,
                    target_hardware TEXT,
                    FOREIGN KEY (architecture_id) REFERENCES architectures(id)
                )
            """)
            
            # Search metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_metadata (
                    architecture_id TEXT PRIMARY KEY,
                    search_strategy TEXT,
                    search_time REAL,
                    search_space_type TEXT,
                    generation INTEGER,
                    FOREIGN KEY (architecture_id) REFERENCES architectures(id)
                )
            """)
            
            # Tags table for flexible categorization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    architecture_id TEXT,
                    tag TEXT,
                    PRIMARY KEY (architecture_id, tag),
                    FOREIGN KEY (architecture_id) REFERENCES architectures(id)
                )
            """)
            
            # Create indices for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_problem_type 
                ON dataset_metadata(problem_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_accuracy 
                ON performance_metrics(accuracy)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON architectures(created_at)
            """)
            
            self.conn.commit()
            logger.info("Database tables created successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def save_architecture(
        self,
        architecture: Architecture,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        hardware_metrics: Optional[Dict[str, float]] = None,
        search_metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save an architecture to the repository.
        
        Args:
            architecture: Architecture object to save
            dataset_metadata: Metadata about the dataset (problem_type, n_samples, etc.)
            performance_metrics: Performance metrics (accuracy, loss, etc.)
            hardware_metrics: Hardware metrics (latency, memory, etc.)
            search_metadata: Search process metadata (strategy, time, etc.)
            tags: List of tags for categorization
        
        Returns:
            Architecture ID
        """
        try:
            cursor = self.conn.cursor()
            arch_id = architecture.id
            now = datetime.now().isoformat()
            
            # Save main architecture
            cursor.execute("""
                INSERT OR REPLACE INTO architectures (id, architecture_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (arch_id, architecture.to_json(), now, now))
            
            # Save dataset metadata
            if dataset_metadata:
                cursor.execute("""
                    INSERT OR REPLACE INTO dataset_metadata 
                    (architecture_id, problem_type, n_samples, n_features, n_classes, dataset_name)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    arch_id,
                    dataset_metadata.get('problem_type'),
                    dataset_metadata.get('n_samples'),
                    dataset_metadata.get('n_features'),
                    dataset_metadata.get('n_classes'),
                    dataset_metadata.get('dataset_name')
                ))
            
            # Save performance metrics
            if performance_metrics:
                cursor.execute("""
                    INSERT OR REPLACE INTO performance_metrics 
                    (architecture_id, accuracy, loss, val_accuracy, val_loss, training_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    arch_id,
                    performance_metrics.get('accuracy'),
                    performance_metrics.get('loss'),
                    performance_metrics.get('val_accuracy'),
                    performance_metrics.get('val_loss'),
                    performance_metrics.get('training_time')
                ))
            
            # Save hardware metrics
            if hardware_metrics:
                cursor.execute("""
                    INSERT OR REPLACE INTO hardware_metrics 
                    (architecture_id, latency_ms, memory_mb, model_size_mb, flops, 
                     num_parameters, target_hardware)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    arch_id,
                    hardware_metrics.get('latency_ms'),
                    hardware_metrics.get('memory_mb'),
                    hardware_metrics.get('model_size_mb'),
                    hardware_metrics.get('flops'),
                    hardware_metrics.get('num_parameters'),
                    hardware_metrics.get('target_hardware')
                ))
            
            # Save search metadata
            if search_metadata:
                cursor.execute("""
                    INSERT OR REPLACE INTO search_metadata 
                    (architecture_id, search_strategy, search_time, search_space_type, generation)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    arch_id,
                    search_metadata.get('search_strategy'),
                    search_metadata.get('search_time'),
                    search_metadata.get('search_space_type'),
                    search_metadata.get('generation')
                ))
            
            # Save tags
            if tags:
                # Delete existing tags first
                cursor.execute("DELETE FROM tags WHERE architecture_id = ?", (arch_id,))
                # Insert new tags
                for tag in tags:
                    cursor.execute("""
                        INSERT INTO tags (architecture_id, tag)
                        VALUES (?, ?)
                    """, (arch_id, tag))
            
            self.conn.commit()
            logger.info(f"Saved architecture {arch_id} to repository")
            return arch_id
            
        except sqlite3.Error as e:
            logger.error(f"Failed to save architecture: {e}")
            self.conn.rollback()
            raise
    
    def load_architecture(self, architecture_id: str) -> Optional[Tuple[Architecture, Dict[str, Any]]]:
        """
        Load an architecture from the repository.
        
        Args:
            architecture_id: ID of the architecture to load
        
        Returns:
            Tuple of (Architecture, metadata_dict) or None if not found
        """
        try:
            cursor = self.conn.cursor()
            
            # Load main architecture
            cursor.execute("""
                SELECT architecture_json, created_at, updated_at
                FROM architectures
                WHERE id = ?
            """, (architecture_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Architecture {architecture_id} not found")
                return None
            
            architecture = Architecture.from_json(row['architecture_json'])
            
            # Load all metadata
            metadata = {
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            }
            
            # Load dataset metadata
            cursor.execute("""
                SELECT * FROM dataset_metadata WHERE architecture_id = ?
            """, (architecture_id,))
            dataset_row = cursor.fetchone()
            if dataset_row:
                metadata['dataset_metadata'] = dict(dataset_row)
                del metadata['dataset_metadata']['architecture_id']
            
            # Load performance metrics
            cursor.execute("""
                SELECT * FROM performance_metrics WHERE architecture_id = ?
            """, (architecture_id,))
            perf_row = cursor.fetchone()
            if perf_row:
                metadata['performance_metrics'] = dict(perf_row)
                del metadata['performance_metrics']['architecture_id']
            
            # Load hardware metrics
            cursor.execute("""
                SELECT * FROM hardware_metrics WHERE architecture_id = ?
            """, (architecture_id,))
            hw_row = cursor.fetchone()
            if hw_row:
                metadata['hardware_metrics'] = dict(hw_row)
                del metadata['hardware_metrics']['architecture_id']
            
            # Load search metadata
            cursor.execute("""
                SELECT * FROM search_metadata WHERE architecture_id = ?
            """, (architecture_id,))
            search_row = cursor.fetchone()
            if search_row:
                metadata['search_metadata'] = dict(search_row)
                del metadata['search_metadata']['architecture_id']
            
            # Load tags
            cursor.execute("""
                SELECT tag FROM tags WHERE architecture_id = ?
            """, (architecture_id,))
            tags = [row['tag'] for row in cursor.fetchall()]
            if tags:
                metadata['tags'] = tags
            
            logger.info(f"Loaded architecture {architecture_id} from repository")
            return architecture, metadata
            
        except sqlite3.Error as e:
            logger.error(f"Failed to load architecture: {e}")
            return None
    
    def list_architectures(
        self,
        problem_type: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        List architectures in the repository with optional filtering.
        
        Args:
            problem_type: Filter by problem type
            min_accuracy: Minimum accuracy threshold
            tags: Filter by tags (architectures must have all specified tags)
            limit: Maximum number of results
        
        Returns:
            List of (architecture_id, summary_dict) tuples
        """
        try:
            cursor = self.conn.cursor()
            
            # Build query with filters
            query = """
                SELECT DISTINCT a.id, a.created_at,
                       dm.problem_type, dm.n_samples, dm.n_features,
                       pm.accuracy, pm.val_accuracy,
                       hm.latency_ms, hm.model_size_mb
                FROM architectures a
                LEFT JOIN dataset_metadata dm ON a.id = dm.architecture_id
                LEFT JOIN performance_metrics pm ON a.id = pm.architecture_id
                LEFT JOIN hardware_metrics hm ON a.id = hm.architecture_id
                WHERE 1=1
            """
            params = []
            
            if problem_type:
                query += " AND dm.problem_type = ?"
                params.append(problem_type)
            
            if min_accuracy is not None:
                query += " AND pm.accuracy >= ?"
                params.append(min_accuracy)
            
            if tags:
                # Filter by tags using subquery
                placeholders = ','.join('?' * len(tags))
                query += f"""
                    AND a.id IN (
                        SELECT architecture_id 
                        FROM tags 
                        WHERE tag IN ({placeholders})
                        GROUP BY architecture_id
                        HAVING COUNT(DISTINCT tag) = ?
                    )
                """
                params.extend(tags)
                params.append(len(tags))
            
            query += " ORDER BY pm.accuracy DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                summary = {
                    'id': row['id'],
                    'created_at': row['created_at'],
                    'problem_type': row['problem_type'],
                    'n_samples': row['n_samples'],
                    'n_features': row['n_features'],
                    'accuracy': row['accuracy'],
                    'val_accuracy': row['val_accuracy'],
                    'latency_ms': row['latency_ms'],
                    'model_size_mb': row['model_size_mb']
                }
                results.append((row['id'], summary))
            
            logger.info(f"Listed {len(results)} architectures from repository")
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Failed to list architectures: {e}")
            return []
    
    def delete_architecture(self, architecture_id: str) -> bool:
        """
        Delete an architecture from the repository.
        
        Args:
            architecture_id: ID of the architecture to delete
        
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Delete from all tables (cascading)
            cursor.execute("DELETE FROM tags WHERE architecture_id = ?", (architecture_id,))
            cursor.execute("DELETE FROM search_metadata WHERE architecture_id = ?", (architecture_id,))
            cursor.execute("DELETE FROM hardware_metrics WHERE architecture_id = ?", (architecture_id,))
            cursor.execute("DELETE FROM performance_metrics WHERE architecture_id = ?", (architecture_id,))
            cursor.execute("DELETE FROM dataset_metadata WHERE architecture_id = ?", (architecture_id,))
            cursor.execute("DELETE FROM architectures WHERE id = ?", (architecture_id,))
            
            self.conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Deleted architecture {architecture_id} from repository")
                return True
            else:
                logger.warning(f"Architecture {architecture_id} not found for deletion")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Failed to delete architecture: {e}")
            self.conn.rollback()
            return False
    
    def compute_similarity(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> float:
        """
        Compute similarity score between two architecture metadata dictionaries.
        
        The similarity score is based on dataset characteristics:
        - Problem type match (40% weight)
        - Dataset size similarity (30% weight)
        - Feature count similarity (30% weight)
        
        Args:
            metadata1: First architecture's dataset metadata
            metadata2: Second architecture's dataset metadata
        
        Returns:
            Similarity score between 0 and 1 (1 = most similar)
        """
        score = 0.0
        
        # Problem type match (0.4 weight)
        if metadata1.get('problem_type') == metadata2.get('problem_type'):
            score += 0.4
        
        # Dataset size similarity (0.3 weight)
        n_samples1 = metadata1.get('n_samples', 0)
        n_samples2 = metadata2.get('n_samples', 0)
        if n_samples1 > 0 and n_samples2 > 0:
            size_ratio = min(n_samples1, n_samples2) / max(n_samples1, n_samples2)
            score += 0.3 * size_ratio
        
        # Feature count similarity (0.3 weight)
        n_features1 = metadata1.get('n_features', 0)
        n_features2 = metadata2.get('n_features', 0)
        if n_features1 > 0 and n_features2 > 0:
            feat_ratio = min(n_features1, n_features2) / max(n_features1, n_features2)
            score += 0.3 * feat_ratio
        
        return score
    
    def find_similar_architectures(
        self,
        dataset_metadata: Dict[str, Any],
        top_k: int = 3,
        min_similarity: float = 0.3
    ) -> List[Tuple[Architecture, Dict[str, Any], float]]:
        """
        Find architectures similar to the given dataset characteristics.
        
        This method is used for transfer learning to identify architectures
        that performed well on similar problems.
        
        Args:
            dataset_metadata: Metadata about the target dataset
                (problem_type, n_samples, n_features, etc.)
            top_k: Number of similar architectures to return
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            List of (Architecture, metadata, similarity_score) tuples,
            sorted by similarity score (descending)
        """
        try:
            cursor = self.conn.cursor()
            
            # Get all architectures with their metadata
            cursor.execute("""
                SELECT a.id, a.architecture_json,
                       dm.problem_type, dm.n_samples, dm.n_features, dm.n_classes,
                       pm.accuracy, pm.val_accuracy,
                       hm.latency_ms, hm.model_size_mb
                FROM architectures a
                LEFT JOIN dataset_metadata dm ON a.id = dm.architecture_id
                LEFT JOIN performance_metrics pm ON a.id = pm.architecture_id
                LEFT JOIN hardware_metrics hm ON a.id = hm.architecture_id
            """)
            
            # Compute similarity for each architecture
            candidates = []
            for row in cursor.fetchall():
                arch_metadata = {
                    'problem_type': row['problem_type'],
                    'n_samples': row['n_samples'],
                    'n_features': row['n_features'],
                    'n_classes': row['n_classes']
                }
                
                similarity = self.compute_similarity(dataset_metadata, arch_metadata)
                
                if similarity >= min_similarity:
                    architecture = Architecture.from_json(row['architecture_json'])
                    
                    full_metadata = {
                        'dataset_metadata': arch_metadata,
                        'performance_metrics': {
                            'accuracy': row['accuracy'],
                            'val_accuracy': row['val_accuracy']
                        },
                        'hardware_metrics': {
                            'latency_ms': row['latency_ms'],
                            'model_size_mb': row['model_size_mb']
                        }
                    }
                    
                    candidates.append((architecture, full_metadata, similarity))
            
            # Sort by similarity (descending) and return top k
            candidates.sort(key=lambda x: x[2], reverse=True)
            results = candidates[:top_k]
            
            logger.info(f"Found {len(results)} similar architectures (min_similarity={min_similarity})")
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Failed to find similar architectures: {e}")
            return []
    
    def adapt_architecture(
        self,
        architecture: Architecture,
        new_input_shape: Tuple[int, ...],
        new_output_shape: Tuple[int, ...],
        dataset_size: Optional[int] = None,
        scale_factor: Optional[float] = None
    ) -> Architecture:
        """
        Adapt an architecture to a new problem.
        
        This method modifies the input and output layers to match the new
        problem's requirements, and optionally scales layer sizes based on
        dataset size.
        
        Args:
            architecture: Source architecture to adapt
            new_input_shape: New input shape (e.g., (784,) for MNIST)
            new_output_shape: New output shape (e.g., (10,) for 10 classes)
            dataset_size: Size of the new dataset (for scaling)
            scale_factor: Manual scaling factor for layer sizes (overrides dataset_size)
        
        Returns:
            Adapted architecture with new ID
        """
        import copy
        
        # Clone the architecture
        adapted = architecture.clone()
        
        # Modify first layer (input layer)
        if adapted.layers:
            first_layer = adapted.layers[0]
            
            # Update input shape
            first_layer.input_shape = new_input_shape
            
            # For dense layers, update units based on input size
            if first_layer.layer_type == 'dense':
                # Keep the same relative size or use new input size
                if len(new_input_shape) == 1:
                    # For tabular data, input size is feature count
                    pass  # Units are independent of input shape
            
            # For convolutional layers, update input shape
            elif first_layer.layer_type in ['conv2d', 'conv1d']:
                # Input shape is handled automatically by the layer
                pass
        
        # Modify last layer (output layer)
        if adapted.layers:
            last_layer = adapted.layers[-1]
            
            # Update output shape
            last_layer.output_shape = new_output_shape
            
            # For dense output layers, update units to match output size
            if last_layer.layer_type == 'dense':
                if len(new_output_shape) == 1:
                    last_layer.params['units'] = new_output_shape[0]
        
        # Scale layer sizes based on dataset size or scale factor
        if scale_factor is not None:
            self._scale_architecture_layers(adapted, scale_factor)
        elif dataset_size is not None:
            # Compute scale factor based on dataset size
            # Use log scale to avoid extreme scaling
            # Baseline: 10,000 samples -> scale_factor = 1.0
            baseline_size = 10000
            computed_scale = (dataset_size / baseline_size) ** 0.5
            # Clamp scale factor between 0.5 and 2.0
            computed_scale = max(0.5, min(2.0, computed_scale))
            self._scale_architecture_layers(adapted, computed_scale)
        
        # Update metadata
        adapted.metadata['adapted_from'] = architecture.id
        adapted.metadata['adaptation'] = {
            'new_input_shape': list(new_input_shape),
            'new_output_shape': list(new_output_shape),
            'dataset_size': dataset_size,
            'scale_factor': scale_factor
        }
        
        logger.info(f"Adapted architecture {architecture.id} to new problem")
        return adapted
    
    def _scale_architecture_layers(self, architecture: Architecture, scale_factor: float):
        """
        Scale the size of layers in an architecture.
        
        This is a helper method for adapt_architecture that scales layer
        parameters (units, filters) by the given factor.
        
        Args:
            architecture: Architecture to scale (modified in-place)
            scale_factor: Scaling factor (e.g., 1.5 = 50% larger)
        """
        for i, layer in enumerate(architecture.layers):
            # Skip first and last layers (input/output)
            if i == 0 or i == len(architecture.layers) - 1:
                continue
            
            # Scale dense layer units
            if layer.layer_type == 'dense':
                if 'units' in layer.params:
                    original_units = layer.params['units']
                    scaled_units = int(original_units * scale_factor)
                    # Ensure at least 8 units and round to nearest multiple of 8
                    scaled_units = max(8, (scaled_units // 8) * 8)
                    layer.params['units'] = scaled_units
            
            # Scale convolutional layer filters
            elif layer.layer_type in ['conv2d', 'conv1d']:
                if 'filters' in layer.params:
                    original_filters = layer.params['filters']
                    scaled_filters = int(original_filters * scale_factor)
                    # Ensure at least 8 filters and round to nearest multiple of 8
                    scaled_filters = max(8, (scaled_filters // 8) * 8)
                    layer.params['filters'] = scaled_filters
            
            # Scale LSTM/GRU units
            elif layer.layer_type in ['lstm', 'gru']:
                if 'units' in layer.params:
                    original_units = layer.params['units']
                    scaled_units = int(original_units * scale_factor)
                    # Ensure at least 16 units and round to nearest multiple of 16
                    scaled_units = max(16, (scaled_units // 16) * 16)
                    layer.params['units'] = scaled_units
    
    def export_architecture(
        self,
        architecture_id: str,
        output_path: str,
        include_metadata: bool = True
    ) -> bool:
        """
        Export an architecture to a JSON file.
        
        Args:
            architecture_id: ID of the architecture to export
            output_path: Path to the output JSON file
            include_metadata: Whether to include all metadata in export
        
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            # Load architecture and metadata
            result = self.load_architecture(architecture_id)
            if not result:
                logger.error(f"Architecture {architecture_id} not found for export")
                return False
            
            architecture, metadata = result
            
            # Prepare export data
            export_data = {
                'architecture': architecture.to_dict(),
                'export_version': '1.0',
                'exported_at': datetime.now().isoformat()
            }
            
            if include_metadata:
                export_data['metadata'] = metadata
            
            # Write to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported architecture {architecture_id} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export architecture: {e}")
            return False
    
    def import_architecture(
        self,
        input_path: str,
        validate: bool = True,
        save_to_repository: bool = True
    ) -> Optional[Architecture]:
        """
        Import an architecture from a JSON file.
        
        Args:
            input_path: Path to the input JSON file
            validate: Whether to validate the architecture after import
            save_to_repository: Whether to save to repository after import
        
        Returns:
            Imported Architecture object or None if import failed
        """
        try:
            # Read from file
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Validate export format
            if 'architecture' not in import_data:
                logger.error("Invalid export format: missing 'architecture' key")
                return None
            
            # Create architecture from data
            architecture = Architecture.from_dict(import_data['architecture'])
            
            # Validate architecture if requested
            if validate:
                if not self._validate_imported_architecture(architecture):
                    logger.error("Architecture validation failed")
                    return None
            
            # Save to repository if requested
            if save_to_repository:
                metadata = import_data.get('metadata', {})
                
                self.save_architecture(
                    architecture,
                    dataset_metadata=metadata.get('dataset_metadata'),
                    performance_metrics=metadata.get('performance_metrics'),
                    hardware_metrics=metadata.get('hardware_metrics'),
                    search_metadata=metadata.get('search_metadata'),
                    tags=metadata.get('tags')
                )
            
            logger.info(f"Imported architecture from {input_path}")
            return architecture
            
        except Exception as e:
            logger.error(f"Failed to import architecture: {e}")
            return None
    
    def _validate_imported_architecture(self, architecture: Architecture) -> bool:
        """
        Validate an imported architecture.
        
        Checks:
        - Architecture has at least one layer
        - All layers have valid types
        - Connections reference valid layer indices
        - Layer parameters are reasonable
        
        Args:
            architecture: Architecture to validate
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check has layers
            if not architecture.layers:
                logger.error("Architecture has no layers")
                return False
            
            # Check layer types
            valid_layer_types = {
                'dense', 'conv2d', 'conv1d', 'maxpooling2d', 'maxpooling1d',
                'avgpooling2d', 'avgpooling1d', 'dropout', 'batchnormalization',
                'lstm', 'gru', 'flatten', 'reshape', 'activation'
            }
            
            for i, layer in enumerate(architecture.layers):
                if layer.layer_type not in valid_layer_types:
                    logger.warning(f"Layer {i} has unknown type: {layer.layer_type}")
                
                # Check layer parameters are reasonable
                if layer.layer_type == 'dense':
                    units = layer.params.get('units', 0)
                    if units <= 0 or units > 10000:
                        logger.error(f"Dense layer {i} has invalid units: {units}")
                        return False
                
                elif layer.layer_type in ['conv2d', 'conv1d']:
                    filters = layer.params.get('filters', 0)
                    if filters <= 0 or filters > 2048:
                        logger.error(f"Conv layer {i} has invalid filters: {filters}")
                        return False
                
                elif layer.layer_type in ['lstm', 'gru']:
                    units = layer.params.get('units', 0)
                    if units <= 0 or units > 2048:
                        logger.error(f"Recurrent layer {i} has invalid units: {units}")
                        return False
                
                elif layer.layer_type == 'dropout':
                    rate = layer.params.get('rate', 0)
                    if rate < 0 or rate >= 1:
                        logger.error(f"Dropout layer {i} has invalid rate: {rate}")
                        return False
            
            # Check connections
            max_idx = len(architecture.layers) - 1
            for from_idx, to_idx in architecture.connections:
                if from_idx < 0 or from_idx > max_idx:
                    logger.error(f"Invalid connection: from_idx {from_idx} out of range")
                    return False
                if to_idx < 0 or to_idx > max_idx:
                    logger.error(f"Invalid connection: to_idx {to_idx} out of range")
                    return False
                if from_idx >= to_idx:
                    logger.error(f"Invalid connection: from_idx {from_idx} >= to_idx {to_idx}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the repository.
        
        Returns:
            Dictionary with repository statistics
        """
        try:
            cursor = self.conn.cursor()
            
            stats = {}
            
            # Total architectures
            cursor.execute("SELECT COUNT(*) as count FROM architectures")
            stats['total_architectures'] = cursor.fetchone()['count']
            
            # Architectures by problem type
            cursor.execute("""
                SELECT problem_type, COUNT(*) as count
                FROM dataset_metadata
                GROUP BY problem_type
            """)
            stats['by_problem_type'] = {row['problem_type']: row['count'] 
                                       for row in cursor.fetchall()}
            
            # Average accuracy
            cursor.execute("""
                SELECT AVG(accuracy) as avg_accuracy, MAX(accuracy) as max_accuracy
                FROM performance_metrics
            """)
            row = cursor.fetchone()
            stats['avg_accuracy'] = row['avg_accuracy']
            stats['max_accuracy'] = row['max_accuracy']
            
            # Most common tags
            cursor.execute("""
                SELECT tag, COUNT(*) as count
                FROM tags
                GROUP BY tag
                ORDER BY count DESC
                LIMIT 10
            """)
            stats['top_tags'] = {row['tag']: row['count'] for row in cursor.fetchall()}
            
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed architecture repository connection")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()

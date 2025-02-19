import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
import pickle
from datetime import datetime
from contextlib import contextmanager
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from app.config import settings

from app.models import StoredSignature, ComparisonParams

DATABASE_PATH = Path("signatures.db")

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DATABASE_PATH))
    try:
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        # Return dictionaries instead of tuples
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize database with required tables"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Table for storing signatures
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signatures (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            image_data BLOB NOT NULL,
            features BLOB NOT NULL,
            metadata TEXT,
            detection_params BLOB,
            comparison_params BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Table for storing detection results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_results (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            signature_locations BLOB NOT NULL,
            params BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Table for storing comparison results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS comparison_results (
            id TEXT PRIMARY KEY,
            signature1_id TEXT NOT NULL,
            signature2_id TEXT NOT NULL,
            similarity_score REAL NOT NULL,
            comparison_params BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (signature1_id) REFERENCES signatures (id),
            FOREIGN KEY (signature2_id) REFERENCES signatures (id)
        )
        ''')
        
        conn.commit()

def store_signature(
    signature_id: str,
    filename: str,
    image_data: bytes,
    features: Dict[str, np.ndarray],
    metadata: Optional[str] = None,
    detection_params: Optional[Dict] = None,
    comparison_params: Optional[Dict] = None
) -> str:
    """Store a signature and its features in the database"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO signatures (
            id, filename, image_data, features, metadata,
            detection_params, comparison_params, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signature_id,
            filename,
            image_data,
            pickle.dumps(features),
            metadata,
            pickle.dumps(detection_params) if detection_params else None,
            pickle.dumps(comparison_params) if comparison_params else None,
            datetime.utcnow()
        ))
        
        conn.commit()
        return signature_id

def get_signature(signature_id: str) -> Optional[StoredSignature]:
    """Retrieve a signature by its ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM signatures WHERE id = ?
        ''', (signature_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        return StoredSignature(
            id=row['id'],
            filename=row['filename'],
            metadata=row['metadata'],
            features=pickle.loads(row['features']),
            created_at=datetime.fromisoformat(row['created_at'])
        )

def get_signature_path(signature_id: str) -> Optional[Path]:
    """Get path of stored signature file"""
    signature_dir = Path(settings.TEMP_DIR) / "signatures" / signature_id
    
    if not signature_dir.exists():
        return None
    
    # Find signature file
    signature_files = list(signature_dir.glob("signature.*"))
    return signature_files[0] if signature_files else None

def get_signature_image(signature_id: str) -> Optional[bytes]:
    """Retrieve signature image data by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT image_data FROM signatures WHERE id = ?
        ''', (signature_id,))
        
        row = cursor.fetchone()
        return row['image_data'] if row else None

def delete_signature(signature_id: str) -> bool:
    """Delete a signature by its ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        DELETE FROM signatures WHERE id = ?
        ''', (signature_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted

def find_matches(
    features: Dict[str, np.ndarray],
    limit: int = 5,
    params: Optional[ComparisonParams] = None
) -> List[Dict[str, Any]]:
    """Find similar signatures in the database"""
    if params is None:
        params = ComparisonParams()
    
    matches = []
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get all signatures
        cursor.execute('SELECT id, features FROM signatures')
        
        for row in cursor:
            stored_features = pickle.loads(row['features'])
            
            # Calculate similarity score
            similarity = compare_features(features, stored_features, params)
            
            if similarity >= params.threshold:
                matches.append({
                    'signature_id': row['id'],
                    'similarity_score': similarity
                })
    
    # Sort by similarity score and limit results
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    return matches[:limit]

def store_comparison_result(
    signature1_id: str,
    signature2_id: str,
    similarity_score: float,
    comparison_params: Optional[Dict] = None
) -> str:
    """Store the result of a signature comparison"""
    comparison_id = str(uuid.uuid4())
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO comparison_results (
            id, signature1_id, signature2_id,
            similarity_score, comparison_params, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            comparison_id,
            signature1_id,
            signature2_id,
            similarity_score,
            pickle.dumps(comparison_params) if comparison_params else None,
            datetime.utcnow()
        ))
        
        conn.commit()
        return comparison_id

def get_comparison_history(
    signature_id: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get comparison history for a signature"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM comparison_results
        WHERE signature1_id = ? OR signature2_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        ''', (signature_id, signature_id, limit))
        
        results = []
        for row in cursor:
            results.append({
                'id': row['id'],
                'signature1_id': row['signature1_id'],
                'signature2_id': row['signature2_id'],
                'similarity_score': row['similarity_score'],
                'created_at': row['created_at']
            })
        
        return results

def compare_features(
    features1: Dict[str, np.ndarray],
    features2: Dict[str, np.ndarray],
    params: ComparisonParams
) -> float:
    """Calculate similarity score between two sets of features"""
    total_weight = params.hog_weight + params.contour_weight + params.density_weight
    if total_weight == 0:
        return 0.0
    
    similarity = 0.0
    
    # HOG features comparison
    if params.hog_weight > 0:
        hog_sim = cosine_similarity(
            features1['hog'].reshape(1, -1),
            features2['hog'].reshape(1, -1)
        )[0][0]
        similarity += hog_sim * params.hog_weight
    
    # Contour features comparison
    if params.contour_weight > 0:
        # Pad arrays to same length
        max_len = max(len(features1['contour']), len(features2['contour']))
        cont1 = np.pad(features1['contour'], (0, max_len - len(features1['contour'])))
        cont2 = np.pad(features2['contour'], (0, max_len - len(features2['contour'])))
        
        contour_sim = cosine_similarity(
            cont1.reshape(1, -1),
            cont2.reshape(1, -1)
        )[0][0]
        similarity += contour_sim * params.contour_weight
    
    # Density features comparison
    if params.density_weight > 0:
        density_sim = cosine_similarity(
            features1['density'].reshape(1, -1),
            features2['density'].reshape(1, -1)
        )[0][0]
        similarity += density_sim * params.density_weight
    
    return similarity / total_weight
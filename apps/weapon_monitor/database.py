import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

DB_PATH = Path(__file__).parent / "monitor.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inicializa la base de datos con las tablas necesarias."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Tabla de cámaras
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence_threshold REAL DEFAULT 0.85,
            cooldown INTEGER DEFAULT 10,
            consecutive_frames INTEGER DEFAULT 5,
            enabled BOOLEAN DEFAULT 1
        )
    ''')
    
    # Tabla de alarmas
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            weapon_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT,
            crop_path TEXT,
            FOREIGN KEY (camera_id) REFERENCES cameras (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# --- Operaciones de Cámaras ---

def add_camera(name: str, source: str, confidence: float = 0.85, cooldown: int = 10, consecutive_frames: int = 5) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO cameras (name, source, confidence_threshold, cooldown, consecutive_frames) VALUES (?, ?, ?, ?, ?)',
        (name, source, confidence, cooldown, consecutive_frames)
    )
    conn.commit()
    cam_id = cursor.lastrowid
    conn.close()
    return cam_id

def get_cameras(only_enabled: bool = True) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    query = 'SELECT * FROM cameras'
    if only_enabled:
        query += ' WHERE enabled = 1'
    
    cameras = conn.execute(query).fetchall()
    conn.close()
    return [dict(cam) for cam in cameras]

def get_camera(camera_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cam = conn.execute('SELECT * FROM cameras WHERE id = ?', (camera_id,)).fetchone()
    conn.close()
    return dict(cam) if cam else None

def update_camera(camera_id: int, **kwargs):
    conn = get_db_connection()
    # Construir query dinámica
    fields = []
    values = []
    for k, v in kwargs.items():
        fields.append(f"{k} = ?")
        values.append(v)
    
    values.append(camera_id)
    query = f"UPDATE cameras SET {', '.join(fields)} WHERE id = ?"
    
    conn.execute(query, values)
    conn.commit()
    conn.close()

def delete_camera(camera_id: int):
    conn = get_db_connection()
    conn.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))
    conn.commit()
    conn.close()

# --- Operaciones de Alarmas ---

def add_alarm(camera_id: int, weapon_type: str, confidence: float, image_path: str, crop_path: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO alarms (camera_id, weapon_type, confidence, image_path, crop_path) VALUES (?, ?, ?, ?, ?)',
        (camera_id, weapon_type, confidence, image_path, crop_path)
    )
    conn.commit()
    alarm_id = cursor.lastrowid
    conn.close()
    return alarm_id

def get_alarms(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    alarms = conn.execute('''
        SELECT a.*, c.name as camera_name 
        FROM alarms a 
        LEFT JOIN cameras c ON a.camera_id = c.id 
        ORDER BY timestamp DESC LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()
    return [dict(a) for a in alarms]

def get_latest_alarm_timestamp() -> Optional[str]:
    conn = get_db_connection()
    res = conn.execute('SELECT MAX(timestamp) as last_ts FROM alarms').fetchone()
    conn.close()
    return res['last_ts'] if res else None

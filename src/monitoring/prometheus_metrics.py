"""
═══════════════════════════════════════════════════════════════════════════════
🎯 PROMETHEUS METRICS - Export de métriques MLOps
═══════════════════════════════════════════════════════════════════════════════
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest #make_asgi_app  #generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import os

# ═══════════════════════════════════════════════════════════════════════════
# 📊 MÉTRIQUES CUSTOM - Spécifiques au modèle CV cats/dogs
# ═══════════════════════════════════════════════════════════════════════════

database_status = Gauge(
    'cv_database_connected',
    'Database connection status (1=connected, 0=disconnected)'
)

# Exercice 1 :
inference_time_histogram = Histogram(
    'cv_inference_time_seconds',
    'Temps d\'inférence en secondes',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 SETUP - Configuration de l'instrumentation Prometheus
# ═══════════════════════════════════════════════════════════════════════════


# Version origine
def setup_prometheus(app):
    """
    Configure Prometheus pour FastAPI
    Compatible avec l'API existante V2
    """
    if os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true':
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        print("✅ Prometheus metrics enabled at /metrics")
    else:
        print("ℹ️  Prometheus metrics disabled")

'''
# Version pour exposition toutes métriques
def setup_prometheus(app):
    """
    Configure Prometheus pour FastAPI
    Expose TOUTES les métriques (HTTP auto + custom)
    """
    if os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true':
        # 1. Instrumenter FastAPI (ajoute métriques HTTP)
        Instrumentator().instrument(app)
        
        # 2. Créer endpoint /metrics qui expose TOUT
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
        
        print("✅ Prometheus metrics enabled at /metrics")
    else:
        print("ℹ️  Prometheus metrics disabled")
'''
# ═══════════════════════════════════════════════════════════════════════════
# 📝 HELPERS - Fonctions de tracking appelées par l'API
# ═══════════════════════════════════════════════════════════════════════════

def update_db_status(is_connected: bool):
    """
    Met à jour le statut de la base de données
    """
    database_status.set(1 if is_connected else 0)

# Exercice 1 :
def track_inference_time(inference_time_ms: float):
    """Enregistre le temps d'inférence"""
    inference_time_histogram.observe(inference_time_ms / 1000)
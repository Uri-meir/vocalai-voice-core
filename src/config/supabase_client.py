from functools import lru_cache
import os
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)


def upload_call_recording(
    file_path: str,
    storage_path: str,
    max_size_mb: int = 50
) -> dict:
    """
    Upload recording to Supabase Storage with size guard.
    
    Args:
        file_path: Local temp file path
        storage_path: Target path in bucket (e.g. "slug/assistant/call.mp3")
        max_size_mb: Maximum file size to prevent memory issues
    
    Returns:
        {"storage_path": str, "signed_url": str}
    
    Raises:
        ValueError: If file exceeds max_size_mb
    """
    supabase = get_supabase_client()
    bucket_name = "call-recordings"
    
    # Note: storage_path is now simple (assistant_id/call_sid.mp3) - always URL-safe
    logger.info(f"📦 Uploading to: {storage_path}")
    
    # Guard: check file size before loading into memory
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"Recording too large: {file_size_mb:.1f}MB > {max_size_mb}MB")
    
    with open(file_path, "rb") as f:
        file_data = f.read()
    
    # Upload (upsert mode to handle retries)
    try:
        supabase.storage.from_(bucket_name).upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": "audio/mpeg", "upsert": "true"}
        )
        logger.info(f"☁️ Uploaded recording to Supabase: {storage_path}")
    except Exception as e:
        # If "already exists" and not upsert-capable, try update
        if "already exists" in str(e).lower():
            logger.info(f"📝 Recording exists, updating: {storage_path}")
            supabase.storage.from_(bucket_name).update(
                path=storage_path,
                file=file_data,
                file_options={"content-type": "audio/mpeg"}
            )
        else:
            raise
    
    # Generate signed URL (1 year expiry)
    signed_url_response = supabase.storage.from_(bucket_name).create_signed_url(
        path=storage_path,
        expires_in=31536000
    )
    
    return {
        "storage_path": storage_path,
        "signed_url": signed_url_response["signedURL"]
    }

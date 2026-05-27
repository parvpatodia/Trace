"""
Tigris Data — S3-compatible globally-distributed object storage.

Trace uses Tigris to persist uploaded history files and Apify-scraped
artifacts so they survive server restarts and are accessible from any
region without manual replication.

WHAT GOES IN TIGRIS:
  uploads/{upload_id}/{filename}     — raw user-uploaded history files
  artifacts/apify/{topic}/{run_id}   — raw Apify scrape results (JSON)
  graphs/{profile_id}/{ts}.json      — serialised CuriosityGraph snapshots

WHY TIGRIS OVER LOCAL DISK:
  - Tigris is globally distributed (fly.io PoPs) — files served from edge
  - S3-compatible API: zero new concepts, standard boto3 calls
  - Automatic geo-replication — Render deploys in multiple regions just work
  - Free tier covers demo usage; scales to production without code changes

GRACEFUL DEGRADATION:
  - TIGRIS_ACCESS_KEY_ID not set → TigrisStore.enabled = False
  - All methods silently return None/False when disabled
  - The rest of the pipeline uses local disk as the fallback path
"""
from __future__ import annotations

import io
import json
import logging
from typing import Any

_log = logging.getLogger(__name__)

try:
    import boto3  # type: ignore[import]
    from botocore.config import Config  # type: ignore[import]
    _BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore[assignment]
    Config = None  # type: ignore[assignment]
    _BOTO3_AVAILABLE = False


class TigrisStore:
    """Thin wrapper around the Tigris S3-compatible API.

    Uses synchronous boto3 under the hood (called from async contexts via
    run_in_executor where latency matters; for background artifact storage
    a blocking call is fine).
    """

    def __init__(
        self,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        bucket_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        from trace.config import get_settings
        s = get_settings()
        self._access_key_id = access_key_id or s.tigris_access_key_id
        self._secret_access_key = secret_access_key or s.tigris_secret_access_key
        self._bucket_name = bucket_name or s.tigris_bucket_name
        self._endpoint_url = endpoint_url or s.tigris_endpoint_url
        self._client: Any | None = None

    @property
    def enabled(self) -> bool:
        return (
            _BOTO3_AVAILABLE
            and bool(self._access_key_id)
            and bool(self._secret_access_key)
        )

    def _get_client(self) -> Any | None:
        if not self.enabled:
            return None
        if self._client is None:
            try:
                self._client = boto3.client(
                    "s3",
                    endpoint_url=self._endpoint_url,
                    aws_access_key_id=self._access_key_id,
                    aws_secret_access_key=self._secret_access_key,
                    config=Config(
                        signature_version="s3v4",
                        retries={"max_attempts": 3, "mode": "standard"},
                    ),
                    region_name="auto",
                )
                # Ensure bucket exists (idempotent).
                try:
                    self._client.head_bucket(Bucket=self._bucket_name)
                except Exception:
                    self._client.create_bucket(Bucket=self._bucket_name)
                _log.info("[Tigris] Connected to bucket=%r at %s", self._bucket_name, self._endpoint_url)
            except Exception as exc:
                _log.warning("[Tigris] Failed to initialise S3 client: %s", exc)
                self._client = None
        return self._client

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str | None:
        """Upload raw bytes to Tigris. Returns the S3 URI on success, None on failure."""
        client = self._get_client()
        if client is None:
            return None
        try:
            client.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            uri = f"s3://{self._bucket_name}/{key}"
            _log.info("[Tigris] Stored %d bytes at %s", len(data), uri)
            return uri
        except Exception as exc:
            _log.warning("[Tigris] put_bytes failed for key=%r: %s", key, exc)
            return None

    def put_json(self, key: str, obj: Any) -> str | None:
        """Serialise obj to JSON and upload to Tigris."""
        try:
            data = json.dumps(obj, default=str).encode()
        except Exception as exc:
            _log.warning("[Tigris] JSON serialisation failed: %s", exc)
            return None
        return self.put_bytes(key, data, content_type="application/json")

    def get_bytes(self, key: str) -> bytes | None:
        """Download raw bytes from Tigris. Returns None if missing or on error."""
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client.get_object(Bucket=self._bucket_name, Key=key)
            return resp["Body"].read()
        except Exception as exc:
            _log.debug("[Tigris] get_bytes key=%r: %s", key, exc)
            return None

    def get_json(self, key: str) -> Any | None:
        """Download and deserialise a JSON object from Tigris."""
        raw = self._get_bytes_safe(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as exc:
            _log.warning("[Tigris] JSON deserialisation failed for key=%r: %s", key, exc)
            return None

    def _get_bytes_safe(self, key: str) -> bytes | None:
        return self.get_bytes(key)

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str | None:
        """Generate a pre-signed GET URL valid for `expires_in` seconds."""
        client = self._get_client()
        if client is None:
            return None
        try:
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket_name, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception as exc:
            _log.warning("[Tigris] presigned URL failed for key=%r: %s", key, exc)
            return None

    def health(self) -> dict[str, Any]:
        """Return health/config status for the /health endpoint."""
        if not _BOTO3_AVAILABLE:
            return {"status": "unavailable", "reason": "boto3_not_installed"}
        if not self.enabled:
            return {"status": "disabled", "reason": "credentials_not_set"}
        client = self._get_client()
        if client is None:
            return {"status": "error", "reason": "client_init_failed"}
        return {
            "status": "ok",
            "bucket": self._bucket_name,
            "endpoint": self._endpoint_url,
        }

    def store_upload(self, upload_id: str, filename: str, data: bytes) -> str | None:
        """Persist an uploaded history file. Returns S3 URI or None."""
        key = f"uploads/{upload_id}/{filename}"
        ct = "application/json" if filename.endswith(".json") else "application/zip"
        return self.put_bytes(key, data, content_type=ct)

    def store_apify_artifact(self, topic: str, run_id: str, items: list[Any]) -> str | None:
        """Persist raw Apify scrape results for a topic run."""
        key = f"artifacts/apify/{topic.replace(' ', '_')}/{run_id}.json"
        return self.put_json(key, {"topic": topic, "run_id": run_id, "items": items})

    def store_graph_snapshot(self, profile_id: str, graph_json: str) -> str | None:
        """Archive a CuriosityGraph JSON snapshot."""
        import time
        ts = int(time.time())
        key = f"graphs/{profile_id}/{ts}.json"
        return self.put_bytes(key, graph_json.encode(), content_type="application/json")


# Module-level singleton — shared across the process.
_store: TigrisStore | None = None


def get_tigris_store() -> TigrisStore:
    global _store
    if _store is None:
        _store = TigrisStore()
    return _store

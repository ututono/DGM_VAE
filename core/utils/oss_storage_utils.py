"""
Storage utilities for OSS (Object Storage Service) operations.
"""

import os
import logging
import tarfile
import tempfile
import shutil
import zipfile
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from core.configs.values import CheckpointCompressionFormat as CFM, OSSConfigKeys

logger = logging.getLogger(__name__)


class StorageService(ABC):
    @abstractmethod
    def upload_checkpoint(self, checkpoint_path):
        pass

    @abstractmethod
    def download_checkpoint(self, checkpoint_name: str, local_path: Optional[str] = None):
        pass

    def list_checkpoints(self):
        pass

    def get_latest_checkpoint(self):
        pass


class R2StorageService(StorageService):
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket_name: str,
                 region_name: Optional[str] = "auto"):

        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket_name = bucket_name
        self._zip_format = CFM.GZ.value  # Default compression format
        self._region_name = region_name
        self._client = self._create_client()

    def _create_client(self):
        """
        Create a boto3 S3 client.
        """
        try:
            client = boto3.client(
                's3',
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                region_name=self._region_name
            )
            return client
        except NoCredentialsError:
            logger.error("Credentials not available for R2 storage service.")
            raise

    def upload_checkpoint(self, local_checkpoint_dir: Union[str, Path], timestamp: str,
                          metadata: Optional[dict] = None):
        local_checkpoint_dir = Path(local_checkpoint_dir)

        if not local_checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {local_checkpoint_dir}")

        compress_format = self._zip_format

        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix=compress_format, delete=False) as tmp_file:
            zip_path = tmp_file.name

        zip_path = Path(zip_path)

        try:
            # Compress checkpoint directory
            self._create_checkpoint_archive(local_checkpoint_dir, zip_path, zip_format=compress_format)

            # Upload to R2
            remote_key = f"checkpoints/{timestamp}.{self._zip_format}"
            self._upload_file(zip_path, remote_key, metadata)

            logger.info(
                f"Checkpoint uploaded successfully: Local:{local_checkpoint_dir} \n |\n |\n \\/\nRemote: {remote_key}")
            return remote_key

        finally:
            # Clean up temporary file
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete temporary file {zip_path}: {e}")

    @staticmethod
    def _create_checkpoint_archive(source_dir: Path, zip_path: Path, zip_format=None):
        try:
            if not source_dir.exists():
                raise FileNotFoundError(f"Source directory not found: {source_dir}")

            if zip_format not in ['gz', 'zip']:
                raise ValueError("Unsupported format. Use 'gz' or 'zip'.")

            if zip_format == CFM.ZIP:
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in source_dir.rglob('*'):
                        if file_path.is_file():
                            zipf.write(file_path, file_path.relative_to(source_dir))

            elif zip_format == CFM.GZ:
                with tarfile.open(zip_path, "w:gz") as tarf:
                    tarf.add(source_dir, arcname=source_dir.name)

            else:
                raise ValueError(f"Unsupported compression format: {zip_format}")

        except Exception as e:
            logger.error(f"Error creating checkpoint archive: {e}")
            return None

    def _upload_file(self, local_path: Path, remote_key: str, metadata: Optional[dict] = None):
        """
        Upload a file to R2 storage.
        """
        extra_args = {}
        if metadata:
            extra_args['Metadata'] = {str(k): str(v) for k, v in metadata.items()}

        try:
            self._client.upload_file(local_path, self._bucket_name, remote_key, ExtraArgs=extra_args)
        except ClientError as e:
            logger.error(f"Failed to upload file to R2: {e}")
            raise

    @staticmethod
    def _resolve_checkpoint_key(checkpoint_identifier: str) -> str:
        """
        Resolve the R2 key for a checkpoint.
        """
        if checkpoint_identifier.startswith("checkpoints/"):
            return checkpoint_identifier
        else:
            return f"checkpoints/{checkpoint_identifier}"

    def _download_file(self, remote_key: str, local_path: Path):
        """
        Download a file from R2 storage.
        """
        try:
            self._client.download_file(self._bucket_name, remote_key, str(local_path))
        except ClientError as e:
            logger.error(f"Failed to download file from R2: {e}")
            raise

    @staticmethod
    def _extract_checkpoint_archive(zip_path: Path, extract_to: Path) -> Path:
        """Extract checkpoint archive and return checkpoint directory path"""
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_to)

        # Find the extracted checkpoint directory
        extracted_items = list(extract_to.iterdir())
        checkpoint_dir = next((item for item in extracted_items if item.is_dir()), None)

        if checkpoint_dir is None:
            raise RuntimeError("Failed to find checkpoint directory in extracted archive")

        return checkpoint_dir

    def download_checkpoint(self, checkpoint_identifier: str, extract_to: Optional[str] = None):
        # Resolve R2 key
        remote_key = self._resolve_checkpoint_key(checkpoint_identifier)

        # Create extraction directory
        if extract_to is None:
            extract_to = Path(tempfile.mkdtemp(prefix='checkpoint_'))
        else:
            extract_to = Path(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)

        # Download and extract
        with tempfile.NamedTemporaryFile(suffix=self._zip_format, delete=False) as tmp_file:
            zip_path = tmp_file.name

        zip_path = Path(zip_path)
        try:
            self._download_file(remote_key, zip_path)
            checkpoint_dir = self._extract_checkpoint_archive(zip_path, extract_to)

            logger.info(f"Checkpoint downloaded and extracted: {checkpoint_dir}")
            return checkpoint_dir

        finally:
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete temporary file {zip_path}: {e}")

    @classmethod
    def is_remote_path(path: str) -> bool:
        """
        Check if the given path is a remote R2 path.
        """
        return (
                path.startswith("r2://")
                or path.startswith("s3://")
                or path.startswith("https://")
                or path.startswith("http://")
        ) and is_oss_enabled()

    def list_checkpoints(self):
        """List all available checkpoints in R2"""
        try:
            response = self._client.list_objects_v2(
                Bucket=self._bucket_name,
                Prefix='checkpoints/',
                Delimiter='/'
            )

            checkpoints = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith(self._zip_format):
                    timestamp = key.replace('checkpoints/', '').replace(self._zip_format, '')
                    checkpoints.append({
                        'timestamp': timestamp,
                        'key': key,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })

            return sorted(checkpoints, key=lambda x: x['last_modified'], reverse=True)

        except ClientError as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []


def is_oss_enabled() -> bool:
    """
    Check if OSS (Object Storage Service) is enabled based on the provided arguments.
    """
    load_dotenv()  # Load environment variables from .env file
    required_vars = [
        OSSConfigKeys.ACCESS_KEY.value,
        OSSConfigKeys.SECRET_KEY.value,
        OSSConfigKeys.BUCKET_NAME.value,
        OSSConfigKeys.ENDPOINT.value
    ]
    return all(os.getenv(var) for var in required_vars)


def get_storage_service(service_type) -> type(StorageService):
    """
    Factory function to get the appropriate storage service instance.
    """
    if service_type == 'r2':
        return R2StorageService
    else:
        raise ValueError(f"Unsupported storage service type: {service_type}")

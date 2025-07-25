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
import requests
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
        checkpoint_identifier = str(checkpoint_identifier)
        if checkpoint_identifier.startswith("checkpoints/"):
            return checkpoint_identifier
        else:
            if checkpoint_identifier.endswith('.gz'):
                return f"checkpoints/{checkpoint_identifier}"
            else:
                return f"checkpoints/{checkpoint_identifier}.gz"

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
        logger.info(f"Extracting archive: {zip_path} (size: {zip_path.stat().st_size} bytes)")

        try:
            # Check file magic numbers to determine actual format
            with open(zip_path, 'rb') as f:
                magic = f.read(4)

            # Reset file pointer
            is_gzip = magic[:2] == b'\x1f\x8b'  # gzip magic number
            is_zip = magic[:2] == b'PK'  # ZIP magic number

            logger.info(f"File magic: {magic.hex()}")
            logger.info(f"Detected format - gzip: {is_gzip}, zip: {is_zip}")

            if is_gzip or zip_path.suffix == '.gz' or zip_path.name.endswith('.tar.gz'):
                logger.info("Extracting as tar.gz archive...")
                with tarfile.open(zip_path, 'r:gz') as tarf:
                    def is_within_directory(directory, target):
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                        prefix = os.path.commonpath([abs_directory, abs_target])
                        return prefix == abs_directory

                    def safe_extract(tar, path):
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception(f"Attempted Path Traversal in Tar File: {member.name}")
                        tar.extractall(path)

                    safe_extract(tarf, extract_to)

            elif is_zip or zip_path.suffix == '.zip':
                logger.info("Extracting as ZIP archive...")
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    zipf.extractall(extract_to)
            else:
                logger.warning(f"Unknown format for {zip_path}, trying tar.gz first...")
                try:
                    with tarfile.open(zip_path, 'r:gz') as tarf:
                        tarf.extractall(extract_to)
                    logger.info("Successfully extracted as tar.gz")
                except Exception as e1:
                    logger.warning(f"tar.gz extraction failed: {e1}, trying ZIP...")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zipf:
                            zipf.extractall(extract_to)
                        logger.info("Successfully extracted as ZIP")
                    except Exception as e2:
                        raise Exception(f"Failed to extract as both tar.gz ({e1}) and ZIP ({e2})")

            # Find the extracted checkpoint directory
            extracted_items = list(extract_to.iterdir())
            logger.info(f"Extracted items: {[item.name for item in extracted_items]}")

            # Look for directory that looks like a checkpoint
            checkpoint_dir = None
            for item in extracted_items:
                if item.is_dir():
                    # Check if this directory contains checkpoint structure
                    if (item / 'model').exists() or (item / 'artifacts').exists():
                        checkpoint_dir = item
                        break

            if checkpoint_dir is None:
                # If no obvious checkpoint directory, take the first directory
                checkpoint_dir = next((item for item in extracted_items if item.is_dir()), None)

            if checkpoint_dir is None:
                raise RuntimeError(
                    f"Failed to find checkpoint directory in extracted archive. Found: {[item.name for item in extracted_items]}")

            logger.info(f"Found checkpoint directory: {checkpoint_dir}")
            return checkpoint_dir

        except Exception as e:
            logger.error(f"Failed to extract checkpoint archive: {e}")
            logger.error(f"Archive path: {zip_path}")
            logger.error(f"Archive exists: {zip_path.exists()}")
            if zip_path.exists():
                logger.error(f"Archive size: {zip_path.stat().st_size} bytes")

                # ======= DEBUG: Show first few bytes of file =======
                try:
                    with open(zip_path, 'rb') as f:
                        first_bytes = f.read(16)
                    logger.error(f"First 16 bytes: {first_bytes.hex()}")
                except Exception:
                    pass
            raise

    @classmethod
    def download_with_signed_url(cls, signed_url: str, extract_to: Optional[str] = None) -> Path:
        """
        Download and extract checkpoint using a signed URL.
        This method doesn't require R2 credentials.

        Args:
            signed_url: Presigned download URL
            extract_to: Directory to extract checkpoint (optional)

        Returns:
            Path to extracted checkpoint directory
        """
        # Create extraction directory
        if extract_to is None:
            extract_to = Path(tempfile.mkdtemp(prefix='checkpoint_'))
        else:
            extract_to = Path(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)

        # Download file using requests
        with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as tmp_file:
            zip_path = Path(tmp_file.name)

        try:
            logger.info(f"Downloading checkpoint from signed URL...")
            response = requests.get(signed_url, stream=True, timeout=300)
            response.raise_for_status()

            # Write downloaded content to temporary file
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract checkpoint
            checkpoint_dir = cls._extract_checkpoint_archive(zip_path, extract_to)

            logger.info(f"Checkpoint downloaded and extracted to: {checkpoint_dir}")
            return checkpoint_dir

        except requests.RequestException as e:
            logger.error(f"Failed to download from signed URL: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during download/extraction: {e}")
            raise
        finally:
            # Clean up temporary zip file
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete temporary file {zip_path}: {e}")


    def generate_signed_download_url(self, checkpoint_identifier: str, expires_in: int = 900) -> str:
        """
        Generate a presigned URL for downloading a checkpoint.

        @:param checkpoint_identifier: Checkpoint timestamp or key
        @:param    expires_in: URL expiration time in seconds (default: 15 minutes)

        :return: Presigned URL for downloading the checkpoint
        """
        remote_key = self._resolve_checkpoint_key(checkpoint_identifier)

        try:
            url = self._client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self._bucket_name, 'Key': remote_key},
                ExpiresIn=expires_in
            )

            logger.info(f"Generated signed URL for {checkpoint_identifier}, expires in {expires_in}s")
            return url

        except ClientError as e:
            logger.error(f"Failed to generate signed URL: {e}")
            raise

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
    def is_remote_path(cls, path_: str) -> bool:
        """
        Check if the given path is a remote R2 path.
        """
        return (
                path_.startswith("r2://")
                or path_.startswith("s3://")
                or path_.startswith("https://")
                or path_.startswith("http://")
        ) and is_oss_enabled()

    @classmethod
    def is_remote_path_or_checkpoint_id(cls, path_: str) -> tuple[bool, str]:
        """
        Check if the given path is a remote signed URL or checkpoint ID.
        :return: (is_remote, path_type) where path_type is 'signed_url', 'checkpoint_id', or 'local'
        """
        if (path_.startswith("https://") or path_.startswith("http://")) or cls.is_remote_path(path_):
            return True, "signed_url"
        elif not path_.startswith("/") and not path_.startswith("./") and not Path(path_).exists():
            # Looks like a checkpoint ID (e.g., "2025-01-18_14-30-25" or "latest")
            return True, "checkpoint_id"
        else:
            return False, "local"

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

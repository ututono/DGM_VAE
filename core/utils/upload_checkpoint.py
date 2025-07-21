import argparse
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pytimeparse import parse
from core.configs.values import OSSConfigKeys as OSK
from core.utils.oss_storage_utils import get_storage_service, is_oss_enabled

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


def extract_timestamp_from_path(checkpoint_path: Path) -> Optional[str]:
    """
    Extract timestamp from checkpoint path

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Extracted timestamp or None if not found
    """
    # Try to find timestamp pattern in path parts
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'

    for part in checkpoint_path.parts:
        if re.match(timestamp_pattern, part):
            return part

    # If no timestamp found in path, try the directory name
    dir_name = checkpoint_path.name
    if re.match(timestamp_pattern, dir_name):
        return dir_name

    return None


def validate_checkpoint_structure(checkpoint_path: Path) -> bool:
    """
    Validate that the checkpoint directory has the expected structure

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        True if structure is valid, False otherwise
    """
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
        return False

    if not checkpoint_path.is_dir():
        logger.error(f"Checkpoint path is not a directory: {checkpoint_path}")
        return False

    # Check for expected files/directories
    expected_items = ['model', 'artifacts']
    found_items = []

    for item in expected_items:
        item_path = checkpoint_path / item
        if item_path.exists():
            found_items.append(item)

    if not found_items:
        logger.warning(f"No expected checkpoint structure found in {checkpoint_path}")
        logger.warning("Expected items: model/, artifacts/")
        logger.warning(f"Found items: {list(checkpoint_path.iterdir())}")

        # Ask user if they want to continue anyway
        response = input("Continue upload anyway? (y/N): ")
        if response.lower() != 'y':
            return False

    logger.info(f"Found checkpoint structure: {found_items}")
    return True


def get_checkpoint_metadata(checkpoint_path: Path, timestamp: str) -> dict:
    """
    Extract metadata from checkpoint directory

    Args:
        checkpoint_path: Path to checkpoint directory
        timestamp: Timestamp identifier

    Returns:
        Dictionary containing metadata
    """
    metadata = {
        'upload_time': datetime.now().isoformat(),
        'original_path': str(checkpoint_path),
        'timestamp': timestamp
    }

    # Try to read args.json for additional metadata
    args_file = checkpoint_path / 'artifacts' / 'args.json'
    if args_file.exists():
        try:
            import json
            with open(args_file, 'r') as f:
                args_data = json.load(f)

            # Extract relevant information
            metadata.update({
                'model_type': args_data.get('model', 'unknown'),
                'dataset_names': ','.join(args_data.get('dataset_names', [])),
                'epochs': str(args_data.get('epochs', 'unknown')),
                'batch_size': str(args_data.get('batch_size', 'unknown')),
                'learning_rate': str(args_data.get('learning_rate', 'unknown'))
            })

            logger.info(f"Extracted metadata from args.json: model={metadata['model_type']}, "
                        f"datasets={metadata['dataset_names']}, epochs={metadata['epochs']}")

        except Exception as e:
            logger.warning(f"Could not read metadata from args.json: {e}")

    # Check model file size
    model_dir = checkpoint_path / 'model'
    if model_dir.exists():
        model_files = list(model_dir.glob('*.pth.tar'))
        if model_files:
            model_size = sum(f.stat().st_size for f in model_files)
            metadata['model_size_mb'] = str(round(model_size / (1024 * 1024), 2))

    return metadata

def convert_period_to_seconds(period: str) -> int:
    """
    Convert a time period string to seconds.

    """
    try:
        seconds = parse(period)
        if seconds is None:
            raise ValueError("Invalid time period format")
        return seconds
    except Exception as e:
        logger.error(f"Failed to parse time period '{period}': {e}")
        return 3600  # Default to 1 hour if parsing fails



def upload_checkpoint(
        checkpoint_path: str,
        custom_timestamp: Optional[str] = None,
        args = None,
        force: bool = False
) -> bool:
    """
    Upload checkpoint to R2 storage

    @param checkpoint_path: Path to checkpoint directory
    @param custom_timestamp: Custom timestamp to use instead of auto-detected
    @param force: Force upload even if validation fails
    @param args: Additional arguments (e.g. for metadata)
    @return: True if upload successful, False otherwise
    """

    # Check if R2 is configured
    if not is_oss_enabled():
        logger.error("OSS storage is not configured. Please check your .env file.")
        logger.error("Required variables: ENDPOINT_URL, ACCESS_KEY, SECRET_KEY, BUCKET_NAME")
        return False

    # Convert to Path object
    checkpoint_path = Path(checkpoint_path).resolve()

    # Validate checkpoint structure
    if not force and not validate_checkpoint_structure(checkpoint_path):
        return False

    # Determine timestamp
    if custom_timestamp:
        timestamp = custom_timestamp
        logger.info(f"Using custom timestamp: {timestamp}")
    else:
        timestamp = extract_timestamp_from_path(checkpoint_path)
        if not timestamp:
            # Generate timestamp from current time
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            logger.warning(f"Could not extract timestamp from path, using current time: {timestamp}")
        else:
            logger.info(f"Extracted timestamp from path: {timestamp}")

    # Get R2 storage instance
    StorageService = get_storage_service(args.oss_type)
    oss_service = StorageService(
        endpoint_url=args.oss_endpoint_url,
        access_key=args.oss_access_key,
        secret_key=args.oss_secret_key,
        bucket_name=args.oss_bucket_name,
        region_name=args.oss_region_name
    )

    if not oss_service:
        logger.error("Failed to initialize R2 storage")
        return False

    # Check if checkpoint already exists
    existing_checkpoints = oss_service.list_checkpoints()
    existing_timestamps = [cp['timestamp'] for cp in existing_checkpoints]

    if timestamp in existing_timestamps:
        logger.warning(f"Checkpoint with timestamp '{timestamp}' already exists in R2")
        if not force:
            response = input("Overwrite existing checkpoint? (y/N): ")
            if response.lower() != 'y':
                logger.info("Upload cancelled")
                return False

    # Get metadata
    metadata = get_checkpoint_metadata(checkpoint_path, timestamp)

    # Display upload summary
    logger.info("=" * 60)
    logger.info("UPLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Local path: {checkpoint_path}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Model type: {metadata.get('model_type', 'unknown')}")
    logger.info(f"Datasets: {metadata.get('dataset_names', 'unknown')}")
    if 'model_size_mb' in metadata:
        logger.info(f"Model size: {metadata['model_size_mb']} MB")
    logger.info("=" * 60)

    # Confirm upload
    if not force:
        response = input("Proceed with upload? (y/N): ")
        if response.lower() != 'y':
            logger.info("Upload cancelled")
            return False

    # Perform upload
    try:
        logger.info("Starting upload to R2...")
        remote_key = oss_service.upload_checkpoint(
            local_checkpoint_dir=checkpoint_path,
            timestamp=timestamp,
            metadata=metadata
        )

        # create pre-signed URL if enabled
        expires_in = convert_period_to_seconds(args.expires_in)
        presigned_url = None
        if args.enable_presigned:
            presigned_url = oss_service.generate_signed_download_url(
                checkpoint_identifier=remote_key,
                expires_in=expires_in,
            )

        logger.info("=" * 60)
        logger.info("UPLOAD SUCCESSFUL")
        logger.info("=" * 60)
        logger.info(f"R2 key: {remote_key}")
        logger.info(f"Presigned URL: {presigned_url if presigned_url else 'Not generated'}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False

def argparse_args():
    arg_parser = argparse.ArgumentParser(
        description="Upload a checkpoint to a remote storage service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          python upload_checkpoint.py outputs/2025-01-18_14-30-25
          python upload_checkpoint.py /path/to/checkpoint --timestamp my-experiment
          python upload_checkpoint.py outputs/2025-01-18_14-30-25 --force
          python upload_checkpoint.py --list_local
          python upload_checkpoint.py --list_remote
                """
    )
    arg_parser.add_argument('checkpoint_path', nargs='?', type=str,
                            help='Path to checkpoint directory to upload')
    arg_parser.add_argument('--timestamp', type=str,
                            help='Custom timestamp/name for the checkpoint')
    arg_parser.add_argument('--force', action='store_true',
                            help='Force upload without confirmation prompts')
    arg_parser.add_argument('--list_local', action='store_true',
                            help='List available local checkpoints')
    arg_parser.add_argument('--list_remote', action='store_true',
                            help='List available remote checkpoints in R2')
    arg_parser.add_argument('--oss_type', type=str, default='r2',
                            help='Type of OSS storage service (default: r2)')
    arg_parser.add_argument('--expires_in', type=str, default='1h',
                            help='Expiration time for pre-signed URLs (default: 1 hour, format: 1h, 30m, etc.)')
    arg_parser.add_argument('--enable_presigned', action='store_true',
                            help='Enable generation of pre-signed URLs for download')

    args = arg_parser.parse_args()

    return args

def load_oss_config_from_env(args):
    """
    Load OSS configuration from environment variables.
    """
    import dotenv

    dotenv.load_dotenv()

    args.oss_endpoint_url = os.getenv(OSK.ENDPOINT.value)
    args.oss_access_key = os.getenv(OSK.ACCESS_KEY.value)
    args.oss_secret_key = os.getenv(OSK.SECRET_KEY.value)
    args.oss_bucket_name = os.getenv(OSK.BUCKET_NAME.value)
    args.oss_region_name = os.getenv(OSK.REGION_NAME.value)

def main():
    args = argparse_args()
    load_oss_config_from_env(args)

    if args.list_local:
        return

    if args.list_local:
        return

    print(f"{convert_period_to_seconds(args.expires_in)} seconds")
    upload_checkpoint(
        checkpoint_path=args.checkpoint_path,
        custom_timestamp=args.timestamp,
        force=args.force,
        args=args
    )


if __name__ == '__main__':
    main()

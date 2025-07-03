import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, DefaultDict, Any

import torch

logger = logging.getLogger(__name__)

@dataclass
class TrainingRecord:
    train_count: int = 0
    timestamp: str = ""
    args: dict = None  # Arguments for the current training run
    metrics: List[dict] = None
    others: dict = None  # Additional information, e.g., model state, optimizer state

    def __post_init__(self):
        if self.args is None:
            self.args = {}
        if self.metrics is None:
            self.metrics = []
        if self.others is None:
            self.others = {}

class TrainingRecordsManager:
    def __init__(self):
        self.records: List[TrainingRecord] = []
        self.latest_train_count = 0

    def add_record(self, args, metrics=None, others=None, timestamp=None) -> TrainingRecord:
        """Add a new training record"""
        self.latest_train_count += 1

        record = TrainingRecord(
            train_count=self.latest_train_count,
            timestamp=timestamp or datetime.now().isoformat(),
            args=args if isinstance(args, dict) else vars(args),
            metrics=metrics or [],
            others=others or {}
        )

        self.records.append(record)
        return record

    def get_latest_record(self) -> TrainingRecord:
        """Get the most recent training record"""
        if not self.records:
            return None
        return self.records[-1]

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'records': [asdict(record) for record in self.records],
            'latest_train_count': self.latest_train_count
        }

    def from_dict(self, data):
        """Load from dictionary"""
        self.latest_train_count = data.get('latest_train_count', 0)
        self.records = []

        for record_data in data.get('records', []):
            record = TrainingRecord(**record_data)
            self.records.append(record)

    def save_to_file(self, filepath):
        """Save records to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_file(self, filepath):
        """Load records from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.from_dict(data)


class ConfigCompatibilityChecker:
    """Check if the current configuration is compatible with the training records."""
    CRITICAL_PARAMS = [
        'image_size', 'latent_dim'
    ]

    WARNING_PARAMS = [
        'learning_rate', 'optimizer', 'weight_decay', 'batch_size'
    ]

    @classmethod
    def check_compatibility(cls, record_1: TrainingRecord, record_2: TrainingRecord) -> Dict[str, Any]:
        """
        Check if two training records are compatible based on critical parameters.
        Returns True if compatible, False otherwise.
        """
        result = {
            'compatible': True,
            'critical_issues': [],
            'warnings': [],
            'recommendation': 'proceed'
        }

        args_1 = record_1.args
        args_2 = record_2.args

        # Check critical parameters
        for param in cls.CRITICAL_PARAMS:
            val_1 = args_1.get(param)
            val_2 = args_2.get(param)

            if val_1 != val_2:
                result['critical_issues'].append({
                    'parameter': param,
                    'current': val_1,
                    'saved': val_2,
                    'message': f"Critical parameter mismatch: {param}"
                })
                result['compatible'] = False

        # Check warning parameters
        for param in cls.WARNING_PARAMS:
            val_1 = args_1.get(param)
            val_2 = args_2.get(param)

            if val_1 != val_2:
                result['warnings'].append({
                    'parameter': param,
                    'current': val_1,
                    'saved': val_2,
                    'message': f"Parameter changed: {param}"
                })

        # Set recommendation
        if not result['compatible']:
            result['recommendation'] = 'incompatible'
        elif result['warnings']:
            result['recommendation'] = 'proceed_with_caution'
        else:
            result['recommendation'] = 'proceed'

        return result

    @classmethod
    def print_compatibility_report(cls, compatibility_result: Dict[str, Any]):
        """
        Print a report of the compatibility check between two training records.
        :param compatibility_result:
        :return:
        """
        message = "\n" + "="*60 + "\n" + "TRAINING COMPATIBILITY CHECK\n" + "="*60

        if compatibility_result['compatible']:
            message += "\n" + "Compatibility check passed"
        else:
            message += "\n" + "Compatibility check failed"

        if compatibility_result['critical_issues']:
            message += "\n" + "CRITICAL ISSUES:"
            for issue in compatibility_result['critical_issues']:
                message += f"\n  • {issue['parameter']}: {issue['message']}"
                message += f"\n    Current: {issue['current']}"
                message += f"\n    Saved:   {issue['saved']}"

        if compatibility_result['warnings']:
            message += "\n" + "WARNINGS:"
            for warning in compatibility_result['warnings']:
                message += f"\n  <UNK> {warning['parameter']}: {warning['message']}"
                message += f"\n    Current: {warning['current']}"
                message += f"\n    Saved:   {warning['saved']}"

        message += "\n" + "="*60
        message += f"\n RECOMMENDATION: {compatibility_result['recommendation'].upper()}"
        message += "\n" + "="*60 + "\n"

        logger.info(message)


def save_checkpoint(model, save_dir, args, metrics=None, records_manager=None, ckpt_name='model.pth.tar'):
    """
    Save the model and its training records to a checkpoint file.
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir() and save_dir.suffix == '.pth.tar':
        save_path = save_dir
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir.joinpath(ckpt_name)

    # Create or use existing records manager
    if records_manager is None:
        records_manager = TrainingRecordsManager()

    # Add new record
    _ = records_manager.add_record(args, metrics)

    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'records': records_manager.to_dict()
    }

    # Save model checkpoint
    torch.save(checkpoint, save_path)

    # Save records as separate JSON file for the sake of readability
    records_path = save_dir.joinpath('records.json')
    records_manager.save_to_file(records_path)

    logger.info(f"Checkpoint saved to {save_path}")
    logger.info(f"Records saved to {records_path}")

    return records_manager

def load_checkpoint(model, current_args, load_dir, force_continue:bool=False, ckpt_name='model.pth.tar'):
    """
    Load model checkpoint and check compatibility
    """
    load_dir = Path(load_dir)
    if not load_dir.is_dir() and load_dir.suffix == '.pth.tar':
        load_path = load_dir
    else:
        load_path = load_dir.joinpath(ckpt_name)


    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint file {load_path} does not exist.")

    checkpoint = torch.load(load_path, map_location='cpu')

    # Load training records
    records_manager = TrainingRecordsManager()
    if 'records' not in checkpoint:
        logger.warning(f"Checkpoint file {load_path} does not contain any records.")
    else:
        records_manager.from_dict(checkpoint['records'])

    if records_manager.records:
        latest_record = records_manager.get_latest_record()

        # Create current record for comparison
        current_record = TrainingRecord(
            args=current_args if isinstance(current_args, dict) else vars(current_args)
        )

        # Check compatibility
        compatibility = ConfigCompatibilityChecker.check_compatibility(
            current_record, latest_record
        )

        # Print report
        ConfigCompatibilityChecker.print_compatibility_report(compatibility)

        # Print report
        ConfigCompatibilityChecker.print_compatibility_report(compatibility)

        # Handle incompatibility
        if not compatibility['compatible'] and not force_continue:
            raise ValueError(
                "Current configuration is incompatible with saved model. "
                "Use --force_continue to override this check."
            )

        if compatibility['warnings'] and not force_continue:
            response = input("Configuration differs from saved model. Continue? (y/N): ")
            if response.lower() != 'y':
                raise ValueError("Training cancelled by user")

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")

    return records_manager


def update_metrics(records_manager, metrics):
    """Update metrics for the latest record"""
    if records_manager.records:
        latest_record = records_manager.get_latest_record()
        if latest_record.metrics is None:
            latest_record.metrics = []
        latest_record.metrics.append(metrics)



import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from models.trial_filters import (
    TrialFilter, TrainingFilter,
    WideNarrowFilter,
    WideNarrowCongruencyFilter,
    ViewSpecificFilter,
    TrialSplitter,
    WideNarrowTrainingFilter,
    ViewSpecificWideNarrowTrainingFilter,
    IndividualMiniblockTrainingFilter
)
from sklearn.utils import Bunch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Global configuration for GLM models"""
    include_button_press: bool = True
    event_of_interest: Optional[int] = None
    duration_end_event: Optional[int] = None


class GLMModel:
    """Main GLM model that composes different filters"""
    
    def __init__(self, name: str, test_filter: Optional[TrialFilter] = None, 
                 training_filter: Optional[TrainingFilter] = None, config: Optional[ModelConfig] = None):
        self.name = name
        self.test_filter = test_filter
        self.training_filter = training_filter
        self.config = config or ModelConfig()
    
    def specify_model(self, events_file: Union[str, Path], behavior: Optional[pd.DataFrame] = None) -> Bunch:
        """Main entry point for model specification"""
        events = self._load_events(events_file)
        task = self._extract_task_from_filename(str(events_file))
        
        if task == 'train':
            return self._specify_training_model(events)
        elif task == 'test':
            if behavior is None:
                raise ValueError("Behavioral data required for test task")
            return self._specify_test_model(events, behavior)
        else:
            raise ValueError(f"Unknown task type: {task}")
    
    def _load_events(self, events_file: Union[str, Path]) -> pd.DataFrame:
        """Load and validate events file"""
        events = pd.read_csv(events_file, sep='\t')
        logger.info(f"Loaded {len(events)} events from {events_file}")
        return events
    
    def _extract_task_from_filename(self, filename: str) -> str:
        """Extract task name from filename"""
        if 'task-train' in filename:
            return 'train'
        elif 'task-test' in filename:
            return 'test'
        else:
            raise ValueError(f"Cannot determine task from filename: {filename}")
    
    def _specify_training_model(self, events: pd.DataFrame) -> Bunch:
        """Training model using the training filter"""
        if self.training_filter is None:
            raise NotImplementedError(f"Training model not configured for {self.name}")
        
        # Process events into mini-blocks and get conditions
        condition_data = self.training_filter.process_events(events)
        conditions = list(condition_data.keys())
        
        onsets = []
        durations = []
        
        for condition in conditions:
            onset_list, duration_list = condition_data[condition]
            onsets.append(onset_list)
            durations.append(duration_list)
        
        # Add button press if configured
        self._add_button_press_regressor(events, conditions, onsets, durations)
        
        self._validate_model_spec(conditions, onsets, durations)
        return Bunch(conditions=conditions, onsets=onsets, durations=durations)
    
    def _specify_test_model(self, events: pd.DataFrame, behavior: pd.DataFrame) -> Bunch:
        """Test model using the test filter"""
        if self.test_filter is None:
            raise NotImplementedError(f"Test model not configured for {self.name}")
            
        # Get conditions and trial assignments from the filter
        trial_assignments = self.test_filter.filter_trials(behavior)
        conditions = list(trial_assignments.keys())
        
        onsets = []
        durations = []
        
        # Get onsets and durations for each condition
        for condition in conditions:
            trial_indices = trial_assignments[condition]
            onsets.append(self._get_trial_onsets(events, trial_indices))
            durations.append(self._calculate_durations(events, trial_indices))
        
        # Add button press if configured
        self._add_button_press_regressor(events, conditions, onsets, durations)
        
        self._validate_model_spec(conditions, onsets, durations)
        return Bunch(conditions=conditions, onsets=onsets, durations=durations)
    
    def _get_trial_onsets(self, events: pd.DataFrame, trial_indices: List[int]) -> List[float]:
        """Get onset times for specified trials"""
        trial_events = events[
            events['trial_no'].isin(trial_indices) & 
            (events['event_no'] == self.config.event_of_interest)
        ]
        return list(trial_events['onset'])
    
    def _calculate_durations(self, events: pd.DataFrame, trial_indices: List[int]) -> List[float]:
        """Calculate event durations from start to end event"""
        durations = []
        start_event = self.config.event_of_interest
        end_event = self.config.duration_end_event
        
        for trial_idx in trial_indices:
            trial_events = events[events['trial_no'] == trial_idx]
            
            start_time = trial_events[trial_events['event_no'] == start_event]['onset'].values
            end_time = trial_events[trial_events['event_no'] == end_event]['onset'].values
            
            if len(start_time) > 0 and len(end_time) > 0:
                duration = end_time[0] - start_time[0]
                durations.append(max(duration, 0.0))
            else:
                logger.warning(f"Missing events for trial {trial_idx}")
                durations.append(0.0)
        
        return durations
    
    def _add_button_press_regressor(self, events: pd.DataFrame, conditions: List[str], 
                                   onsets: List[List], durations: List[List]):
        """Add button press regressor if present and configured"""
        if self.config.include_button_press and events['trial_type'].str.contains('buttonpress').any():
            conditions.append('buttonpress')
            button_events = events[events['trial_type'] == 'buttonpress']
            onsets.append(list(button_events['onset']))
            durations.append(list(button_events['duration']))
    
    def _validate_model_spec(self, conditions: List[str], onsets: List[List], durations: List[List]):
        """Validate model specification"""
        if not (len(conditions) == len(onsets) == len(durations)):
            raise ValueError(f"Mismatch in lengths: conditions={len(conditions)}, "
                           f"onsets={len(onsets)}, durations={len(durations)}")
        
        for i, (cond, onset, duration) in enumerate(zip(conditions, onsets, durations)):
            if len(onset) != len(duration):
                raise ValueError(f"Condition '{cond}' has mismatched onset/duration lengths")
            logger.info(f"Condition '{cond}': {len(onset)} events")
    
    def generate_contrasts(self, task: str = 'test') -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        """Generate contrasts from the appropriate filter"""
        if task == 'test' and self.test_filter:
            return self.test_filter.generate_contrasts()
        elif task == 'train' and self.training_filter:
            return self.training_filter.generate_contrasts()
        else:
            return None
        
# Factory functions for models

def create_widenarrow_training_model() -> GLMModel:
    """Wide/narrow training model (Experiment 2)"""
    training_filter = WideNarrowTrainingFilter()
    return GLMModel("wide_narrow_training", training_filter=training_filter)

def create_viewspecific_widenarrow_training_model() -> GLMModel:
    """View-specific wide/narrow training model (Experiment 1)"""
    training_filter = ViewSpecificWideNarrowTrainingFilter()
    return GLMModel("viewspecific_widenarrow_training", training_filter=training_filter)

def create_individual_miniblock_training_model(viewspecific: bool = True) -> GLMModel:
    """Individual mini-block training model"""
    if viewspecific:
        base_filter = ViewSpecificWideNarrowTrainingFilter()
    else:
        base_filter = WideNarrowTrainingFilter()
    
    training_filter = IndividualMiniblockTrainingFilter(base_filter)
    return GLMModel("individual_miniblock_training", training_filter=training_filter)


def create_widenarrow_test_model() -> GLMModel:
    """Wide/Narrow test model (Experiment 2)"""
    test_filter = WideNarrowFilter()
    training_filter = WideNarrowTrainingFilter()
    return GLMModel("wide_narrow", test_filter=test_filter, training_filter=training_filter)

def create_exp1_model() -> GLMModel:
    """Full model with view-specificity and trial splitting"""
    base_filter = WideNarrowCongruencyFilter()
    view_filter = ViewSpecificFilter(base_filter)
    test_splitter = TrialSplitter(view_filter, n_splits=3, 
                                 split_conditions=['A_wide_congruent', 'A_narrow_congruent',
                                                 'B_wide_congruent', 'B_narrow_congruent'])
    training_filter = IndividualMiniblockTrainingFilter(ViewSpecificWideNarrowTrainingFilter())
    return GLMModel("full_model", test_filter=test_splitter, training_filter=training_filter)
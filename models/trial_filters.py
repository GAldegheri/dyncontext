import pandas as pd
from nipype.interfaces.base import Bunch
from typing import Dict, List, Tuple, Optional, Union, Protocol
import random

class TrialFilter(Protocol):
    """Protocol for trial filtering components"""
    
    def get_conditions(self) -> List[str]:
        """Return list of condition names this filter creates"""
        ...
    
    def filter_trials(self, behavior: pd.DataFrame) -> Dict[str, List[int]]:
        """Return dict mapping condition names to trial indices"""
        ...
    
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        """Generate contrasts for this filter's conditions (optional)"""
        ...
        
class WideNarrowFilter:
    """Filter trials by proximal shape (wide/narrow), for experiment 2"""
    
    def get_conditions(self) -> List[str]:
        return ['wide', 'narrow']
    
    def filter_trials(self, behavior: pd.DataFrame) -> Dict[str, List[int]]:
        # Wide: (A,30) or (B,90)
        wide_mask = (
            ((behavior['initpos'] == 1) & (behavior['finalview'] == 30)) |
            ((behavior['initpos'] == 2) & (behavior['finalview'] == 90))
        )
        
        # Narrow: (A,90) or (B,30)  
        narrow_mask = (
            ((behavior['initpos'] == 1) & (behavior['finalview'] == 90)) |
            ((behavior['initpos'] == 2) & (behavior['finalview'] == 30))
        )
        
        # Exclude target reappearance trials
        if 'target' in behavior.columns:
            wide_mask = wide_mask & (behavior['target'] == 0)
            narrow_mask = narrow_mask & (behavior['target'] == 0)
            
        return {
            'wide': behavior.index[wide_mask].tolist(),
            'narrow': behavior.index[narrow_mask].tolist()
        }
        
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        return None

class CongruencyFilter:
    """Filter trials by congruency (congruent/incongruent) in experiment 1"""
    
    def get_conditions(self) -> List[str]:
        return ['congruent', 'incongruent']
    
    def filter_trials(self, behavior: pd.DataFrame) -> Dict[str, List[int]]:
        congruent_mask = behavior['consistent'] == 1
        incongruent_mask = behavior['consistent'] == 0
        
        return {
            'congruent': behavior.index[congruent_mask].tolist(),
            'incongruent': behavior.index[incongruent_mask].tolist()
        }
        
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        return [
            ('congruent>incongruent', 'T', ['congruent', 'incongruent'], [1.0, -1.0]),
            ('incongruent>congruent', 'T', ['incongruent', 'congruent'], [1.0, -1.0])
        ]

class WideNarrowCongruencyFilter:
    """Filter trials by both shape and congruency (4 conditions), for experiment 1"""
    
    def get_conditions(self) -> List[str]:
        return ['wide_congruent', 'narrow_congruent', 'wide_incongruent', 'narrow_incongruent']
    
    def filter_trials(self, behavior: pd.DataFrame) -> Dict[str, List[int]]:
        conditions = {}
        
        # Wide congruent: (A,30,consistent) or (B,90,consistent)
        wide_congruent = (
            ((behavior['initpos'] == 1) & (behavior['finalview'] == 30) & (behavior['consistent'] == 1)) |
            ((behavior['initpos'] == 2) & (behavior['finalview'] == 90) & (behavior['consistent'] == 1))
        )
        
        # Narrow congruent: (A,90,consistent) or (B,30,consistent)
        narrow_congruent = (
            ((behavior['initpos'] == 1) & (behavior['finalview'] == 90) & (behavior['consistent'] == 1)) |
            ((behavior['initpos'] == 2) & (behavior['finalview'] == 30) & (behavior['consistent'] == 1))
        )
        
        # Wide incongruent: (A,90,inconsistent) or (B,30,inconsistent)
        wide_incongruent = (
            ((behavior['initpos'] == 1) & (behavior['finalview'] == 90) & (behavior['consistent'] == 0)) |
            ((behavior['initpos'] == 2) & (behavior['finalview'] == 30) & (behavior['consistent'] == 0))
        )
        
        # Narrow incongruent: (A,30,inconsistent) or (B,90,inconsistent)  
        narrow_incongruent = (
            ((behavior['initpos'] == 1) & (behavior['finalview'] == 30) & (behavior['consistent'] == 0)) |
            ((behavior['initpos'] == 2) & (behavior['finalview'] == 90) & (behavior['consistent'] == 0))
        )
        
        # Exclude target reappearance trials
        if 'target' in behavior.columns:
            target_mask = behavior['target'] == 0
            wide_congruent = wide_congruent & target_mask
            narrow_congruent = narrow_congruent & target_mask
            wide_incongruent = wide_incongruent & target_mask
            narrow_incongruent = narrow_incongruent & target_mask
        
        return {
            'wide_congruent': behavior.index[wide_congruent].tolist(),
            'narrow_congruent': behavior.index[narrow_congruent].tolist(),
            'wide_incongruent': behavior.index[wide_incongruent].tolist(),
            'narrow_incongruent': behavior.index[narrow_incongruent].tolist()
        }
        
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        return None
    

class ViewSpecificFilter:
    """Filter trials by view (A/B) for view-specific analysis"""
    
    def __init__(self, base_filter: TrialFilter):
        self.base_filter = base_filter
        
    def get_conditions(self) -> List[str]:
        base_conditions = self.base_filter.get_conditions()
        return [f"A_{cond}" for cond in base_conditions] + [f"B_{cond}" for cond in base_conditions]
    
    def filter_trials(self, behavior: pd.DataFrame) -> Dict[str, List[int]]:
        base_trials = self.base_filter.filter_trials(behavior)
        view_specific_trials = {}
        
        for condition, trial_indices in base_trials.items():
            # Split by view A (initpos=1) and B (initpos=2)
            condition_data = behavior.loc[trial_indices]
            
            view_a_mask = condition_data['initpos'] == 1
            view_b_mask = condition_data['initpos'] == 2
            
            view_specific_trials[f"A_{condition}"] = condition_data.index[view_a_mask].tolist()
            view_specific_trials[f"B_{condition}"] = condition_data.index[view_b_mask].tolist()
        
        return view_specific_trials
    
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        return None
    

class TrialSplitter:
    """Split trials randomly for balanced comparisons"""
    
    def __init__(self, base_filter: TrialFilter, n_splits: int = 3, 
                 split_conditions: Optional[List[str]] = None, seed: int = 0):
        self.base_filter = base_filter
        self.n_splits = n_splits
        self.split_conditions = split_conditions or []
        self.seed = seed
        
    def get_conditions(self) -> List[str]:
        base_conditions = self.base_filter.get_conditions()
        conditions = []
        
        for cond in base_conditions:
            if cond in self.split_conditions:
                conditions.extend([f"{cond}_{i+1}" for i in range(self.n_splits)])
            else:
                conditions.append(cond)
                
        return conditions
    
    def filter_trials(self, behavior: pd.DataFrame) -> Dict[str, List[int]]:
        base_trials = self.base_filter.filter_trials(behavior)
        split_trials = {}
        
        for condition, trial_indices in base_trials.items():
            if condition in self.split_conditions:
                # Split this condition
                splits = self._split_trials_randomly(trial_indices, self.n_splits, self.seed)
                for i, split in enumerate(splits, 1):
                    split_trials[f"{condition}_{i}"] = split
            else:
                # Keep as is
                split_trials[condition] = trial_indices
        
        return split_trials
    
    def _split_trials_randomly(self, trial_indices: List[int], n_splits: int, seed: int) -> List[List[int]]:
        """Split trials randomly into n groups"""
        random.seed(seed)
        trials = trial_indices.copy()
        random.shuffle(trials)
        
        split_size = len(trials) // n_splits
        splits = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            if i == n_splits - 1:  # Last split gets remaining trials
                splits.append(trials[start_idx:])
            else:
                splits.append(trials[start_idx:start_idx + split_size])
        
        return splits
    
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        return None
    

# Training filters

class TrainingFilter(Protocol):
    """Protocol for training task filters that handle mini-blocks"""
    
    def get_conditions(self) -> List[str]:
        """Return list of condition names for training"""
        ...
        
    def process_training_events(self, events: pd.DataFrame) -> Dict[str, Tuple[List[float], List[float]]]:
        """Process events into mini-blocks, return dict mapping conditions to (onsets, durations)"""
        ...
    
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        """Generate contrasts for training conditions (optional)"""
        ...
        

class WideNarrowTrainingFilter:
    """Training filter for wide/narrow mini-blocks (Experiment 2)"""
    
    def get_conditions(self) -> List[str]:
        return ['wide', 'narrow']
    
    def process_training_events(self, events: pd.DataFrame) -> Dict[str, Tuple[List[float], List[float]]]:
        """Process events into shape-based mini-blocks"""
        miniblocks = self._create_miniblocks(events)
        
        wide_blocks = miniblocks[miniblocks['widenarr'] == 'wide']
        narrow_blocks = miniblocks[miniblocks['widenarr'] == 'narrow']
        
        return {
            'wide': (list(wide_blocks['onset']), list(wide_blocks['duration'])),
            'narrow': (list(narrow_blocks['onset']), list(narrow_blocks['duration']))
        }
        
    def _create_miniblocks(self, events: pd.DataFrame) -> pd.DataFrame:
        """Create mini-blocks from individual events"""
        events = events.copy()
        events['rotation'] = events['rotation'].astype(str)
        
        # Get only stimulus events (no button presses)
        stimonly = events[events['trial_type'] != 'buttonpress'].reset_index(drop=True)
        mb_length = 9  # Number of events per mini-block
        
        miniblocks = []
        
        for i in range(0, len(stimonly), mb_length):
            if i + mb_length > len(stimonly):
                break  # Skip incomplete mini-blocks
            
            first_event = stimonly.iloc[i]
            last_event = stimonly.iloc[i + mb_length - 1]
            
            # Calculate mini-block duration
            duration = (last_event['onset'] + last_event['duration']) - first_event['onset']
            
            # Determine wide/narrow based on first event in block
            if first_event['trial_type'] == 'object':
                if ((first_event['view'] == 'A' and first_event['rotation'] == '30') or 
                    (first_event['view'] == 'B' and first_event['rotation'] == '90')):
                    widenarr = 'wide'
                elif ((first_event['view'] == 'B' and first_event['rotation'] == '30') or 
                      (first_event['view'] == 'A' and first_event['rotation'] == '90')):
                    widenarr = 'narrow'
                else:
                    continue  # Skip if cannot categorize
            else:
                continue  # Skip non-object blocks
            
            miniblocks.append({
                'onset': first_event['onset'],
                'duration': duration,
                'trial_type': first_event['trial_type'],
                'widenarr': widenarr
            })
            
            return pd.DataFrame(miniblocks)
        
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        return [
            ('wide>narrow', 'T', ['wide', 'narrow'], [1.0, -1.0]),
            ('narrow>wide', 'T', ['narrow', 'wide'], [1.0, -1.0])
        ]


class ViewSpecificWideNarrowTrainingFilter:
    """Training filter for A30, A90, B30, B90 mini-blocks (Experiment 1)"""
    
    def get_conditions(self) -> List[str]:
        return ['A_wide', 'A_narrow', 'B_wide', 'B_narrow']
    
    def process_training_events(self, events: pd.DataFrame) -> Dict[str, Tuple[List[float], List[float]]]:
        """Process events into view x rotation mini-blocks"""
        miniblocks = self._create_miniblocks(events)
        
        condition_data = {}
        for condition in self.get_conditions():
            view, shape = condition.split('_')
            rotation = '30' if (view == 'A' and shape == 'wide') or (view == 'B' and shape == 'narrow') else '90'
            
            blocks = miniblocks[
                (miniblocks['view'] == view) & 
                (miniblocks['rotation'] == rotation)
            ]
            
            condition_data[condition] = (list(blocks['onset']), list(blocks['duration']))
        
        return condition_data
    
    def _create_miniblocks(self, events: pd.DataFrame) -> pd.DataFrame:
        """Create mini-blocks preserving view and rotation info"""
        events = events.copy()
        events['rotation'] = events['rotation'].astype(str)
        
        stimonly = events[events['trial_type'] != 'buttonpress'].reset_index(drop=True)
        mb_length = 18  # Experiment 1 uses 18 events per mini-block
        
        miniblocks = []
        
        for i in range(0, len(stimonly), mb_length):
            if i + mb_length > len(stimonly):
                break
            
            first_event = stimonly.iloc[i]
            last_event = stimonly.iloc[i + mb_length - 1]
            
            duration = (last_event['onset'] + last_event['duration']) - first_event['onset']
            
            miniblocks.append({
                'onset': first_event['onset'],
                'duration': duration,
                'trial_type': first_event['trial_type'],
                'view': first_event['view'],
                'rotation': first_event['rotation']
            })
        
        return pd.DataFrame(miniblocks)
    
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        return None
    

class IndividualMiniblockTrainingFilter:
    """Training filter that creates separate conditions for each mini-block"""
    
    def __init__(self, base_filter: TrainingFilter):
        self.base_filter = base_filter
        
    def get_conditions(self) -> List[str]:
        # This will be determined dynamically based on number of mini-blocks found
        base_conditions = self.base_filter.get_conditions()
        # For now, assume max 10 mini-blocks per condition (adjust as needed)
        conditions = []
        for base_cond in base_conditions:
            for i in range(1, 11):  # Assuming max 10 mini-blocks
                conditions.append(f"{base_cond}_{i}")
        return conditions
    
    def process_training_events(self, events: pd.DataFrame) -> Dict[str, Tuple[List[float], List[float]]]:
        """Create individual conditions for each mini-block"""
        base_data = self.base_filter.process_training_events(events)
        
        individual_conditions = {}
        
        for base_condition, (onsets, durations) in base_data.items():
            for i, (onset, duration) in enumerate(zip(onsets, durations), 1):
                condition_name = f"{base_condition}_{i}"
                individual_conditions[condition_name] = ([onset], [duration])
        
        return individual_conditions
    
    def generate_contrasts(self) -> Optional[List[Tuple[str, str, List[str], List[float]]]]:
        # Individual mini-block models typically don't need contrasts
        return None
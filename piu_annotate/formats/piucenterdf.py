import pandas as pd
from tqdm import tqdm


class PiuCenterDataFrame:
    def __init__(self, csv_file: pd.DataFrame):
        """ Loads data struct from old piucenter d_annotate.
            s3://piu-app/d_annotate/
        """
        self.df = pd.read_csv(csv_file)

    @classmethod
    def from_s3(filename: str):
        """ Alternate constructor: download from s3://piu-app/d_annotate/"""
        raise NotImplementedError()
    
    def get_limb_annotations(self) -> list[str]:
        """ Obtain limb annotation in format for ChartStruct,
            by parsing columns
            {Left foot / Right foot / Left hand / Right hand} {1/2/3/4}
            
            Concatenated string of l (left foot), r (right foot),
            e (either foot), h (either hand), ? (unknown).
            Length must be:
                = Number of non-0 symbols in `Line with active holds`:
                    limb per symbol in same order.
                = 0: (blank) equivalent to ? * (n. non-0 symbols)
        """
        panel_to_idx = {
            'p1,1': 0,
            'p1,7': 1,
            'p1,5': 2,
            'p1,9': 3,
            'p1,3': 4,
            'p2,1': 5,
            'p2,7': 6,
            'p2,5': 7,
            'p2,9': 8,
            'p2,3': 9,
        }
        df = self.df
        panels = set(list(panel_to_idx.keys()))
        limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']
        actions = ['1', '2', '3', '4']
        limb_cols = [f'{limb} {action}' for limb in limbs for action in actions]
        limb_annots = []
        for _, row in tqdm(df.iterrows(), total = len(df)):
            idx_s_tuples = []
            for limb_col in limb_cols:
                if row[limb_col] in panels:
                    panel = row[limb_col]
                    line_idx = panel_to_idx[panel]

                    if 'Left foot' in limb_col:
                        s = 'l'
                    if 'Right foot' in limb_col:
                        s = 'r'
                    if 'hand' in limb_col:
                        s = 'h'
                    
                    idx_s_tuples.append((line_idx, s))
            
            if len(idx_s_tuples) > 0:
                idx_s_tuples = sorted(idx_s_tuples)
                limb_annot = ''.join(t[1] for t in idx_s_tuples)
            else:
                limb_annot = ''
            limb_annots.append(limb_annot)
        return limb_annots
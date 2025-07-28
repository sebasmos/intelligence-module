import numpy as np
import nannyml as nml
import pandas as pd 
from processing.utils import *

def create_drift_detector(selected_column, steps_back, reference_df):
                     calc = nml.UnivariateDriftCalculator(
                        column_names = selected_column,
                        chunk_size = steps_back,# chunk-size can be modified as the input series size if desired
                    )
                     return  calc.fit(reference_df)
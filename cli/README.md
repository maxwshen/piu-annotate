# CLI

### Main workflow
- ingest: 
- limbuse: Predict which limbs (left foot, right foot) to use on arrows
- segment: Perform segmentation on stepcharts
- difficulty: Scripts for training stepchart/segment difficulty prediction model, and running predictions
- chartjson: Scripts for converting ChartStruct to chartjson for web app visualization, and scripts for preparing for visualization


### Miscellaneous
- analysis: Scripts that extract data from stepcharts, typically outputting a CSV, for downstream data analysis typically in jupyter notebooks
- debug: Scripts for manual debugging, to improve internal code, or to catch likely errors in limb annotations
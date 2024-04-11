# CFST_af_prediction
Datasets and codes for "Real-time axial force prediction of CFST columns under real fire based on modular artificial intelligence"

**dataset:**
temperature field prediction module:
There are 585 cases of CFST columns under realistic fire conditions, considering 5 different column widths, 9 distinct heating curves, 4 modes of fire exposure, and 3 rotations for each non-uniform fire condition. Every case has a compilation of 97 slices.
Since the dataset is too large to be uploaded to GitHub, readers can access the dataset via Google Drive: https://drive.google.com/file/d/1w0iO9UlxXS3DZcEMLBq1hx0_U1ZxZmbl/view?usp=drive_link

**axial force prediction module:**
There are 180 cases of CFST columns under realistic structural conditions, considering 3 different column widths, 4 load ratios, 5 axial restraint stiffness ratios, and 3 slenderness. The sliding window approach causes that each segment spanned 20 minutes, yielding 58 segments per case. Totally, the dataset comprised 10,440 data samples.

**framework:**
temperature field prediction module:
Code, loss result and the best model for the temperature field prediction module.

**axial force prediction module:**
Code, loss result and the best model for the axial force prediction module.

**MAI:**
Code and the best model for the modular artificial intelligence.

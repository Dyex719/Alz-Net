# Biomedical Image Processing and Classification
### The aim of this project is to identify whether a person is suffering from Alzheimer from the persons MRI scan.

## Dependencies
1. Install the python dependencies using the command `pip install numpy scipy nibabel matplotlib pprint pandas tensorflow`

## Usage
The directory structure will be very similar to the popularly used BIDS data format:

```
.
+-- OASIS
    +-- OAS1_PATIENT-ID_MRy ( where y represents
an incrementing number to reflect the imaging visit number for the subject)
        +-- FSL_SEG
        +-- PROCESSED
            +-- MPRAGE
                +-- SUBJ_111
                    +-- OAS1_PATIENT-ID_MRy_mpr_n4_sbj_<no.>.hdr
                    +-- OAS1_PATIENT-ID_MRy_mpr_n4_sbj_<no.>.img        
        +-- RAW
        +-- OAS1_PATIENT-ID_MRy.txt
        +-- OAS1_PATIENT-ID_MRy.xml
```
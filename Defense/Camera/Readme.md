# Perspective Shift Attack Detection -- Camera

This code can be used to identify whether a perspective shift happened on captured camera data.

_Hint:_ This tool is specific to the mounting position of the camera in a vehicle. It was tested of a standard front-facing camera. Example data (benign and PSA) is available at [example_data](./example_data/).

## Requirements
* Python >= 3.9
* The following Python packages:
    * OpenCV, Numpy, MatplotLib: `pip install opencv-python numpy matplotlib`
* A semantically segmented image, using [OneFormer](https://huggingface.co/spaces/shi-labs/OneFormer) (configurtion: semantic task; COCO model; DiNAT-L backbone)

## Syntax

```bash
python engineHood_Detection.py <semantically segmented oneformer image>
```
Parameters:
- `semantically segmented oneformer image`: Semantically segmented image, using [OneFormer](https://huggingface.co/spaces/shi-labs/OneFormer) (configurtion: semantic task; COCO model; DiNAT-L backbone)

Example:

```bash
python engineHood_Detection.py example_data/psa.png
```

## Output 

The tool will print debugging information (amount of detected pixels, example coordinates and approximated quadratic coefficients). _If_ the image is detected to be adversarially shifted, it will additionally print **adversarial image**. Additionally the image and identified coordinates will be plotted and displayed.
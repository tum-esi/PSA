# Perspective Shift Attack Simulation -- Camera

This code can be used to create a digital simulation of our perspective shift attack against cameras.

## Requirements
* Python >= 3.9
* The following Python packages:
    * OpenCV, Numpy: `pip install opencv-python numpy`

## Syntax

```bash
python PSAcameraSimulation.py <spherical image> <PSA direction> [<x0,y0>]
```
Parameters:
- `spherical image`: Path to the spherical image used. The image must be captured as a dual-fisheye representation of a spherical image. An example image is available at [`./example_data/spherical.jpg`](./example_data/spherical.jpg)
- `<PSA direction>`: Simulate the direction of the perspective shift. Possible values are one of: `up`, `down`, `left`, or `right`
- `<x0,y0>` (_Optional_): Optionaly parameter of the center point. If none is given, the user can select via a GUI.

Example:

```bash
python PSAcameraSimulation.py example_data/spherical.jpg left
```

## Output 

Folder called **`POV`**, containing the curvilinear projection of the benign image around the selected center point and with the selected perspective shift.

## Modifications

* Changing the angle of the perspective shift:
    * In [`PSAcameraSimulation`](PSAcameraSimulation.py) (line 15): Change the value of `PSA_DEGREE` (default: 20.0)
* Changing the FoV to emulate different cameras:
    * Different camera FoVs are available: [IMX 708 Wide](https://www.raspberrypi.com/documentation/accessories/camera.html#camera-module-3), [Bosch multi purpose camera](https://www.bosch-mobility.com/en/solutions/camera/multi-purpose-camera/), and [Aumovio multi function mono camera MFC525](https://amv.re/I38AYQ).
    * To change the desired camera: In [`PSAcameraSimulation`](PSAcameraSimulation.py) (line 14): Change the value of `CAMERA_TO_USE` (default: 'IMX708Wide')
    * To add further cameras: In [`PSAcameraSimulation`](PSAcameraSimulation.py) (line 18): Add your camera to the dict `CAMERA_FOVS` in the format: `<Name>: (<horizontal fov>, <vertical fov>)`
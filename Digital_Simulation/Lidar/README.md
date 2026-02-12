# Digital Simulation Tool For PSA effect evaluation on the KITTI odometry Dataset

## Requirements:

### Kitti Odometry Dataset

You can download it at the following link: [link](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

### Install the requirements file

```python
pip install -r ../../requirements_lidar.txt
```

## Execute the code

### Before Execution

```python
data_root = os.path.join('<root dyrectory where the KITTI data are saved>')
```

### Execute

```python
python odomSym.py
```

## Organization of the results

For every sequence that is fed to the KISS-ICP pipeline, there will be 3 evaluations aved in the following order:

- Reference: The PCD will be only cropped, and it will serve as reference for the comparison
- Left shift 20 degrees
- Right shift 20 degrees

Moreover, for each of the run, the program will generate a preview to make sure that everything is in line with the expectations saved under

```bash
/plots/preview

```

## Visualization of the results

If not installed, install evo

```bash
pip install evo
```

When installed, then it is possible to run the visual comparison

```bash
evo_traj kitti --plot '<left_kitti.txt>'  '<right_kitti.txt>' '<reference_kitti.txt>' --plot_mode=zx -c 'plot_config.json'
```

To have an immediate example, you can run the command with the data included in this repo. The first will be colored green , the second blue and the third one red. This is deifned by the evo traj, and has no specific meaning. 


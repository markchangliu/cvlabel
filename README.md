# cvlabel

A toolbox for converting, evaluating, and visualizing detection/segmentation labels.

## Installation

```
# Uninstall old versions
pip uninstall cvlabel
rm -fr ./build

# Select one of the versions below, you can not install both

# Install GUI version
pip install -e .[gui] --config-settings editable_mode=strict

# Install NON-GUI version
pip install -e .[no_gui] --config-settings editable_mode=strict
```
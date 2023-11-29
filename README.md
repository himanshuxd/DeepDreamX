# DeepDreamX

DeepDreamX is stanalone script I have developed to perfrom Inceptionism, it is also acts as a graphical user interface (GUI) that facilitates standalone deep dream image processing, via one click image selection and result viewing. DeepDream, originally a project by Google, leverages the Caffe library developed by UC Berkeley to perform a Machine Learning technique known as Inceptionism.

## Overview
DeepDreamX provides a user-friendly interface to apply the deep dream algorithm to images, creating visually fascinating and dream-like renditions. The project builds upon Google's original deep dream concept and aims to make the process more accessible through a Python-based GUI image selection and also saving the original image and inception to project root folder with a generated `result_viewer.html` for easy observation.

## Usage
### 1. Clone the repository:

`git clone https://github.com/himanshuxd/DeepDreamX.git`

### 2. Install dependencies:

`pip install -r requirements.txt`

### 3. Run the GUI:

`python dreamify.py`

### 4. Open an image and enjoy the mesmerizing results.
To enhance the dream-like features of the images, you can tweak the `iterations` parameter in the Python script. The `iterations` variable in the existing script controls how many gradient ascent steps are taken during the dreaming process. For example

    # Parameters for the inception process
    step = 0.01  # Gradient ascent step size
    iterations = 20  # Number of ascent steps per scale can be set to 50, 100 etc
    max_loss = 15.0


## Resources to learn more

- [Advanced Guide to Inception v3](https://cloud.google.com/tpu/docs/inception-v3-advanced)

- [DeepDream - a code example for visualizing Neural Networks](https://ai.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html)

- [Inceptionism: Going Deeper into Neural Networks](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)


## Contributing
Contributions are welcome! If you have ideas for improvements or encounter issues, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

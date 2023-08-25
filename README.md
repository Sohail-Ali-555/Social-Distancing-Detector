# Social-Distancing-Detector
A Python program that processes input video for human detection and checks whether observed pedestrians / people are following minimum social distancing rule or not.  
## Installing pre-requisite Python libraries
To install the modules, download the [**requirements.txt**](https://github.com/Sohail-Ali-555/Social-Distancing-Detector/blob/main/requirements.txt) file and open command prompt or powershell window in the same directory and type-in the following command:
**pip install -r requirements.txt**

Note: Modules like NumPy, SciPy, OpenCV, Imutils, Pygame are chosen for their latest versions as of this commit date, since latest versions will generally offer max. possible performance speed.

Another thing to be noted is that, by default the program will use native OpenCV module for Python but if you are having Nvidia CUDA supported GPU in your system then might try to use it for increased performance (FPS) by installing and using OpenCV build with CUDA support, which will be needed to be installed manually from outside.
A link of a useful [**Youtube video**](https://www.youtube.com/watch?v=YsmhKar8oOc&t=463s) demonstrating the complete download and install process is here.

## Download Links for YOLO weights file
Since GitHub doesn't allows to have upload(s) of single file beyond 100MB size (plus I don't want to fill up my free GitHub account space), I am providing the G-Drive download links of both [**yolov3.weights**](https://drive.google.com/file/d/1P1SMncvkgbFfPwPs-TOn2pw39lzG1GCV/view?usp=drive_link) and [**yolov3_tiny.weights**](https://drive.google.com/file/d/1W5XsEmBw3HQ5r-ebwCjRIPTgwEcPo5dt/view?usp=drive_link) file(s). Download and move them to your yolo-coco directoy before running the program.  

## Running the program
The entire Python program is contained within the single [**main.py**](https://github.com/Sohail-Ali-555/Social-Distancing-Detector/blob/main/main.py) file. It can either be run by first opening command prompt or powershell window in the directory and typing the following command: **python .\main.py**
or by opening the main.py file through any IDE like VS Code or PyCharm and running it from there.
Rest of the instructions are explained as comments in the program file.

## Program Screenshots
![image](https://github.com/Sohail-Ali-555/Social-Distancing-Detector/assets/103688890/926d682f-9b2a-499e-b37f-fdb69ecde30d)
Original pre-recorded CCTV footage.
<br><br>
![image](https://github.com/Sohail-Ali-555/Social-Distancing-Detector/assets/103688890/ffcd7319-1f8d-4687-8fe3-2ce0dada0159)
Processed Image/Frame when number of social distancing violations are lower than threshold/safety limit.
<br><br>
![image](https://github.com/Sohail-Ali-555/Social-Distancing-Detector/assets/103688890/b248c566-f0d5-4361-97aa-f4437b98586e)
Processed Image/Frame, here number of violations exceed threshold/safety limit. 

![image](https://github.com/Sohail-Ali-555/Social-Distancing-Detector/assets/103688890/b248c566-f0d5-4361-97aa-f4437b98586e){width=250}

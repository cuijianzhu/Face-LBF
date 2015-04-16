FaceX-LBF
=====

A open source face landmarks detector (face alignment) using linear binary feature.

This project is based on the previous work of Face-X.

Notice
====

This detector is coded in Visual C++ 2013 Express for Desktop, and uses OpenCV. It also works in GCC. I believe it will work on other compilers with minor or even no modifications. The most important thing is linking the OpenCV library correctly. Currently it use these modules: core, highgui, imgproc, objdetect. Also notice that it uses some C++ 11 features, so be sure your compiler is up to date.

When you try the detector code, make sure the three file haarcascade_frontalface_alt2.xml model.xml.gz test.jpg are in the current working directory.

When you try the training code, I suggest you compile FaceX-Train as 64bit code, since it may use large amount of memory (it will load all the images into memory).

Currently, the training code is a little messy. I hope I can clean it up someday.

How To Train
====

First collect face images with face area and landmark labels. I recommend you first download dataset from [3] to see if the works correctly. You can check FaceX-Train/train to know how to organize the training data. The name of the label file must be labels.txt, and the format of labels.txt is like this:

image1.png FACE-LEFT FACE-RIGHT FACE-TOP FACE-BOTTOM X1 Y1 X2 Y2 ...

image2.png ...

Then create a config file for training, you can use FaceX-Train/sample_config.txt as a start point. After that, run command:

Face-LBF-Train config.txt

It will take several minutes to several hours, depending on the training-set size and the speed of your computer.

Known Issue
====

1. The program is very slow if you run it in Visual C++ debugger. Even if you use Release Mode. Therefore, run it directly outside (remember to put the three files in the current working directory). It seems Visual C++ debugger will slow down some program greatly.

Reference
====

[1] Cao X, Wei Y, Wen F, et al. Face alignment by explicit shape regression[J]. International Journal of Computer Vision, 2014, 107(2): 177-190.

[2] http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

[3] http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm

[4] Ren, S., Cao, X., Wei, Y., Sun, J.: Face Alignment at 3000 FPS via Regressing Local Binary Features. In: Computer Vision and Pattern Recognition (2014)
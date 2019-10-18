Install Jupyter, matplotlib and opencv-python, and open mrz-lab.ipynb

Development notes:

On First step we need get mask of brackets from image. It looks like a segmentation task. Let's try to train UNET model, and see how it works


Unet takes a lot of time to train. Testing opencv methods. With matchTemplate function we can solve Second step, without first at all. It works fast, but doesn't work well for test images, but it's not a scale invariant.
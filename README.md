# Processing of Calligraphy Text to Digital Text


** Handwriting Recognition** (HWR) is also known as Handwritten Text Recognition (HTR) is the ability to transcribe the handwritten text into digital text. The attributes are Online Text Recognition and Offline Text Recognition. In Online text Recognition, the information’s are gathered thought real-time data provided by the real-time writing sensors [i.e., Digitizing tablet]. On the other hand, Offline Text Recognition gathers existing information like images and process them based on two techniques. Firstly, Typed Text is used to convert the unreadable existing images into a readable format. Secondly, Handwritten Text uses manual handwritten notes as an input image and process them into a human-readable format. Earlier, before the development of Deep Learning, the HTR uses Hidden Markov Model (HMM) for processing the image into a readable format. Although HMM is very efficient, it cannot be applied on pointwise nonlinearity to the output at every timestep, this being the main reason to opt Convolutional Recurrent Neural Network (CRNN) and Connectionist Temporal Classification (CTC) techniques. In Figure 1.1, the input image is taken from the IAM dataset, and the techniques are being applied to the input image and the following transcript is obtained as output.



 
